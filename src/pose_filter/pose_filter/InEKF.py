import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose, PoseStamped
import numpy as np
from scipy.linalg import expm, logm, block_diag
from scipy.spatial.transform import Rotation as R
import time

class InEKF(Node):

    
    X = None                    # State, SE_2(3) Homogeneous transform (5x5)
    P = None                    # Covariance (9x9)
    Q = None                    # Process Noise Covariance (9x9)
    N = None                    # Measurement Noise Covariance
    H = None                    # Measurement Jacobian (9x9)
    g = None                    # Gravity vector
    v = np.asarray([0, 0, 0])   # last velocity measurement (assume zero velocity to start)
    lastMeas = 0                # Time of last IMU measurement

    def __init__(self):

        super().__init__("pose_filter")

        # Get parameters
        self.configureParameters()

        # Avg Gravity Vector
        self.accumulated_gravity_vectors = np.array([0.0,0.0, 0.0])
        self.num_gravity_readings = 0
        self.avg_gravity_vector = np.array([0.0,0.0, 0.0])

        # Get starting time
        self.lastMeas = time.time()

        # Set up subscribers
        self.imuSub = self.create_subscription(Imu, "imu/data", self.prediction, 1)
        self.poseSub = self.create_subscription(Pose, "forearm_pose_camera", self.correction, 1)
        
        # Set up publishers
        self.posePub = self.create_publisher(PoseStamped, "handPose", 1)

        self.timer = self.create_timer(0.2, self.pubPose)


    def configureParameters(self):

        # Set default and grab initial state
        defaultState = np.zeros((5,5))
        defaultState[3,4] = 1
        XDesc = ParameterDescriptor(
            description="Initial State, SE_2(3) matrix (5x5)",
            read_only=False
        )
        self.declare_parameter("initial_state", defaultState.flatten().tolist(), XDesc)
        self.X = np.asarray(self.get_parameter("initial_state").value).reshape((5,5))

        # Set default and grab initial covariance
        defaultP = np.zeros((9, 9))
        PDesc = ParameterDescriptor(
            description="Initial State covariance (9x9)",
            read_only=False
        )
        self.declare_parameter("initial_covariance", defaultP.flatten().tolist(), PDesc)
        self.P = np.asarray(self.get_parameter("initial_covariance").value).reshape((9,9))

        # Set default and grab process noise covariance
        defaultQ = np.zeros((9, 9))
        QDesc = ParameterDescriptor(
            description="Process Noise Covariance (9x9)",
            read_only=False
        )
        self.declare_parameter("process_noise", defaultQ.flatten().tolist(), QDesc)
        self.Q = np.asarray(self.get_parameter("process_noise").value).reshape((9,9))

        # Set default and grab measurement noise covariance
        defaultN = np.zeros((9, 9))
        NDesc = ParameterDescriptor(
            description="Measurement Noise Covariance (9x9)",
            read_only=False
        )
        self.declare_parameter("measurement_noise", defaultN.flatten().tolist(), NDesc)
        self.N = np.asarray(self.get_parameter("measurement_noise").value).reshape((9,9))

        # Set Measurement Jacobian (constant since this is invariant EKF)
        defaultH = np.eye(9)
        HDesc = ParameterDescriptor(
            description="Measurement Jacobian (9x9)",
            read_only=False
        )
        self.declare_parameter("measurement_jacobian", defaultH.flatten().tolist(), HDesc)
        self.H = np.asarray(self.get_parameter("measurement_jacobian").value).reshape((9,9))

        # Get measured gravity vector from a flat surface
        defaultGrav = np.asarray([0, 0, -9.81])
        gravDesc = ParameterDescriptor(
            description="Gravity vector to subtract for accelerometer data. Measure from IMU.",
            read_only=False
        )
        self.declare_parameter("gravity_vector", defaultGrav.tolist(), gravDesc)
        self.g = np.asarray(self.get_parameter("gravity_vector").value).reshape((3,1))


    # SE_2(3) Adjoint definition (Double check this)
    def Adj(self, X):
        R = X[0:3, 0:3]
        AdjX = block_diag(R, R, R)  #changed
        AdjX[3:6, 0:3] = self.skew(X[0:3, 3]) @ R
        AdjX[6:9, 0:3] = self.skew(X[0:3, 4]) @ R
        return AdjX


    # Wedge function, assumes that eta is a flattened 9 element array
    # Returns a 5x5 array
    # eta: [wx, wy, wz, vx, vy, vz, ax, ay, az], 
    def wedge(self, eta):
        w = np.zeros((5,5))
        w[0:3, 0:3] = self.skew(eta[0:3])
        w[0:3, 3] = np.asarray(eta[6:9]).T
        w[0:3, 4] = np.asarray(eta[3:6]).T
        w[3, 4] = 1
        return w


    # Converts a vector v (3x1) to a skew symmetric matrix (3x3)
    def skew(self, v):
        return np.asarray([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])


    # Converts a skew symmetric matrix (3x3) to a vector (3x1)
    def unskew(self, w):
        return np.asarray([w[2,1],w[0,2],w[1,0]]).reshape((3,1))


    # Converts 5x5 wedge to 9x1 twist
    def vee(self, omega):
        w = self.unskew(omega[0:3, 0:3]).flatten()
        a = omega[0:3, 3].flatten()
        v = omega[0:3, 4].flatten()
        return np.hstack([w, a, v])  #changed


    # Prediction Step (callback on received IMU message)
    def prediction(self, msg):
        current_accel = np.asarray([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        self.accumulated_gravity_vectors += current_accel
        self.num_gravity_readings += 1
        self.avg_gravity_vector = self.accumulated_gravity_vectors / self.num_gravity_readings
        # Get time since last IMU measurement
        currTime = time.time()
        dt = currTime - self.lastMeas
        self.lastMeas = currTime
    
        # Get linear velocity
        accel = self.subGravity(msg).flatten()
        current_vel = self.X[0:3, 3].flatten()
        velocity = accel * dt + current_vel

        # Construct twist, convert to wedge representation 
        omega = np.asarray([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        a = np.asarray([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        eta = np.hstack([omega, v, a])
        V = self.wedge(eta)

        # Get adjoint and propagate state from IMU readings
        AdjX = self.Adj(self.X)
        self.X = self.X @ expm(V * dt)
        self.P = self.P + AdjX @ self.Q @ AdjX.T

        # --- Human Readable State Output ---------------------------------------------------------- #
        
        # Extract blocks
        rot_matrix = self.X[0:3, 0:3]
        vel = self.X[0:3, 3].flatten()
        pos = self.X[0:3, 4].flatten()

        # Convert rotation to Euler angles (Roll, Pitch, Yaw) in degrees
        rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)

        # Format string to 3 decimal places
        state_str = (
            f"\n--- InEKF Prediction State ---\n"
            f"Position (x,y,z) [m]   : [{pos[0]: 8.3f}, {pos[1]: 8.3f}, {pos[2]: 8.3f}]\n"
            f"Velocity (x,y,z) [m/s] : [{vel[0]: 8.3f}, {vel[1]: 8.3f}, {vel[2]: 8.3f}]\n"
            f"Rotation (r,p,y) [deg] : [{rpy[0]: 8.3f}, {rpy[1]: 8.3f}, {rpy[2]: 8.3f}]\n"
            f"RESIDUAL Linear Accel: [{accel[0]: 8.3f}, {accel[1]: 8.3f}, {accel[2]: 8.3f}]\n"
            f"------------------------------"
        )
        
        # Print to terminal 
        print(state_str)
        

    # Substracts gravity from accelerometer measurements (takes full sensor_msgs/msg/Imu)
    def subGravity(self, msg):
        state = self.X
        rotation = R.from_matrix(state[0:3,0:3]) # Orientation Matrix in R3

        accel = msg.linear_acceleration

        accel_array = np.asarray([accel.x, accel.y, accel.z])
        accel_rotated = rotation.apply(accel_array).reshape((3,1))
        return accel_rotated - self.g
    
    # Correction step (callback on pose from pose tracker)
    def correction(self, msg):
        
        # Convert the pose to homogeneous form
        Y = self.poseToHomogeneous(msg)

        # Calculate Kalman gain, innovation
        S = self.H @ self.P @ self.H.T + self.N
        L = self.P @ self.H.T @ np.linalg.inv(S)   
        v = self.vee(logm(self.X @ np.linalg.inv(Y))).reshape((9,1))
        
        # Correct state
        self.X = self.X @ expm(self.wedge((L @ v).flatten())) # Correct state
        
        # Correct covariance
        I = np.eye(9)
        self.P = (I - (L @ self.H)) @ self.P @ (I - (L @ self.H)).T + L @ self.N @ L.T

    # Takes a geometry_msg/msg/Pose and converts it to a 5x5 homogeneous transform in SE_2(3)
    def poseToHomogeneous(self, msg, velocity=[0, 0, 0]):

        Y = np.eye(5)
        v = np.asarray(velocity).reshape((3,1))
        p = np.asarray([msg.position.x, msg.position.y, msg.position.z]).reshape((3,1))
        q_list = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        orientation = R.from_quat(q_list)
        Y[0:3, 0:3] = orientation.as_matrix()  
        Y[0:3, 3] = v
        Y[0:3, 4] = p

        return Y
    
    def pubPose(self):
        
        pose = PoseStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        q = R.from_matrix(self.X[0:3, 0:3]).as_quat()
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        p = self.X[0:3, 4].flatten()
        pose.pose.position.x = p[0]
        pose.pose.position.y = p[1]
        pose.pose.position.z = p[2]

        self.posePub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    pose_filter = InEKF()
    rclpy.spin(pose_filter)
    pose_filter.destroy_node()
    rclpy.shutdown()