import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
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
        self.poseSub = self.create_subscription(PoseStamped, "forearm_pose_camera", self.correction, 1)
        
        # Set up publishers
        self.posePub = self.create_publisher(PoseWithCovarianceStamped, "handPose", 1)

        self.timer = self.create_timer(0.2, self.pubPose)


    def configureParameters(self):

        # Set default and grab initial state
        defaultState = np.zeros((4,4))
        XDesc = ParameterDescriptor(
            description="Initial State, SE(3) matrix (4x4)",
            read_only=False
        )
        self.declare_parameter("initial_state", defaultState.flatten().tolist(), XDesc)
        self.X = np.asarray(self.get_parameter("initial_state").value).reshape((4,4))

        # Set default and grab initial covariance
        defaultP = np.zeros((6, 6))
        PDesc = ParameterDescriptor(
            description="Initial State covariance (6x6)",
            read_only=False
        )
        self.declare_parameter("initial_covariance", defaultP.flatten().tolist(), PDesc)
        self.P = np.asarray(self.get_parameter("initial_covariance").value).reshape((6,6))

        # Set default and grab process noise covariance
        defaultQ = np.zeros((6, 6))
        QDesc = ParameterDescriptor(
            description="Process Noise Covariance (6x6)",
            read_only=False
        )
        self.declare_parameter("process_noise", defaultQ.flatten().tolist(), QDesc)
        self.Q = np.asarray(self.get_parameter("process_noise").value).reshape((6,6))

        # Set default and grab measurement noise covariance
        defaultN = np.zeros((6, 6))
        NDesc = ParameterDescriptor(
            description="Measurement Noise Covariance (6x6)",
            read_only=False
        )
        self.declare_parameter("measurement_noise", defaultN.flatten().tolist(), NDesc)
        self.N = np.asarray(self.get_parameter("measurement_noise").value).reshape((6,6))

        # Set Measurement Jacobian (constant since this is invariant EKF)
        defaultH = np.eye(6)
        HDesc = ParameterDescriptor(
            description="Measurement Jacobian (6x6)",
            read_only=False
        )
        self.declare_parameter("measurement_jacobian", defaultH.flatten().tolist(), HDesc)
        self.H = np.asarray(self.get_parameter("measurement_jacobian").value).reshape((6,6))

        # Get measured gravity vector from a flat surface
        defaultGrav = np.asarray([0, 0, -9.81])
        gravDesc = ParameterDescriptor(
            description="Gravity vector to subtract for accelerometer data. Measure from IMU.",
            read_only=False
        )
        self.declare_parameter("gravity_vector", defaultGrav.tolist(), gravDesc)
        self.g = np.asarray(self.get_parameter("gravity_vector").value).reshape((3,1))


    # SE(3) Adjoint definition (Double check this)
    def Adj(self, X):
        R = X[0:3, 0:3]
        AdjX = block_diag(R, R, R)  #changed
        AdjX[3:6, 0:3] = self.skew(X[0:3, 3]) @ R
        AdjX[6:9, 0:3] = self.skew(X[0:3, 4]) @ R
        return AdjX


    # Wedge function, assumes that eta is a flattened 6 element array
    # Assuming angular first, then translation
    def wedge(self, eta):
        w = np.zeros((4,4))
        w[0:3, 0:3] = self.skew(eta[0:3])
        w[0:3, 3] = np.asarray(eta[3:6]).T
        return w
    
    # Vee function, assumes that xi is a 4x4 array 
    def vee(self, xi):
        w = self.unskew(xi[0:3, 0:3]).flatten()
        v = xi[0:3, 3].flatten()
        return np.hstack([w, v])

    # Adjoint for SO(3) is just the rotation matrix itself
    def Ad_SO3(self, R):
        return R

    # Adjoint for SE(3)
    def Ad_SE3(self, X):
        R = X[0:3, 0:3]
        p = X[0:3, 3]
        AdX = np.zeros((6, 6))
        AdX[0:3, 0:3] = R
        AdX[3:6, 3:6] = R
        AdX[3:6, 0:3] = self.skew(p) @ R
        return AdX

    # Converts a vector v (3x1) to a skew symmetric matrix (3x3)
    def skew(self, v):
        return np.asarray([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])


    # Converts a skew symmetric matrix (3x3) to a vector (3x1)
    def unskew(self, w):
        return np.asarray([w[2,1],w[0,2],w[1,0]]).reshape((3,1))


    # Computes the discrete noise covariance
    def getQd(self, dt):
        Ad = self.Ad_SE3(self.X)
        return Ad @ (self.Q * dt) @ Ad.T


    # Propagate orientation only using the IMU data
    def prediction(self, msg):

        # Grab the time since the last IMU message
        currTime = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        if self.lastMeas == 0: # Handle first run
            self.lastMeas = currTime
            return
        dt = currTime - self.lastMeas
        self.lastMeas = currTime

        # Propagate the state and update with left invariant form
        u = np.hstack([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z, np.zeros(3)]) * dt
        self.X = self.X @ expm(self.wedge(u)) 
        # self.X = self.X @ (np.eye(4) + self.wedge(u)) # Left invariant Euler integration, should be fine for small dt   

        # Propagate the covariance using the discrete noise covariance, state transition matrix is Identity
        # TODO: Verify this, we might need to compute the state transition matrix using the adjoint
        self.P = self.P + self.Q

        # print(f"--- Covariance Propagated with Q ---")
        # print(np.round(self.P, 3))

        # # Get the position and orientation for printing
        # pos = self.X[0:3, 3].flatten()
        # rpy = R.from_matrix(self.X[0:3, 0:3]).as_euler('xyz', degrees=True)
        # state_str = (
        #     f"\n--- SE_2(3) Prediction ---\n"
        #     f"Pos [m]   : {np.round(pos, 3)}\n"
        #     f"Ori [deg] : {np.round(rpy, 3)}\n"
        #     f"--------------------------"
        # )
        # print(state_str)


    # Correction step (callback on pose from pose tracker)
    def correction(self, msg):
    
        # Set it to work for both Pose and PoseStamped
        pose_msg = msg.pose if hasattr(msg, "pose") else msg

        # Calculate innovation
        v = self.vee(logm(np.linalg.inv(self.X) @ self.poseToSE3(pose_msg)))

        AdInv = self.Ad_SE3(np.linalg.inv(self.X))
        N_local = AdInv @ self.N @ AdInv.T

        # Calculate innovation matrix, transform measurement noise into the lie algebra, calculate kalman gain
        S = self.H @ self.P @ self.H.T + N_local
        L = self.P @ self.H.T @ np.linalg.inv(S)

        # Apply correction to state and covariance
        delta = L @ v
        self.X = self.X @ expm(self.wedge(delta))
        I = np.eye(6)
        self.P = (I - L @ self.H) @ self.P @ (I - L @ self.H).T + L @ N_local @ L.T

        # # Print
        # rpy = R.from_matrix(self.X[0:3, 0:3]).as_euler('xyz', degrees=True)
        # print(f"--- Correction Applied ---")
        # print(f"Pos: {np.round(self.X[0:3, 3], 3)}")
        # print(f"Ori: {np.round(rpy, 3)}")


    def poseToSE3(self, pose):
        T = np.eye(4)
        T[0:3, 3] = np.asarray([pose.position.x, pose.position.y, pose.position.z])
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        T[0:3, 0:3] = R.from_quat(q).as_matrix()
        return T
    
    def pubPose(self):
        
        pose = PoseWithCovarianceStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        q = R.from_matrix(self.X[0:3, 0:3]).as_quat()
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]

        p = self.X[0:3, 3].flatten()
        pose.pose.pose.position.x = p[0]
        pose.pose.pose.position.y = p[1]
        pose.pose.pose.position.z = p[2]

        pose.pose.covariance = self.P.flatten().tolist()

        self.posePub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    pose_filter = InEKF()
    rclpy.spin(pose_filter)
    pose_filter.destroy_node()
    rclpy.shutdown()