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


    def calculate_qd(self, Phi, dt):
        """
        Calculates the discrete-time noise covariance matrix Qd (9x9) 
        using the pre-defined continuous noise matrix self.Q.
        
        :param Phi: The 9x9 State Transition Matrix (I + A*dt)
        :param dt: Time step
        :return: 9x9 numpy array Qd
        """
        # self.Q is assumed to be your continuous-time covariance matrix (9x9)
        # Qd ≈ Phi * (self.Q * dt) * Phi^T
        
        # We multiply self.Q by dt to convert from power spectral density 
        # to discrete covariance for this specific interval.
        Qd = Phi @ (self.Q * dt) @ Phi.T
        
        return Qd

    # Wedge function, assumes that eta is a flattened 9 element array
    # Returns a 5x5 array
    # eta: [wx, wy, wz, vx, vy, vz, ax, ay, az], 
    def wedge(self, eta):
        w = np.zeros((5,5))
        w[0:3, 0:3] = self.skew(eta[0:3])
        w[0:3, 3] = np.asarray(eta[6:9]).T
        w[0:3, 4] = np.asarray(eta[3:6]).T
        # w[3, 4] = 1 # Gemini suggests we remove
        return w


    # Converts a vector v (3x1) to a skew symmetric matrix (3x3)
    # def skew(self, v):
    #     return np.asarray([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])


    # Converts a skew symmetric matrix (3x3) to a vector (3x1)
    def unskew(self, w):
        return np.asarray([w[2,1],w[0,2],w[1,0]]).reshape((3,1))


    # Converts 5x5 wedge to 9x1 twist
    def vee(self, omega):
        w = self.unskew(omega[0:3, 0:3]).flatten()
        # Previous mapping (kept for reference):
        # a = omega[0:3, 3].flatten()
        # v = omega[0:3, 4].flatten()
        # return np.hstack([w, a, v])

        # Keep vee consistent with wedge():
        # wedge([w, v, a]) puts a in col 3 and v in col 4, so vee must output [w, v, a].
        v = omega[0:3, 4].flatten()
        a = omega[0:3, 3].flatten()
        return np.hstack([w, v, a])

    def skew(self, vector):
        """Square-to-skew-symmetric matrix mapping."""
        return np.array([
            [0,         -vector[2],  vector[1]],
            [vector[2],  0,         -vector[0]],
            [-vector[1], vector[0],  0]
        ])

    def wedge(self, xi):
        phi = xi[0:3]
        alpha = xi[3:6]
        beta = xi[6:9]
        W = np.zeros((5, 5))
        W[0:3, 0:3] = self.skew(phi)
        W[0:3, 3] = alpha
        W[0:3, 4] = beta
        return W


    def get_transition_matrix(self, msg, dt):
        """
        Computes the State Transition Matrix Phi (9x9) for SE2(3).
        
        :param msg: sensor_msgs/Imu message
        :param dt: Time step
        :return: 9x9 numpy array Phi
        """
        # 1. Extract raw IMU rates
        omega = np.array([msg.angular_velocity.x, 
                        msg.angular_velocity.y, 
                        msg.angular_velocity.z])
        
        accel = np.array([msg.linear_acceleration.x, 
                        msg.linear_acceleration.y, 
                        msg.linear_acceleration.z])

        # 2. Create Skew-Symmetric Matrices
        def skew(v):
            return np.array([
                [0,    -v[2],  v[1]],
                [v[2],  0,    -v[0]],
                [-v[1], v[0],  0]
            ])

        Om = skew(omega)
        Ac = skew(accel)
        I3 = np.eye(3)
        Z3 = np.zeros((3, 3))

        # 3. Construct the Continuous-Time A matrix (9x9)
        # Row 1: Orientation error dynamics
        # Row 2: Velocity error dynamics
        # Row 3: Position error dynamics
        A = np.block([
            [-Om,  Z3,  Z3],
            [-Ac, -Om,  Z3],
            [ Z3,  I3, -Om]
        ])

        # 4. Discretize: Phi = I + A*dt 
        # (Higher order terms (A^2 * dt^2 / 2) can be added for more precision)
        Phi = np.eye(9) + A * dt
        
        return Phi


    def prediction(self, msg):
        # 1. Timing and Basic IMU parsing
        currTime = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
        if self.lastMeas == 0: # Handle first run
            self.lastMeas = currTime
            return
        dt = currTime - self.lastMeas
        self.lastMeas = currTime




        # 2. Extract current state components
        # X = [ R  v  p ]
        #     [ 0  1  0 ]
        #     [ 0  0  1 ]
        R_curr = self.X[0:3, 0:3]
        v_world = self.X[0:3, 3]
        
        # 3. Prepare the SE2(3) Lie Algebra vector (xi)
        # All components must be displacements in the BODY frame
        omega = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])

        # Rotate Global Gravity into the Body Frame
        # This is what the IMU 'feels' while static
        g_body = R_curr.T @ self.g.flatten()

        # 2. Subtract gravity from the IMU measurement to get 'true' linear acceleration
        # This leaves you with only the movement acceleration
        accel_pure = accel - g_body
        
        phi = omega * dt              # Rotation displacement (rad)
        alpha = accel_pure * dt            # Velocity displacement (m/s)
        beta = (R_curr.T @ v_world) * dt  # Position displacement (m) - Rotate world v to body frame
        
        # Construct the 9D vector for the wedge
        xi = np.hstack([phi, alpha, beta])
        V_matrix = self.wedge(xi) 

        # 4. State Propagation (Group Update)
        # We do NOT multiply by dt here because it is already baked into V_matrix
        self.X = self.X @ expm(V_matrix)

        # 5. Covariance Propagation
        # For LI-EKF, we use the Adjoint of the increment (exp(xi)) or the system Jacobian
        # A simplified approach for covariance:
        Phi = self.get_transition_matrix(msg, dt) # See note below
        self.P = Phi @ self.P @ Phi.T + self.calculate_qd(Phi, dt)

        # 6. Global Gravity Correction
        # Gravity acts in the world frame, so we apply it after the body-frame IMU update
        # g_vec = self.g.flatten()
        # self.X[0:3, 3] += g_vec * dt            # Update velocity
        # self.X[0:3, 4] += 0.5 * g_vec * (dt**2) # Update position

        # 7. Monitoring/Logging
        rot_matrix = self.X[0:3, 0:3]
        vel = self.X[0:3, 3]
        pos = self.X[0:3, 4]
        rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)

        state_str = (
            f"\n--- SE_2(3) Prediction ---\n"
            f"Pos [m]   : {np.round(pos, 3)}\n"
            f"Vel [m/s] : {np.round(vel, 3)}\n"
            f"Ori [deg] : {np.round(rpy, 3)}\n"
            f"--------------------------"
        )
        print(state_str)

    
    # Correction step (callback on pose from pose tracker)
    def correction(self, msg):
    
        # Set it to work for both Pose and PoseStamped
        pose_msg = msg.pose if hasattr(msg, "pose") else msg

        # Grab the measured orientation/position as a rotation matrix/vector
        R_meas = R.from_quat([
            pose_msg.orientation.x, pose_msg.orientation.y, 
            pose_msg.orientation.z, pose_msg.orientation.w
        ]).as_matrix()
        p_meas = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])

        # Calculate the innovation
        R_state = self.X[0:3, 0:3]
        p_state = self.X[0:3, 4]
        delta_R = R_state.T @ R_meas # change in orientation from state to measurement on group
        v_phi = R.from_matrix(delta_R).as_rotvec() # Log map of rotation error gives us the innovation in the Lie algebra (3x1)
        v_p = R_state.T @ (p_meas - p_state) # Position innovation
        v = np.hstack([v_phi, v_p]) # Combined innovation vector

        # Jacobian (6x9 to remove observation of velocity)
        H_pose = np.zeros((6, 9))
        H_pose[0:3, 0:3] = np.eye(3) # Orientation error
        H_pose[3:6, 6:9] = np.eye(3) # Position error

        N_pose = self.N[0:6, 0:6] # Measurement noise for pose

        S = H_pose @ self.P @ H_pose.T + N_pose
        L = self.P @ H_pose.T @ np.linalg.inv(S) # Kalman Gain (9x6)

        delta_xi = L @ v # 9x1 correction in the Lie algebra

        self.X = self.X @ expm(self.wedge(delta_xi))
        
        # Correct covariance
        I = np.eye(9)
        self.P = (I - (L @ H_pose)) @ self.P @ (I - (L @ H_pose)).T + L @ N_pose @ L.T

        # Print
        rpy = R.from_matrix(self.X[0:3, 0:3]).as_euler('xyz', degrees=True)
        print(f"--- Correction Applied ---")
        print(f"Pos: {np.round(self.X[0:3, 4], 3)}")
        print(f"Ori: {np.round(rpy, 3)}")

    import numpy as np

    def get_ros_pose_covariance(self, cov_9x9: np.ndarray, R_b2w: np.ndarray = None) -> np.ndarray:
        """
        Extracts a 6x6 Pose covariance from a 9x9 SE_2(3) covariance matrix.
        
        Assumed 9x9 state ordering: [Rotation (0:3), Velocity (3:6), Position (6:9)]
        Target 6x6 ROS ordering:    [Position (0:3), Rotation (3:6)]
        
        Args:
            cov_9x9: The 9x9 covariance matrix from the InEKF.
            R_b2w: Optional 3x3 rotation matrix (body to world). If provided, 
                rotates the covariance from the body frame to the world frame.
                
        Returns:
            cov_6x6: A 6x6 numpy array suitable for geometry_msgs/PoseWithCovariance.
        """
        # 1. Extract the relevant 3x3 sub-blocks from the 9x9 matrix
        # (Update these slice indices if your filter uses a different state order)
        sigma_phi_phi = cov_9x9[0:3, 0:3]  # Rotation variance
        sigma_p_p     = cov_9x9[6:9, 6:9]  # Position variance
        sigma_p_phi   = cov_9x9[6:9, 0:3]  # Cross-correlation: Position-Rotation
        sigma_phi_p   = cov_9x9[0:3, 6:9]  # Cross-correlation: Rotation-Position
        
        # 2. Assemble the 6x6 matrix in ROS format (Position first, then Rotation)
        cov_6x6 = np.zeros((6, 6))
        cov_6x6[0:3, 0:3] = sigma_p_p
        cov_6x6[3:6, 3:6] = sigma_phi_phi
        cov_6x6[0:3, 3:6] = sigma_p_phi
        cov_6x6[3:6, 0:3] = sigma_phi_p
        
        # 3. Transform to world frame if a rotation matrix is provided
        if R_b2w is not None:
            # Create a 6x6 block diagonal matrix with R_b2w
            R_6x6 = np.zeros((6, 6))
            R_6x6[0:3, 0:3] = R_b2w
            R_6x6[3:6, 3:6] = R_b2w
            
            # Apply the transformation: R * Sigma * R^T
            cov_6x6 = R_6x6 @ cov_6x6 @ R_6x6.T
            
        return cov_6x6

    
    def pubPose(self):
        
        pose = PoseWithCovarianceStamped()

        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()

        q = R.from_matrix(self.X[0:3, 0:3]).as_quat()
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]

        p = self.X[0:3, 4].flatten()
        pose.pose.pose.position.x = p[0]
        pose.pose.pose.position.y = p[1]
        pose.pose.pose.position.z = p[2]

        cov6x6 = self.get_ros_pose_covariance(self.P, R_b2w=self.X[0:3, 0:3])
        pose.pose.covariance = cov6x6.flatten().tolist()

        self.posePub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    pose_filter = InEKF()
    rclpy.spin(pose_filter)
    pose_filter.destroy_node()
    rclpy.shutdown()