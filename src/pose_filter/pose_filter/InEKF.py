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
        self.poseSub = self.create_subscription(PoseStamped, "forearm_pose_camera", self.correction, 1)
        
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

    
    # # Correction step (callback on pose from pose tracker)
    # def correction(self, msg):
    
    #     print("This is the correction step. Received pose measurement from camera, updating state estimate...")

    #     # Convert the pose to homogeneous form
    #     # Previous lift (kept for reference):
    #     # Y = self.poseToHomogeneous(msg)

    #     # Camera gives pose only, so carry predicted velocity into the lifted measurement
    #     # to avoid injecting an unintended zero-velocity pseudo-measurement.
    #     Y = self.poseToHomogeneous(msg, velocity=self.X[0:3, 3].flatten())

    #     # Calculate Kalman gain, innovation
    #     S = self.H @ self.P @ self.H.T + self.N
        
    #     # COPILOT SUGGESTED Enforce symmetry and add a tiny diagonal jitter for numerical stability.
    #     S = 0.5 * (S + S.T)
    #     S_reg = S + 1e-9 * np.eye(S.shape[0])

    #     PHt = self.P @ self.H.T
    #     # Previous gain (kept for reference):
    #     # L = self.P @ self.H.T @ np.linalg.inv(S)
    #     try:
    #         # Solve L * S_reg = P * H^T without forming S^{-1} explicitly.
    #         L = np.linalg.solve(S_reg.T, PHt.T).T
    #     except np.linalg.LinAlgError:
    #         self.get_logger().warn("Innovation covariance is singular; using pseudo-inverse for Kalman gain.")
    #         L = PHt @ np.linalg.pinv(S_reg)

    #     v = self.vee(logm(self.X @ np.linalg.inv(Y))).reshape((9,1))
        
    #     # Correct state
    #     # Previous correction update (kept for reference):
    #     # self.X = self.X @ expm(self.wedge((L @ v).flatten()))

    #     # Use negative feedback with residual r = Log(X Y^{-1}) in a right-multiplicative update.
    #     self.X = self.X @ expm(self.wedge((-L @ v).flatten()))
        
    #     # Correct covariance
    #     I = np.eye(9)
    #     self.P = (I - (L @ self.H)) @ self.P @ (I - (L @ self.H)).T + L @ self.N @ L.T

    #     #DEBUG
    #     self.X[0:3, 3] = [0, 0, 0]
    #     self.X[0:3, 4] = [0, 0, 0]

    #     # Extract blocks
    #     rot_matrix = self.X[0:3, 0:3]
    #     vel = self.X[0:3, 3].flatten()
    #     pos = self.X[0:3, 4].flatten()

    #     # Convert rotation to Euler angles (Roll, Pitch, Yaw) in degrees
    #     rpy = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)

    #     # Format string to 3 decimal places
    #     state_str = (
    #         f"\n--- InEKF Correction State ---\n"            
    #         f"Position (x,y,z) [m]   : [{pos[0]: 8.3f}, {pos[1]: 8.3f}, {pos[2]: 8.3f}]\n"
    #         f"Velocity (x,y,z) [m/s] : [{vel[0]: 8.3f}, {vel[1]: 8.3f}, {vel[2]: 8.3f}]\n"
    #         f"Rotation (r,p,y) [deg] : [{rpy[0]: 8.3f}, {rpy[1]: 8.3f}, {rpy[2]: 8.3f}]\n"
    #         # f"RESIDUAL Linear Accel: [{accel[0]: 8.3f}, {accel[1]: 8.3f}, {accel[2]: 8.3f}]\n"
    #         f"------------------------------"
    #     )

    #     # Print to terminal
    #     print(state_str)

    def correction(self, msg):
        # 1. Lift Pose to SE(3) - We don't need SE2(3) for the measurement itself
        # because we are only observing 6 degrees of freedom (Rot + Pos)
        pose_msg = msg.pose if hasattr(msg, "pose") else msg
        R_meas = R.from_quat([
            pose_msg.orientation.x, pose_msg.orientation.y, 
            pose_msg.orientation.z, pose_msg.orientation.w
        ]).as_matrix()
        p_meas = np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z])

        # 2. Calculate Innovation (Left-Invariant style)
        # Innovation in rotation: Log(R_est^T * R_meas)
        R_est = self.X[0:3, 0:3]
        p_est = self.X[0:3, 4]
        
        delta_R_mat = R_est.T @ R_meas
        # scipy.spatial.transform.Rotation can handle the log map via as_rotvec()
        z_phi = R.from_matrix(delta_R_mat).as_rotvec()
        
        # Innovation in position: R_est^T * (p_meas - p_est)
        # We rotate into the body frame to match the Left-Invariant error definition
        z_p = R_est.T @ (p_meas - p_est)
        
        # Combined Innovation Vector (6x1)
        z = np.hstack([z_phi, z_p])

        # 3. Define the 6x9 Observation Matrix H
        # State error order: [rotation, velocity, position]
        # We only observe rotation (0:3) and position (6:9)
        H_pose = np.zeros((6, 9))
        H_pose[0:3, 0:3] = np.eye(3) # Orientation error
        H_pose[3:6, 6:9] = np.eye(3) # Position error

        # 4. Kalman Gain Logic
        # self.N should be a 6x6 matrix (noise for roll, pitch, yaw, x, y, z)
        # If your self.N is 9x9, slice it: N_6x6 = self.N[[0,1,2,6,7,8], :][:, [0,1,2,6,7,8]]
        N_pose = self.N[0:6, 0:6] 
        
        S = H_pose @ self.P @ H_pose.T + N_pose
        K = self.P @ H_pose.T @ np.linalg.inv(S)

        # 5. Update State
        # delta_xi is 9x1: [rot_corr, vel_corr, pos_corr]
        delta_xi = K @ z
        
        # Apply update using the Exponential Map
        # Left-Invariant Update: X_new = X_old * exp(delta_xi^wedge)
        self.X = self.X @ expm(self.wedge(delta_xi))

        # 6. Update Covariance (Joseph Form for stability)
        I = np.eye(9)
        IKH = I - K @ H_pose
        self.P = IKH @ self.P @ IKH.T + K @ N_pose @ K.T
        
        # 7. Logging (removed DEBUG zeroing)
        rpy = R.from_matrix(self.X[0:3, 0:3]).as_euler('xyz', degrees=True)
        print(f"--- Correction Applied ---")
        print(f"Pos: {np.round(self.X[0:3, 4], 3)}")
        print(f"Ori: {np.round(rpy, 3)}")

    # Takes a geometry_msgs/msg/Pose or PoseStamped and converts it to a 5x5
    # homogeneous transform in SE_2(3).
    def poseToHomogeneous(self, msg, velocity=[0, 0, 0]):

        Y = np.eye(5)
        pose_msg = msg.pose if hasattr(msg, "pose") else msg

        # Columns in Y expect shape (3,), so flatten vectors before assignment.
        v = np.asarray(velocity, dtype=float).reshape(3)
        p = np.asarray([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z], dtype=float).reshape(3)
        q_list = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
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