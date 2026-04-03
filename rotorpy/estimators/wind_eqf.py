"""
Equivariant Filter (EqF) for quadrotor wind estimation.

Based on the EqF framework of:
  [1] P. van Goor, T. Hamel, R. Mahony, "Equivariant Filter (EqF),"
      IEEE Trans. Automatic Control, vol. 68, no. 6, 2023.
  [2] A. Fornasier, Y. Ge, P. van Goor, R. Mahony, S. Weiss,
      "Equivariant Filter Design for Inertial Navigation Systems with
      Input Measurement Biases," ICRA 2022.

State manifold:
    xi = (R, v, w) in SO(3) x R^3 x R^3
    R:  attitude (body -> world rotation matrix)
    v:  velocity in world frame [m/s]
    w:  wind velocity in world frame [m/s]

Symmetry group:
    G = SO(3) x R^3 x R^3   (direct product acting on (R, v, w))
    Element X = (A, b_v, b_w):
        phi(X, xi) = (A @ R,  A @ v + b_v,  A @ w + b_w)

Inputs (from IMU):
    omega_m:  measured angular velocity [rad/s]  (body frame)
    a_m:      measured specific force [m/s^2]    (body frame)

Measurements (from MoCap):
    q_m:    quaternion [i,j,k,w]
    v_m:    velocity in world frame [m/s]

Key advantages over the existing EKF/UKF:
    1. Attitude on SO(3) — no gimbal lock, no Euler angle singularities
    2. Equivariant error definition — linearization around the symmetry
       origin gives second-order output approximation (O(|eps|^3))
    3. Fixed-structure linearization — A_0 depends on origin input, not estimate
    4. Geometric consistency for unobservable directions
"""

import numpy as np
from scipy.spatial.transform import Rotation


# =====================================================================
#  SO(3) utilities
# =====================================================================

def hat(v):
    """R^3 -> so(3): skew-symmetric matrix."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

def vee(S):
    """so(3) -> R^3: inverse of hat map."""
    return np.array([S[2,1], S[0,2], S[1,0]])

def SO3_exp(phi):
    """Exponential map so(3) -> SO(3) using Rodrigues' formula."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + hat(phi)
    axis = phi / angle
    K = hat(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def SO3_log(R):
    """Logarithmic map SO(3) -> so(3), returns the rotation vector in R^3."""
    cos_angle = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return vee(R - R.T) / 2.0
    return angle / (2 * np.sin(angle)) * vee(R - R.T)

def SO3_left_jacobian(phi):
    """Left Jacobian of SO(3):  J_l(phi) such that Exp(phi + dphi) ≈ Exp(J_l * dphi) * Exp(phi)."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3)
    K = hat(phi / angle)
    return (np.sin(angle)/angle * np.eye(3)
            + (1 - np.sin(angle)/angle) * np.outer(phi, phi) / angle**2
            + (1 - np.cos(angle))/angle * K)


# =====================================================================
#  Equivariant Filter for Wind Estimation
# =====================================================================

class WindEqF:
    """
    Equivariant Filter for attitude + velocity + wind estimation.

    Parameters
    ----------
    quad_params : dict
        Vehicle parameter dictionary (needs 'mass', 'c_Dx', 'c_Dy', 'c_Dz').
    Q : np.ndarray (9,9)
        Process noise covariance.
    R_meas : np.ndarray (9,9)
        Measurement noise covariance.
    R0 : np.ndarray (3,3)
        Initial attitude estimate (rotation matrix).
    v0 : np.ndarray (3,)
        Initial velocity estimate [m/s].
    w0 : np.ndarray (3,)
        Initial wind estimate [m/s].
    P0 : np.ndarray (9,9)
        Initial error covariance.
    dt : float
        Time step [s].

    Notes
    -----
    The 9-dim error state is ordered as:
        eps = [eps_R (3), eps_v (3), eps_w (3)]
    where eps_R is the SO(3) rotation error vector (right-invariant).
    """

    def __init__(self, quad_params,
                 Q=None, R_meas=None,
                 R0=None, v0=None, w0=None,
                 P0=None, dt=1/100):

        self.mass = quad_params['mass']
        self.g = np.array([0, 0, -9.81])

        # Drag coefficients (parasitic drag in body frame)
        # Support both RotorPy model (c_Dx) and alternative model (cd1_x)
        if quad_params.get('cd1_x', None) is not None and quad_params['cd1_x'] > 0:
            # Alternative drag model: F_drag = -diag(cd1) @ v_air (linear)
            self.cd1x = quad_params['cd1_x']
            self.cd1y = quad_params['cd1_y']
            self.cd1z = quad_params['cd1_z']
            self.C_D = np.diag([self.cd1x, self.cd1y, self.cd1z])
            self.drag_model = 'linear'   # D = -C_D @ v_air / m
        else:
            # RotorPy model: F_drag = -||v_air|| * diag(c_D) @ v_air (quadratic)
            self.c_Dx = quad_params.get('c_Dx', 0.0)
            self.c_Dy = quad_params.get('c_Dy', 0.0)
            self.c_Dz = quad_params.get('c_Dz', 0.0)
            self.C_D = np.diag([self.c_Dx, self.c_Dy, self.c_Dz])
            self.drag_model = 'quadratic'  # D = -||v_air|| * C_D @ v_air / m

        self.dt = dt

        # ---- Observer state (group element) ----
        # Instead of tracking X in G, we directly track the state estimate
        # hat_xi = (hat_R, hat_v, hat_w)
        self.hat_R = R0 if R0 is not None else np.eye(3)
        self.hat_v = v0 if v0 is not None else np.zeros(3)
        self.hat_w = w0 if w0 is not None else np.zeros(3)

        # ---- Covariance ----
        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.diag([0.1, 0.1, 0.1,     # attitude [rad^2]
                              1.0, 1.0, 1.0,       # velocity [(m/s)^2]
                              4.0, 4.0, 4.0])       # wind [(m/s)^2]

        # ---- Noise ----
        if Q is not None:
            self.Q = Q.copy()
        else:
            # Process noise: gyro noise, accel noise, wind drift
            self.Q = np.diag([0.01, 0.01, 0.01,    # gyro noise [rad^2/s]
                              0.5,  0.5,  0.5,      # accel noise [(m/s^2)^2/s]
                              0.01, 0.01, 0.01])     # wind random walk [(m/s)^2/s]

        if R_meas is not None:
            self.R_meas = R_meas.copy()
        else:
            # Measurement noise: attitude (3) + velocity (3) = 6 measurements
            # Wind is NOT directly measured — it is observed indirectly
            # through drag coupling in the process model (A matrix).
            self.R_meas = np.diag([0.001, 0.001, 0.001,   # attitude [rad^2]
                                   0.01,  0.01,  0.01])    # velocity [(m/s)^2]

    # -----------------------------------------------------------------
    #  Aerodynamic drag in body frame (parasitic drag model)
    # -----------------------------------------------------------------
    def _drag_body(self, v_air_body):
        """
        Parasitic drag acceleration in body frame (force / mass).

        Linear model:   a_drag = -C_D @ v_air / m
        Quadratic model: a_drag = -||v_air|| * C_D @ v_air / m
        """
        if self.drag_model == 'linear':
            return -self.C_D @ v_air_body / self.mass
        else:
            speed = np.linalg.norm(v_air_body)
            return -speed * self.C_D @ v_air_body / self.mass

    def _drag_jacobian_v(self, v_air_body):
        """
        Jacobian of drag acceleration w.r.t. body-frame airspeed.
        d(a_drag)/d(v_air_body) — used for linearization.
        """
        if self.drag_model == 'linear':
            # d/dv [-C_D * v / m] = -C_D / m
            return -self.C_D / self.mass
        else:
            speed = np.linalg.norm(v_air_body)
            if speed < 1e-8:
                return np.zeros((3, 3))
            return -self.C_D / self.mass * (
                np.outer(v_air_body, v_air_body)/speed + speed * np.eye(3))

    # -----------------------------------------------------------------
    #  Prediction step (process model)
    # -----------------------------------------------------------------
    def propagate(self, omega_m, a_m):
        """
        Propagate the state estimate using IMU measurements.

        Parameters
        ----------
        omega_m : array (3,)  — measured angular velocity [rad/s] (body)
        a_m     : array (3,)  — measured specific force [m/s^2] (body)
        """
        dt = self.dt
        R, v, w = self.hat_R, self.hat_v, self.hat_w

        # Body-frame airspeed at current estimate
        v_air_body = R.T @ (v - w)

        # Drag acceleration (body frame) -> world frame
        drag_world = R @ self._drag_body(v_air_body)

        # ---- State propagation ----
        # Attitude: R+ = R * Exp(omega_m * dt)
        R_new = R @ SO3_exp(omega_m * dt)

        # Velocity: v+ = v + (R * a_m + g + drag) * dt
        v_dot = R @ a_m + self.g + drag_world
        v_new = v + v_dot * dt

        # Wind: w+ = w  (constant model)
        w_new = w.copy()

        # ---- Linearized error dynamics (A matrix) ----
        # Error state: eps = [eps_R, eps_v, eps_w] in R^9
        # Right-invariant error:
        #   eta_R = hat_R^T @ R  ≈ I + [eps_R]x
        #   eta_v = hat_R^T @ (v - hat_v) ≈ eps_v
        #   eta_w = hat_R^T @ (w - hat_w) ≈ eps_w
        #
        # Error dynamics (continuous time, evaluated at origin eps=0):

        # Drag Jacobian in body frame
        D_v = self._drag_jacobian_v(v_air_body)  # 3x3, d(drag_body)/d(v_air_body)

        # A matrix blocks (9x9):
        #   d(eps_R_dot)/d(eps_R) = -[omega_m]x     (3x3)
        #   d(eps_R_dot)/d(eps_v) = 0                (3x3)
        #   d(eps_R_dot)/d(eps_w) = 0                (3x3)
        #
        #   d(eps_v_dot)/d(eps_R) = -[R*a_m]x - [R*drag_body]x  (3x3)  [gravity coupling + drag coupling]
        #   d(eps_v_dot)/d(eps_v) = R @ D_v @ R^T    (3x3)  [drag -> velocity]
        #   d(eps_v_dot)/d(eps_w) = -R @ D_v @ R^T   (3x3)  [drag -> wind (opposite sign)]
        #
        #   d(eps_w_dot)/d(eps_R) = 0                (3x3)
        #   d(eps_w_dot)/d(eps_v) = 0                (3x3)
        #   d(eps_w_dot)/d(eps_w) = 0                (3x3)

        A = np.zeros((9, 9))

        # Attitude block
        A[0:3, 0:3] = -hat(omega_m)

        # Velocity-attitude coupling (equivariant formulation)
        drag_body = self._drag_body(v_air_body)
        A[3:6, 0:3] = -hat(R @ a_m) - hat(R @ drag_body)

        # Velocity-velocity and velocity-wind (through drag)
        RDvRT = R @ D_v @ R.T
        A[3:6, 3:6] = RDvRT
        A[3:6, 6:9] = -RDvRT

        # Wind block: all zeros (constant wind model)

        # ---- Discrete-time Riccati propagation ----
        # P+ = F P F^T + Q*dt,  where F = I + A*dt
        F = np.eye(9) + A * dt
        self.P = F @ self.P @ F.T + self.Q * dt

        # Update state
        self.hat_R = R_new
        self.hat_v = v_new
        self.hat_w = w_new

    # -----------------------------------------------------------------
    #  Update step (measurement correction)
    # -----------------------------------------------------------------
    def update(self, R_meas, v_meas):
        """
        Update the state estimate using attitude and velocity measurements.

        Wind is NOT directly measured.  Its observability comes from the
        process model coupling: drag depends on airspeed (v − w), so the
        off-diagonal blocks in P propagate velocity-measurement information
        into the wind state.

        Parameters
        ----------
        R_meas : array (3,3)  — measured rotation matrix (from MoCap)
        v_meas : array (3,)   — measured velocity in world frame (from MoCap)
        """
        R, v, w = self.hat_R, self.hat_v, self.hat_w

        # ---- Innovation (equivariant output error) ----
        # Attitude innovation: eps_R ≈ Log(hat_R^T @ R_meas)  ∈ R^3
        dR = R.T @ R_meas
        eps_R = SO3_log(dR)

        # Velocity innovation: eps_v = hat_R^T @ (v_meas − hat_v)  ∈ R^3
        eps_v = R.T @ (v_meas - v)

        # 6-dim innovation (attitude + velocity only)
        y_tilde = np.concatenate([eps_R, eps_v])          # (6,)

        # ---- Measurement matrix  C  (6 × 9) ----
        # Measures attitude (first 3 error states) and velocity (next 3).
        # Wind (last 3 error states) is NOT directly measured.
        C = np.zeros((6, 9))
        C[0:3, 0:3] = np.eye(3)   # attitude
        C[3:6, 3:6] = np.eye(3)   # velocity

        # ---- Kalman gain  K  (9 × 6) ----
        S = C @ self.P @ C.T + self.R_meas          # (6 × 6)
        K = self.P @ C.T @ np.linalg.inv(S)         # (9 × 6)

        # ---- State correction ----
        dx = K @ y_tilde                              # (9,)

        # Apply correction on the Lie group
        self.hat_R = R @ SO3_exp(dx[0:3])             # attitude
        self.hat_v = v + R @ dx[3:6]                  # velocity
        self.hat_w = w + R @ dx[6:9]                  # wind (indirect!)

        # ---- Covariance update (Joseph form) ----
        I_KC = np.eye(9) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R_meas @ K.T

    # -----------------------------------------------------------------
    #  Interface: step() — compatible with RotorPy estimator framework
    # -----------------------------------------------------------------
    def step(self, ground_truth_state, controller_command, imu_measurement, mocap_measurement):
        """
        One step of the EqF: propagate with IMU, then update with MoCap.

        Parameters
        ----------
        ground_truth_state : dict
            True vehicle state (used only to extract body rates if needed).
            Keys: 'x', 'v', 'q', 'w', 'rotor_speeds', ...
        controller_command : dict
            Controller output. Keys: 'cmd_thrust', 'cmd_moment', ...
        imu_measurement : dict
            IMU readings. Keys: 'accel' (3,), 'gyro' (3,).
        mocap_measurement : dict
            MoCap readings. Keys: 'x' (3,), 'v' (3,), 'q' (4,), 'w' (3,).

        Returns
        -------
        dict with keys:
            'filter_state' : np.ndarray (15,)
                Flattened [R (9), v (3), w (3)] — 15 elements.
            'covariance' : np.ndarray (9, 9)
                Error covariance matrix.
        """

        # ---- Extract IMU data ----
        accel = imu_measurement.get('accel', np.zeros(3))
        gyro  = imu_measurement.get('gyro',  np.zeros(3))

        # The accelerometer measures specific force in body frame:
        #   a_m = R^T @ (v_dot - g)  (approximately)
        # The gyroscope measures angular velocity in body frame:
        #   omega_m = omega  (approximately)

        # ---- Propagate ----
        self.propagate(gyro, accel)

        # ---- Extract MoCap data for update ----
        if mocap_measurement is not None and 'q' in mocap_measurement:
            q_mocap = mocap_measurement['q']       # [i,j,k,w]
            v_mocap = mocap_measurement['v']       # world frame velocity [m/s]

            # Convert quaternion to rotation matrix
            R_mocap = Rotation.from_quat(q_mocap).as_matrix()

            # ---- Update ----
            self.update(R_mocap, v_mocap)

        # ---- Pack output ----
        filter_state = np.concatenate([
            self.hat_R.flatten(),   # 9 elements
            self.hat_v,             # 3 elements
            self.hat_w              # 3 elements
        ])

        return {
            'filter_state': filter_state,
            'covariance':   self.P.copy()
        }

    # -----------------------------------------------------------------
    #  Convenience accessors
    # -----------------------------------------------------------------
    @property
    def attitude(self):
        """Estimated attitude as rotation matrix."""
        return self.hat_R.copy()

    @property
    def velocity(self):
        """Estimated velocity in world frame [m/s]."""
        return self.hat_v.copy()

    @property
    def wind(self):
        """Estimated wind velocity in world frame [m/s]."""
        return self.hat_w.copy()

    @property
    def euler_deg(self):
        """Estimated attitude as Euler angles [deg] (roll, pitch, yaw)."""
        return Rotation.from_matrix(self.hat_R).as_euler('xyz', degrees=True)
