"""
Equivariant Filter (EqF) for full-state INS estimation on SE_2(3).

Based on:
  [1] A. Barrau, S. Bonnabel, "The Invariant Extended Kalman Filter as
      a Stable Observer," IEEE TAC, vol. 62, no. 4, 2017.
  [2] A. Fornasier, P. van Goor, R. Mahony, S. Weiss, "Equivariant Filter
      Design for Inertial Navigation Systems with Input Measurement Biases,"
      ICRA 2022.

State:
    xi = (R, v, p, b_omega, b_a)
    R      : SO(3)  attitude (body -> world)
    v      : R^3    velocity in world frame [m/s]
    p      : R^3    position in world frame [m]
    b_omega: R^3    gyroscope bias [rad/s]
    b_a    : R^3    accelerometer bias [m/s^2]

Symmetry group: SE_2(3) x R^6
    The navigation states (R, v, p) form the extended pose T in SE_2(3).
    The right-invariant error eta = hat_T^{-1} T has autonomous error
    dynamics at the linearization origin --- the hallmark of the EqF.

Inputs:
    omega_m = omega + b_omega + n_omega   (gyroscope)
    a_m     = R^T(v_dot - g) + b_a + n_a (accelerometer)

Measurements (MoCap):
    p_meas  : position in world frame
    R_meas  : attitude (rotation matrix)

Error state (15-dim):
    eps = [d_phi(3), d_v(3), d_p(3), d_b_omega(3), d_b_a(3)]
    where:
        d_phi = Log(hat_R^T R)          (right-invariant attitude error)
        d_v   = hat_R^T (v - hat_v)     (right-invariant velocity error)
        d_p   = hat_R^T (p - hat_p)     (right-invariant position error)
        d_b_omega = b_omega - hat_b_omega
        d_b_a     = b_a     - hat_b_a

Key advantage over Euler-angle EKF:
    The linearized A matrix depends only on (omega_m, hat_R^T g), NOT on the
    full state estimate.  This gives the EqF/IEKF superior convergence
    properties and consistency compared to standard EKF.
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
    """Exponential map so(3) -> SO(3) via Rodrigues."""
    angle = np.linalg.norm(phi)
    if angle < 1e-10:
        return np.eye(3) + hat(phi)
    K = hat(phi / angle)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K

def SO3_log(R):
    """Logarithmic map SO(3) -> R^3 (rotation vector)."""
    cos_angle = np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-10:
        return vee(R - R.T) / 2.0
    return angle / (2 * np.sin(angle)) * vee(R - R.T)


# =====================================================================
#  EqF / Right-Invariant EKF on SE_2(3) x R^6
# =====================================================================

class EqFINS:
    """
    Full-state INS estimator using the Equivariant Filter (Right-IEKF)
    on SE_2(3) with IMU bias estimation.

    Estimates: attitude, velocity, position, gyro bias, accel bias.
    Measurements: position and attitude from motion capture.

    Parameters
    ----------
    quad_params : dict
        Vehicle parameters (only 'mass' is used).
    dt : float
        Time step [s].
    Q : np.ndarray (15,15), optional
        Process noise covariance.
    R_meas : np.ndarray (6,6), optional
        Measurement noise covariance (attitude 3 + position 3).
    P0 : np.ndarray (15,15), optional
        Initial error covariance.
    """

    DIM_STATE = 15   # d_phi(3), d_v(3), d_p(3), d_b_omega(3), d_b_a(3)
    DIM_MEAS  = 6    # attitude(3) + position(3)

    def __init__(self, quad_params, dt=1/100,
                 Q=None, R_meas=None, P0=None):

        self.g_world = np.array([0, 0, -9.81])
        self.dt = dt

        # ---- State estimate ----
        self.hat_R = np.eye(3)
        self.hat_v = np.zeros(3)
        self.hat_p = np.zeros(3)
        self.hat_b_omega = np.zeros(3)
        self.hat_b_a = np.zeros(3)

        # ---- Covariance (15 x 15) ----
        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.diag([
                0.01, 0.01, 0.01,   # attitude [rad^2]
                0.1,  0.1,  0.1,    # velocity [(m/s)^2]
                0.01, 0.01, 0.01,   # position [m^2]
                0.01, 0.01, 0.01,   # gyro bias [(rad/s)^2]
                0.01, 0.01, 0.01,   # accel bias [(m/s^2)^2]
            ])

        # ---- Process noise (15 x 15) ----
        if Q is not None:
            self.Q = Q.copy()
        else:
            self.Q = np.diag([
                0.001, 0.001, 0.001,   # gyro noise
                0.1,   0.1,   0.1,     # accel noise
                1e-6,  1e-6,  1e-6,    # position process (small)
                1e-6,  1e-6,  1e-6,    # gyro bias random walk
                1e-5,  1e-5,  1e-5,    # accel bias random walk
            ])

        # ---- Measurement noise (6 x 6) ----
        if R_meas is not None:
            self.R_meas = R_meas.copy()
        else:
            self.R_meas = np.diag([
                0.0005, 0.0005, 0.0005,   # MoCap attitude [rad^2]
                0.0005, 0.0005, 0.0005,    # MoCap position [m^2]
            ])

    # -----------------------------------------------------------------
    #  Initialize state from first measurement
    # -----------------------------------------------------------------
    def initialize(self, R0, p0, v0=None):
        """Set initial state estimate from first measurement."""
        self.hat_R = R0.copy()
        self.hat_p = p0.copy()
        if v0 is not None:
            self.hat_v = v0.copy()

    # -----------------------------------------------------------------
    #  Prediction (IMU propagation)
    # -----------------------------------------------------------------
    def propagate(self, omega_m, a_m):
        """
        Propagate state with IMU measurements.

        Parameters
        ----------
        omega_m : array (3,)  — gyroscope reading [rad/s] (body frame)
        a_m     : array (3,)  — accelerometer reading [m/s^2] (body frame)
        """
        dt = self.dt
        R  = self.hat_R
        v  = self.hat_v
        p  = self.hat_p
        b_w = self.hat_b_omega
        b_a = self.hat_b_a

        # Bias-corrected IMU
        omega = omega_m - b_w
        accel = a_m - b_a

        # ---- State propagation (strapdown INS) ----
        R_new = R @ SO3_exp(omega * dt)
        v_new = v + (R @ accel + self.g_world) * dt
        p_new = p + v * dt + 0.5 * (R @ accel + self.g_world) * dt**2

        # Biases: constant model
        # b_w_new = b_w, b_a_new = b_a

        # ---- Linearized error dynamics (A matrix, 15x15) ----
        #
        # Right-invariant error:
        #   d_phi_dot = -[omega]x d_phi      - d_b_omega
        #   d_v_dot   = [hat_R^T g]x d_phi   - hat_R^T hat_R d_b_a  = [g_b]x d_phi - d_b_a_body
        #   d_p_dot   =              d_v
        #   d_b_omega_dot = 0
        #   d_b_a_dot     = 0
        #
        # Note: in the right-invariant formulation, the velocity-attitude
        # coupling is through g_body = hat_R^T g, NOT through a_m.
        # This is the key property that makes the A matrix state-independent
        # (g is constant, only hat_R varies, which is "mildly" state-dependent).

        g_body = R.T @ self.g_world   # gravity in estimated body frame

        A = np.zeros((15, 15))

        # d_phi block
        A[0:3, 0:3]   = -hat(omega)       # attitude dynamics
        A[0:3, 9:12]  = -np.eye(3)        # gyro bias coupling

        # d_v block
        A[3:6, 0:3]   = hat(g_body)       # gravity-attitude coupling (EqF key!)
        A[3:6, 12:15] = -np.eye(3)        # accel bias coupling

        # d_p block
        A[6:9, 3:6]   = np.eye(3)         # velocity -> position

        # Bias blocks: all zero (constant model)

        # ---- Discrete-time propagation ----
        F = np.eye(15) + A * dt
        self.P = F @ self.P @ F.T + self.Q * dt

        # Update state
        self.hat_R = R_new
        self.hat_v = v_new
        self.hat_p = p_new

    # -----------------------------------------------------------------
    #  Update (MoCap measurement)
    # -----------------------------------------------------------------
    def update_mocap(self, R_meas, p_meas):
        """
        Update with MoCap measurements (attitude + position).

        Parameters
        ----------
        R_meas : array (3,3) — measured rotation matrix
        p_meas : array (3,)  — measured position [m]
        """
        R = self.hat_R

        # ---- Right-invariant innovation ----
        # Attitude: z_R = Log(hat_R^T R_meas)
        z_R = SO3_log(R.T @ R_meas)

        # Position: z_p = hat_R^T (p_meas - hat_p)
        z_p = R.T @ (p_meas - self.hat_p)

        z = np.concatenate([z_R, z_p])   # (6,)

        # ---- Measurement matrix C (6 x 15) ----
        # Right-invariant observation:
        #   z_R ≈ d_phi
        #   z_p ≈ d_p
        C = np.zeros((6, 15))
        C[0:3, 0:3] = np.eye(3)    # attitude
        C[3:6, 6:9] = np.eye(3)    # position

        # ---- Kalman update ----
        S = C @ self.P @ C.T + self.R_meas
        K = self.P @ C.T @ np.linalg.inv(S)
        dx = K @ z   # (15,)

        # ---- Apply correction on the Lie group ----
        # Attitude: hat_R = hat_R @ Exp(d_phi)
        self.hat_R = R @ SO3_exp(dx[0:3])

        # Velocity: hat_v = hat_v + hat_R @ d_v
        self.hat_v = self.hat_v + R @ dx[3:6]

        # Position: hat_p = hat_p + hat_R @ d_p
        self.hat_p = self.hat_p + R @ dx[6:9]

        # Biases: additive correction
        self.hat_b_omega = self.hat_b_omega + dx[9:12]
        self.hat_b_a     = self.hat_b_a     + dx[12:15]

        # ---- Covariance (Joseph form) ----
        I_KC = np.eye(15) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R_meas @ K.T

    # -----------------------------------------------------------------
    #  Build state dict for controller feedback
    # -----------------------------------------------------------------
    def get_state_estimate(self):
        """
        Return estimated state as a dict compatible with the controller
        interface (same keys as ground truth state).

        Returns
        -------
        dict with keys: 'x', 'v', 'q', 'w'
        """
        q = Rotation.from_matrix(self.hat_R).as_quat()  # [i,j,k,w]

        # Estimated body rates: use bias-corrected gyro from last propagation
        # (stored separately or passed through)
        return {
            'x': self.hat_p.copy(),
            'v': self.hat_v.copy(),
            'q': q,
            'w': self._last_omega.copy() if hasattr(self, '_last_omega') else np.zeros(3),
        }

    # -----------------------------------------------------------------
    #  Interface: step() — RotorPy estimator framework
    # -----------------------------------------------------------------
    def step(self, ground_truth_state, controller_command,
             imu_measurement, mocap_measurement):
        """
        One EqF cycle: propagate with IMU, update with MoCap.

        Returns
        -------
        dict with:
            'filter_state' : np.ndarray (21,)
                [R(9), v(3), p(3), b_omega(3), b_a(3)]
            'covariance' : np.ndarray (15, 15)
        """
        # ---- IMU propagation ----
        gyro  = imu_measurement.get('gyro',  np.zeros(3))
        accel = imu_measurement.get('accel', np.zeros(3))

        # Store bias-corrected gyro for body rate estimate
        self._last_omega = gyro - self.hat_b_omega

        self.propagate(gyro, accel)

        # ---- MoCap update ----
        if mocap_measurement is not None and 'q' in mocap_measurement:
            R_meas = Rotation.from_quat(mocap_measurement['q']).as_matrix()
            p_meas = mocap_measurement['x']
            self.update_mocap(R_meas, p_meas)

        # ---- Pack output ----
        fs = np.concatenate([
            self.hat_R.flatten(),    # 9
            self.hat_v,              # 3
            self.hat_p,              # 3
            self.hat_b_omega,        # 3
            self.hat_b_a,            # 3
        ])

        return {
            'filter_state': fs,
            'covariance':   self.P.copy(),
        }
