"""
Left-Invariant EKF (L-IEKF) for full-state INS estimation on SE_2(3).

Based on:
  [1] A. Barrau, S. Bonnabel, "The Invariant Extended Kalman Filter as
      a Stable Observer," IEEE TAC, vol. 62, no. 4, 2017.
  [2] A. Barrau, S. Bonnabel, "Invariant Kalman Filtering," Annual
      Review of Control, Robotics, and Autonomous Systems, 2018.

State:
    xi = (R, v, p, b_omega, b_a)    — same physical state as EqF/R-IEKF

Left-invariant error (15-dim, WORLD frame):
    eps = [d_phi(3), d_v(3), d_p(3), d_b_omega(3), d_b_a(3)]
    where:
        d_phi   = Log(R hat_R^T)        (left attitude error, world frame)
        d_v     = v - hat_v             (world-frame velocity error)
        d_p     = p - hat_p             (world-frame position error)
        d_b_omega = b_omega - hat_b_omega
        d_b_a     = b_a     - hat_b_a

Correction:
    hat_R ← Exp(d_phi) * hat_R        (LEFT multiplication, cf. right in R-IEKF)
    hat_v ← hat_v + d_v               (additive in world frame)
    hat_p ← hat_p + d_p

Key difference from Right-IEKF (eqf_ins.py):
    - Gravity-attitude coupling is [g]× (CONSTANT), not [hat_R^T g]×
    - But attitude dynamics and bias couplings involve hat_R (state-dependent)
    - Measurement innovation is in world frame: z_p = p_meas - hat_p
    - Both L-IEKF and R-IEKF have autonomous error dynamics at the
      linearization origin for group-affine systems (INS qualifies).
"""

import numpy as np
from scipy.spatial.transform import Rotation


# =====================================================================
#  SO(3) utilities (shared with eqf_ins.py)
# =====================================================================

def hat(v):
    """R^3 -> so(3): skew-symmetric matrix."""
    return np.array([[    0, -v[2],  v[1]],
                     [ v[2],     0, -v[0]],
                     [-v[1],  v[0],     0]])

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
        return np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / 2.0
    return angle / (2 * np.sin(angle)) * np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])


# =====================================================================
#  Left-Invariant EKF on SE_2(3) x R^6
# =====================================================================

class LIEKFINS:
    """
    Left-Invariant EKF for INS on SE_2(3) with IMU bias estimation.

    Same interface as EqFINS (R-IEKF), EKFINS, UKFINS.
    """

    DIM_STATE = 15
    DIM_MEAS  = 6

    def __init__(self, quad_params, dt=1/100,
                 Q=None, R_meas=None, P0=None):

        self.g_world = np.array([0, 0, -9.81])
        self.dt = dt

        # State estimate
        self.hat_R = np.eye(3)
        self.hat_v = np.zeros(3)
        self.hat_p = np.zeros(3)
        self.hat_b_omega = np.zeros(3)
        self.hat_b_a = np.zeros(3)

        # Covariance (15 x 15)
        if P0 is not None:
            self.P = P0.copy()
        else:
            self.P = np.diag([
                0.01, 0.01, 0.01,
                0.1,  0.1,  0.1,
                0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
            ])

        # Process noise (15 x 15)
        if Q is not None:
            self.Q = Q.copy()
        else:
            self.Q = np.diag([
                0.001, 0.001, 0.001,
                0.1,   0.1,   0.1,
                1e-6,  1e-6,  1e-6,
                1e-6,  1e-6,  1e-6,
                1e-5,  1e-5,  1e-5,
            ])

        # Measurement noise (6 x 6)
        if R_meas is not None:
            self.R_meas = R_meas.copy()
        else:
            self.R_meas = np.diag([
                0.0005, 0.0005, 0.0005,
                0.0005, 0.0005, 0.0005,
            ])

    # -----------------------------------------------------------------
    def initialize(self, R0, p0, v0=None):
        self.hat_R = R0.copy()
        self.hat_p = p0.copy()
        if v0 is not None:
            self.hat_v = v0.copy()

    # -----------------------------------------------------------------
    #  Prediction (IMU propagation)
    # -----------------------------------------------------------------
    def propagate(self, omega_m, a_m):
        dt = self.dt
        R  = self.hat_R
        v  = self.hat_v
        p  = self.hat_p
        b_w = self.hat_b_omega
        b_a = self.hat_b_a

        # Bias-corrected IMU
        omega = omega_m - b_w
        accel = a_m - b_a

        # State propagation (same strapdown INS as R-IEKF)
        R_new = R @ SO3_exp(omega * dt)
        v_new = v + (R @ accel + self.g_world) * dt
        p_new = p + v * dt + 0.5 * (R @ accel + self.g_world) * dt**2

        # ---- Left-invariant A matrix (15x15) ----
        #
        # Left-invariant error (world frame):
        #   d_phi = Log(R hat_R^T)
        #   d_v   = v - hat_v
        #   d_p   = p - hat_p
        #
        # Linearized error dynamics:
        #   d_phi_dot = -[hat_R omega]x d_phi      - hat_R d_b_omega
        #   d_v_dot   = [g]x d_phi                 - hat_R d_b_a
        #   d_p_dot   =      d_v
        #   d_b_omega_dot = 0
        #   d_b_a_dot     = 0
        #
        # Key: [g]x is CONSTANT (gravity in world frame)
        # But: -[hat_R omega]x and -hat_R depend on state estimate

        omega_world = R @ omega    # angular velocity in world frame

        A = np.zeros((15, 15))

        # d_phi block
        A[0:3, 0:3]   = -hat(omega_world)     # state-dependent (through R)
        A[0:3, 9:12]  = -R                    # gyro bias coupling (state-dep)

        # d_v block
        A[3:6, 0:3]   = hat(self.g_world)     # gravity coupling — CONSTANT!
        A[3:6, 12:15] = -R                    # accel bias coupling (state-dep)

        # d_p block
        A[6:9, 3:6]   = np.eye(3)

        # Discrete-time
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
        MoCap update with LEFT-invariant innovation (world frame).
        """
        R = self.hat_R

        # ---- Left-invariant innovation (world frame) ----
        # Attitude: z_R = Log(R_meas hat_R^T)
        z_R = SO3_log(R_meas @ R.T)

        # Position: z_p = p_meas - hat_p  (directly in world frame)
        z_p = p_meas - self.hat_p

        z = np.concatenate([z_R, z_p])

        # ---- Measurement matrix C (6 x 15) ----
        C = np.zeros((6, 15))
        C[0:3, 0:3] = np.eye(3)    # attitude
        C[3:6, 6:9] = np.eye(3)    # position

        # ---- Kalman update ----
        S = C @ self.P @ C.T + self.R_meas
        K = self.P @ C.T @ np.linalg.inv(S)
        dx = K @ z

        # ---- Apply correction (LEFT action on Lie group) ----
        # Attitude: hat_R = Exp(d_phi) @ hat_R  (LEFT multiplication!)
        self.hat_R = SO3_exp(dx[0:3]) @ R

        # Velocity: additive in world frame
        self.hat_v = self.hat_v + dx[3:6]

        # Position: additive in world frame
        self.hat_p = self.hat_p + dx[6:9]

        # Biases: additive
        self.hat_b_omega = self.hat_b_omega + dx[9:12]
        self.hat_b_a     = self.hat_b_a     + dx[12:15]

        # Covariance (Joseph form)
        I_KC = np.eye(15) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R_meas @ K.T

    # -----------------------------------------------------------------
    def get_state_estimate(self):
        q = Rotation.from_matrix(self.hat_R).as_quat()
        return {
            'x': self.hat_p.copy(),
            'v': self.hat_v.copy(),
            'q': q,
            'w': self._last_omega.copy() if hasattr(self, '_last_omega') else np.zeros(3),
        }

    # -----------------------------------------------------------------
    def step(self, ground_truth_state, controller_command,
             imu_measurement, mocap_measurement):
        gyro  = imu_measurement.get('gyro',  np.zeros(3))
        accel = imu_measurement.get('accel', np.zeros(3))

        self._last_omega = gyro - self.hat_b_omega

        self.propagate(gyro, accel)

        if mocap_measurement is not None and 'q' in mocap_measurement:
            R_meas = Rotation.from_quat(mocap_measurement['q']).as_matrix()
            p_meas = mocap_measurement['x']
            self.update_mocap(R_meas, p_meas)

        fs = np.concatenate([
            self.hat_R.flatten(),
            self.hat_v,
            self.hat_p,
            self.hat_b_omega,
            self.hat_b_a,
        ])

        return {
            'filter_state': fs,
            'covariance':   self.P.copy(),
        }
