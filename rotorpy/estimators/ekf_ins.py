"""
Standard EKF for full-state INS estimation (Euler-angle parameterization).

State (15-dim):
    x = [phi, theta, psi, vx, vy, vz, px, py, pz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

    phi, theta, psi : Euler angles (roll, pitch, yaw) — ZYX convention
    v               : velocity in world frame [m/s]
    p               : position in world frame [m]
    b_g             : gyroscope bias [rad/s]
    b_a             : accelerometer bias [m/s^2]

Inputs:
    omega_m = omega + b_g + n_g   (gyroscope)
    a_m     = R^T(v_dot - g) + b_a + n_a  (accelerometer)

Measurements (MoCap):
    p_meas  : position in world frame
    q_meas  : attitude (quaternion)

Key difference from EqF:
    The linearized A matrix depends on the FULL attitude estimate (sin/cos of
    Euler angles), which can cause inconsistency and poor convergence in
    aggressive maneuvers or near gimbal lock.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class EKFINS:
    """
    Standard EKF INS with Euler-angle attitude parameterization.
    Same 15-dim state and measurement model as EqFINS for fair comparison.
    """

    DIM_STATE = 15
    DIM_MEAS  = 6   # attitude(3) + position(3)

    def __init__(self, quad_params, dt=1/100,
                 Q=None, R_meas=None, P0=None):

        self.g_world = np.array([0, 0, -9.81])
        self.dt = dt

        # State estimate
        self.euler = np.zeros(3)     # [phi, theta, psi]
        self.hat_v = np.zeros(3)
        self.hat_p = np.zeros(3)
        self.hat_b_omega = np.zeros(3)
        self.hat_b_a = np.zeros(3)

        # Covariance
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

        if R_meas is not None:
            self.R_meas = R_meas.copy()
        else:
            self.R_meas = np.diag([
                0.0005, 0.0005, 0.0005,
                0.0005, 0.0005, 0.0005,
            ])

    def _euler_to_R(self, euler):
        """Euler ZYX (phi, theta, psi) -> rotation matrix."""
        return Rotation.from_euler('xyz', euler).as_matrix()

    def _euler_rate_matrix(self, phi, theta):
        """
        Matrix E such that omega_body = E * euler_dot.
        Inverse: euler_dot = E^{-1} * omega_body.
        """
        cp, sp = np.cos(phi), np.sin(phi)
        ct, tt = np.cos(theta), np.tan(theta)
        # euler_dot = E_inv @ omega
        E_inv = np.array([
            [1, sp*tt,  cp*tt],
            [0, cp,    -sp],
            [0, sp/ct,  cp/ct],
        ])
        return E_inv

    def initialize(self, R0, p0, v0=None):
        self.euler = Rotation.from_matrix(R0).as_euler('xyz')
        self.hat_p = p0.copy()
        if v0 is not None:
            self.hat_v = v0.copy()

    def propagate(self, omega_m, a_m):
        dt = self.dt
        phi, theta, psi = self.euler
        R = self._euler_to_R(self.euler)

        # Bias-corrected
        omega = omega_m - self.hat_b_omega
        accel = a_m - self.hat_b_a

        # Euler angle propagation
        E_inv = self._euler_rate_matrix(phi, theta)
        euler_dot = E_inv @ omega
        euler_new = self.euler + euler_dot * dt

        # Velocity and position
        v_new = self.hat_v + (R @ accel + self.g_world) * dt
        p_new = self.hat_p + self.hat_v * dt + 0.5 * (R @ accel + self.g_world) * dt**2

        # Linearized A matrix (state-dependent through Euler angles!)
        cp, sp = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        tt = st / ct if abs(ct) > 1e-8 else 0.0

        A = np.zeros((15, 15))

        # Euler rate Jacobian w.r.t. Euler angles (state-dependent!)
        p, q, r = omega
        A[0, 0] = (cp*tt)*q - (sp*tt)*r
        A[0, 1] = (sp/(ct**2))*q + (cp/(ct**2))*r if abs(ct) > 1e-8 else 0
        A[1, 0] = -sp*q - cp*r
        A[2, 0] = (cp/ct)*q - (sp/ct)*r if abs(ct) > 1e-8 else 0
        A[2, 1] = (sp*st/(ct**2))*q + (cp*st/(ct**2))*r if abs(ct) > 1e-8 else 0

        # Euler rate w.r.t. gyro bias
        A[0:3, 9:12] = -E_inv

        # Velocity Jacobian w.r.t. Euler angles (depends on R and accel!)
        # d(R @ accel)/d(euler) — this is the problematic term
        ax, ay, az = accel
        # Numerical Jacobian for robustness
        eps = 1e-6
        for j in range(3):
            e_plus = self.euler.copy(); e_plus[j] += eps
            e_minus = self.euler.copy(); e_minus[j] -= eps
            R_p = self._euler_to_R(e_plus)
            R_m = self._euler_to_R(e_minus)
            A[3:6, j] = (R_p @ accel - R_m @ accel) / (2 * eps)

        # Velocity w.r.t. accel bias
        A[3:6, 12:15] = -R

        # Position w.r.t. velocity
        A[6:9, 3:6] = np.eye(3)

        # Discrete-time
        F = np.eye(15) + A * dt
        self.P = F @ self.P @ F.T + self.Q * dt

        self.euler = euler_new
        self.hat_v = v_new
        self.hat_p = p_new

    def update_mocap(self, R_meas, p_meas):
        # Innovation
        euler_meas = Rotation.from_matrix(R_meas).as_euler('xyz')
        z_euler = euler_meas - self.euler
        # Wrap angles to [-pi, pi]
        z_euler = (z_euler + np.pi) % (2*np.pi) - np.pi
        z_p = p_meas - self.hat_p

        z = np.concatenate([z_euler, z_p])

        C = np.zeros((6, 15))
        C[0:3, 0:3] = np.eye(3)
        C[3:6, 6:9] = np.eye(3)

        S = C @ self.P @ C.T + self.R_meas
        K = self.P @ C.T @ np.linalg.inv(S)
        dx = K @ z

        self.euler += dx[0:3]
        self.hat_v += dx[3:6]
        self.hat_p += dx[6:9]
        self.hat_b_omega += dx[9:12]
        self.hat_b_a += dx[12:15]

        I_KC = np.eye(15) - K @ C
        self.P = I_KC @ self.P @ I_KC.T + K @ self.R_meas @ K.T

    def get_state_estimate(self):
        R = self._euler_to_R(self.euler)
        q = Rotation.from_matrix(R).as_quat()
        return {
            'x': self.hat_p.copy(),
            'v': self.hat_v.copy(),
            'q': q,
            'w': self._last_omega.copy() if hasattr(self, '_last_omega') else np.zeros(3),
        }

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

        R = self._euler_to_R(self.euler)
        fs = np.concatenate([
            R.flatten(),
            self.hat_v,
            self.hat_p,
            self.hat_b_omega,
            self.hat_b_a,
        ])

        return {
            'filter_state': fs,
            'covariance':   self.P.copy(),
        }
