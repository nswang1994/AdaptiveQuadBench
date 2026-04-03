"""
Unscented Kalman Filter (UKF) for full-state INS estimation.

Same 15-dim state as EKF INS and EqF INS for fair comparison:
    x = [phi, theta, psi, vx, vy, vz, px, py, pz, b_gx, b_gy, b_gz, b_ax, b_ay, b_az]

Uses filterpy's UKF with Merwe scaled sigma points.

Advantage over EKF: 3rd-order accuracy without Jacobians.
Disadvantage vs EqF: no symmetry exploitation, sigma points don't
    respect SO(3) manifold structure.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints


class UKFINS:
    """
    UKF INS with Euler-angle attitude parameterization.
    Same interface as EqFINS and EKFINS.
    """

    DIM_STATE = 15
    DIM_MEAS  = 6

    def __init__(self, quad_params, dt=1/100,
                 Q=None, R_meas=None, P0=None,
                 alpha=0.1, beta=2.0, kappa=-1):

        self.g_world = np.array([0, 0, -9.81])
        self.dt = dt

        if P0 is None:
            P0 = np.diag([
                0.01, 0.01, 0.01,
                0.1,  0.1,  0.1,
                0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
                0.01, 0.01, 0.01,
            ])
        if Q is None:
            Q = np.diag([
                0.001, 0.001, 0.001,
                0.1,   0.1,   0.1,
                1e-6,  1e-6,  1e-6,
                1e-6,  1e-6,  1e-6,
                1e-5,  1e-5,  1e-5,
            ])
        if R_meas is None:
            R_meas = np.diag([
                0.0005, 0.0005, 0.0005,
                0.0005, 0.0005, 0.0005,
            ])

        self._Q = Q
        self._R_meas = R_meas

        # Store latest IMU for process model
        self._omega_m = np.zeros(3)
        self._a_m = np.zeros(3)

        points = MerweScaledSigmaPoints(15, alpha=alpha, beta=beta, kappa=kappa)

        self.ukf = UnscentedKalmanFilter(
            dim_x=15, dim_z=6, dt=dt,
            fx=self._fx, hx=self._hx, points=points)
        self.ukf.x = np.zeros(15)
        self.ukf.P = P0.copy()
        self.ukf.Q = Q.copy()
        self.ukf.R = R_meas.copy()

    def _euler_to_R(self, euler):
        return Rotation.from_euler('xyz', euler).as_matrix()

    def _fx(self, x, dt):
        """Process model: strapdown INS with Euler angles."""
        phi, theta, psi = x[0], x[1], x[2]
        v = x[3:6]
        p = x[6:9]
        b_g = x[9:12]
        b_a = x[12:15]

        omega = self._omega_m - b_g
        accel = self._a_m - b_a
        R = self._euler_to_R(x[0:3])

        # Euler rate
        cp, sp = np.cos(phi), np.sin(phi)
        ct = np.cos(theta)
        if abs(ct) < 1e-8:
            ct = 1e-8 * np.sign(ct) if ct != 0 else 1e-8
        tt = np.sin(theta) / ct
        E_inv = np.array([
            [1, sp*tt,  cp*tt],
            [0, cp,    -sp],
            [0, sp/ct,  cp/ct],
        ])
        euler_dot = E_inv @ omega

        v_dot = R @ accel + self.g_world
        p_dot = v

        x_new = x.copy()
        x_new[0:3] = x[0:3] + euler_dot * dt
        x_new[3:6] = v + v_dot * dt
        x_new[6:9] = p + p_dot * dt + 0.5 * v_dot * dt**2
        # biases constant
        return x_new

    def _hx(self, x):
        """Measurement model: Euler angles + position."""
        return np.concatenate([x[0:3], x[6:9]])

    def initialize(self, R0, p0, v0=None):
        euler0 = Rotation.from_matrix(R0).as_euler('xyz')
        self.ukf.x[0:3] = euler0
        self.ukf.x[6:9] = p0.copy()
        if v0 is not None:
            self.ukf.x[3:6] = v0.copy()

    def get_state_estimate(self):
        x = self.ukf.x
        R = self._euler_to_R(x[0:3])
        q = Rotation.from_matrix(R).as_quat()
        return {
            'x': x[6:9].copy(),
            'v': x[3:6].copy(),
            'q': q,
            'w': self._last_omega.copy() if hasattr(self, '_last_omega') else np.zeros(3),
        }

    def step(self, ground_truth_state, controller_command,
             imu_measurement, mocap_measurement):
        gyro  = imu_measurement.get('gyro',  np.zeros(3))
        accel = imu_measurement.get('accel', np.zeros(3))

        self._omega_m = gyro
        self._a_m = accel
        self._last_omega = gyro - self.ukf.x[9:12]

        # Predict
        self.ukf.predict()

        # Update with MoCap
        if mocap_measurement is not None and 'q' in mocap_measurement:
            euler_meas = Rotation.from_quat(mocap_measurement['q']).as_euler('xyz')
            p_meas = mocap_measurement['x']
            z = np.concatenate([euler_meas, p_meas])
            self.ukf.update(z)

        x = self.ukf.x
        R = self._euler_to_R(x[0:3])
        fs = np.concatenate([
            R.flatten(),
            x[3:6],
            x[6:9],
            x[9:12],
            x[12:15],
        ])

        return {
            'filter_state': fs,
            'covariance':   self.ukf.P.copy(),
        }
