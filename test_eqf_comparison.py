"""
Compare wind estimation: EKF vs EqF (Mahony) on the same flight.

Both estimators run as "sidecars" — they observe the same noisy sensor
data but their outputs are NOT fed back to the controller.

Setup:
  - Vehicle: 0.826 kg X-config quadrotor
  - Controller: SE3Control (receives ground truth state)
  - Trajectory: circular (r=2 m)
  - Wind: constant 3 m/s in +x direction
  - IMU + MoCap noise: default settings
  - Duration: 15 s

We compare:
  1. EKF (existing, Euler-angle based)
  2. EqF (new, SO(3)-native, Mahony framework)
"""

import sys, os, copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))

from sim_config import get_config, make_x0, make_wind, make_sensors, get_quad_params, NullEstimator

cfg = get_config(
    sim_rate    = 100,
    t_final     = 15.0,
    radius      = 2.0,
    wind_mode   = 'constant',
    wind_vec    = [3.0, 0.0, 0.0],
    aero        = 'std',
)

dt = 1.0 / cfg.sim_rate

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.simulate import simulate
from rotorpy.world import World

# Estimators
from rotorpy.estimators.wind_ekf import WindEKF
from rotorpy.estimators.wind_eqf import WindEqF

# =====================================================================
#  Build shared objects
# =====================================================================
params = get_quad_params(cfg)
params = copy.deepcopy(params)
params['use_bem'] = False

world = World.empty((-5, 5, -5, 5, -5, 5))
wind_profile = make_wind(cfg)
x0 = make_x0(cfg)

WIND_VEC = np.array(cfg.wind_vec)

# =====================================================================
#  Run 1: with EKF
# =====================================================================
print("=" * 60)
print("  Running simulation with EKF estimator...")
print("=" * 60)

vehicle_ekf = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
ctrl_ekf    = SE3Control(params)
vehicle_ekf.initial_state = copy.deepcopy(x0)
ekf = WindEKF(params, dt=dt)

imu, mocap = make_sensors(cfg)

(time_ekf, state_ekf, control_ekf, flat_ekf,
 imu_ekf, imu_gt_ekf, mocap_ekf, est_ekf, exit_ekf) = simulate(
    world, copy.deepcopy(x0), vehicle_ekf, ctrl_ekf,
    CircularTraj(radius=cfg.radius), wind_profile,
    imu, mocap, ekf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False)

print(f"  EKF: {exit_ekf.value}, sim time = {time_ekf[-1]:.2f}s")

# =====================================================================
#  Run 2: with EqF
# =====================================================================
print("\n" + "=" * 60)
print("  Running simulation with EqF estimator (Mahony)...")
print("=" * 60)

vehicle_eqf = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
ctrl_eqf    = SE3Control(params)
vehicle_eqf.initial_state = copy.deepcopy(x0)

# EqF with same initial conditions
eqf = WindEqF(params,
              R0=np.eye(3),
              v0=np.zeros(3),
              w0=np.zeros(3),       # start with zero wind estimate
              dt=dt)

# Need a fresh IMU and MoCap with the same seed behavior
imu2, mocap2 = make_sensors(cfg)

(time_eqf, state_eqf, control_eqf_out, flat_eqf,
 imu_eqf, imu_gt_eqf, mocap_eqf, est_eqf, exit_eqf) = simulate(
    world, copy.deepcopy(x0), vehicle_eqf, ctrl_eqf,
    CircularTraj(radius=cfg.radius), wind_profile,
    imu2, mocap2, eqf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False)

print(f"  EqF: {exit_eqf.value}, sim time = {time_eqf[-1]:.2f}s")

# =====================================================================
#  Extract EKF wind estimates
# =====================================================================
N_ekf = len(time_ekf)
ekf_wind = np.zeros((N_ekf, 3))
ekf_euler = np.zeros((N_ekf, 3))
for k in range(N_ekf):
    fs = est_ekf['filter_state'][k]
    if len(fs) >= 9:
        # EKF state: [psi, theta, phi, u, v, w_body, wx, wy, wz]
        ekf_euler[k] = np.degrees(fs[0:3])   # Euler angles
        ekf_wind[k]  = fs[6:9]               # wind in body frame
    # Convert EKF body-frame wind to world frame using true attitude
    q_true = state_ekf['q'][k]
    R_true = Rotation.from_quat(q_true).as_matrix()
    ekf_wind[k] = R_true @ ekf_wind[k]  # approximate: should use estimated R

# =====================================================================
#  Extract EqF wind estimates
# =====================================================================
N_eqf = len(time_eqf)
eqf_wind = np.zeros((N_eqf, 3))
eqf_euler = np.zeros((N_eqf, 3))
for k in range(N_eqf):
    fs = est_eqf['filter_state'][k]
    if len(fs) >= 15:
        # EqF state: [R(9), v(3), w(3)]
        R_est = fs[0:9].reshape(3, 3)
        eqf_wind[k]  = fs[12:15]             # wind in world frame (already!)
        eqf_euler[k] = Rotation.from_matrix(R_est).as_euler('xyz', degrees=True)

# =====================================================================
#  Plotting
# =====================================================================
print("\n" + "=" * 60)
print("  Generating comparison plots...")
print("=" * 60)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('EKF vs EqF (Mahony) Wind Estimation\n'
             f'Circular trajectory (r={cfg.radius}m), constant wind = {WIND_VEC} m/s',
             fontsize=13, fontweight='bold')

# (0,0) Wind X
ax = axes[0, 0]
ax.axhline(y=WIND_VEC[0], color='k', ls='--', alpha=0.5, label='True')
ax.plot(time_ekf, ekf_wind[:, 0], 'b', alpha=0.7, label='EKF')
ax.plot(time_eqf, eqf_wind[:, 0], 'r', alpha=0.7, label='EqF')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Wind x [m/s]')
ax.set_title('Wind Velocity — x component')
ax.legend(); ax.grid(True, alpha=0.3)

# (0,1) Wind Y
ax = axes[0, 1]
ax.axhline(y=WIND_VEC[1], color='k', ls='--', alpha=0.5, label='True')
ax.plot(time_ekf, ekf_wind[:, 1], 'b', alpha=0.7, label='EKF')
ax.plot(time_eqf, eqf_wind[:, 1], 'r', alpha=0.7, label='EqF')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Wind y [m/s]')
ax.set_title('Wind Velocity — y component')
ax.legend(); ax.grid(True, alpha=0.3)

# (1,0) Wind Z
ax = axes[1, 0]
ax.axhline(y=WIND_VEC[2], color='k', ls='--', alpha=0.5, label='True')
ax.plot(time_ekf, ekf_wind[:, 2], 'b', alpha=0.7, label='EKF')
ax.plot(time_eqf, eqf_wind[:, 2], 'r', alpha=0.7, label='EqF')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Wind z [m/s]')
ax.set_title('Wind Velocity — z component')
ax.legend(); ax.grid(True, alpha=0.3)

# (1,1) Wind magnitude error
ax = axes[1, 1]
# True wind at each time step (constant in this case)
wind_true = np.tile(WIND_VEC, (max(N_ekf, N_eqf), 1))
ekf_err = np.linalg.norm(ekf_wind - wind_true[:N_ekf], axis=1)
eqf_err = np.linalg.norm(eqf_wind - wind_true[:N_eqf], axis=1)
ax.plot(time_ekf, ekf_err, 'b', alpha=0.7, label='EKF')
ax.plot(time_eqf, eqf_err, 'r', alpha=0.7, label='EqF')
ax.set_xlabel('Time [s]'); ax.set_ylabel('||w_est - w_true|| [m/s]')
ax.set_title('Wind Estimation Error (magnitude)')
ax.legend(); ax.grid(True, alpha=0.3)

# (2,0) Pitch angle (estimated vs true)
ax = axes[2, 0]
true_pitch_ekf = np.array([Rotation.from_quat(state_ekf['q'][k]).as_euler('xyz', degrees=True)[1] for k in range(N_ekf)])
true_pitch_eqf = np.array([Rotation.from_quat(state_eqf['q'][k]).as_euler('xyz', degrees=True)[1] for k in range(N_eqf)])
ax.plot(time_ekf, true_pitch_ekf, 'k--', alpha=0.4, label='True')
ax.plot(time_ekf, ekf_euler[:, 1], 'b', alpha=0.7, label='EKF est.')
ax.plot(time_eqf, eqf_euler[:, 1], 'r', alpha=0.7, label='EqF est.')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Pitch [deg]')
ax.set_title('Pitch Angle Estimation')
ax.legend(); ax.grid(True, alpha=0.3)

# (2,1) Covariance trace
ax = axes[2, 1]
ekf_cov_trace = np.array([np.trace(est_ekf['covariance'][k]) if np.ndim(est_ekf['covariance'][k]) == 2 else 0 for k in range(N_ekf)])
eqf_cov_trace = np.array([np.trace(est_eqf['covariance'][k]) if np.ndim(est_eqf['covariance'][k]) == 2 else 0 for k in range(N_eqf)])
ax.plot(time_ekf, ekf_cov_trace, 'b', alpha=0.7, label='EKF')
ax.plot(time_eqf, eqf_cov_trace, 'r', alpha=0.7, label='EqF')
ax.set_xlabel('Time [s]'); ax.set_ylabel('tr(P)')
ax.set_title('Filter Covariance Trace')
ax.legend(); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
out_path = os.path.join(ROOT, 'ekf_vs_eqf_wind.png')
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")

# =====================================================================
#  Summary statistics
# =====================================================================
print(f"\n{'='*60}")
print(f"  Wind Estimation Summary (t > 5s, after convergence)")
print(f"{'='*60}")

idx_ekf = time_ekf > 5.0
idx_eqf = time_eqf > 5.0

metrics = ['Mean |err| [m/s]', 'Std |err| [m/s]', 'Max |err| [m/s]',
           'Mean wx err', 'Mean wy err', 'Mean wz err']
ekf_vals = [ekf_err[idx_ekf].mean(), ekf_err[idx_ekf].std(), ekf_err[idx_ekf].max(),
            (ekf_wind[idx_ekf, 0] - WIND_VEC[0]).mean(),
            (ekf_wind[idx_ekf, 1] - WIND_VEC[1]).mean(),
            (ekf_wind[idx_ekf, 2] - WIND_VEC[2]).mean()]
eqf_vals = [eqf_err[idx_eqf].mean(), eqf_err[idx_eqf].std(), eqf_err[idx_eqf].max(),
            (eqf_wind[idx_eqf, 0] - WIND_VEC[0]).mean(),
            (eqf_wind[idx_eqf, 1] - WIND_VEC[1]).mean(),
            (eqf_wind[idx_eqf, 2] - WIND_VEC[2]).mean()]

print(f"{'Metric':<22} {'EKF':>12} {'EqF':>12}")
print("-" * 48)
for m, ev, qv in zip(metrics, ekf_vals, eqf_vals):
    print(f"{m:<22} {ev:>12.4f} {qv:>12.4f}")

print("\nDone.")
