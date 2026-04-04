"""
Closed-loop test: EqF INS (Right-IEKF on SE_2(3)) feeding state estimates
to the SE3 controller, compared with ground-truth control baseline.

Three runs on the SAME trajectory and wind:
  1. Ground truth  — controller receives true state (ideal baseline)
  2. MoCap         — controller receives noisy MoCap measurements
  3. EqF INS       — controller receives EqF-estimated state

Vehicle: 0.826 kg X-config quadrotor
Trajectory: circular (r=2 m)
Wind: constant 3 m/s in +x
Duration: 15 s
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
from rotorpy.estimators.wind_ekf import WindEKF
from rotorpy.estimators.eqf_ins import EqFINS

params = get_quad_params(cfg)
params = copy.deepcopy(params)
params['use_bem'] = False

world = World.empty((-5, 5, -5, 5, -5, 5))
wind_profile = make_wind(cfg)
x0 = make_x0(cfg)

# =====================================================================
#  Run 1: Ground truth (baseline)
# =====================================================================
print("=" * 60)
print("  Run 1: Ground truth feedback (baseline)")
print("=" * 60)

vehicle1 = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
vehicle1.initial_state = copy.deepcopy(x0)
imu1, mocap1 = make_sensors(cfg)

(t1, s1, c1, f1, _, _, _, _, exit1) = simulate(
    world, copy.deepcopy(x0), vehicle1, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu1, mocap1, NullEstimator(),
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False)
print(f"  Exit: {exit1.value}, t = {t1[-1]:.2f}s")

# =====================================================================
#  Run 2: MoCap feedback
# =====================================================================
print("\n" + "=" * 60)
print("  Run 2: MoCap feedback")
print("=" * 60)

vehicle2 = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
vehicle2.initial_state = copy.deepcopy(x0)
imu2, mocap2 = make_sensors(cfg)

(t2, s2, c2, f2, _, _, _, _, exit2) = simulate(
    world, copy.deepcopy(x0), vehicle2, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu2, mocap2, NullEstimator(),
    cfg.t_final, dt, safety_margin=0.25, use_mocap=True)
print(f"  Exit: {exit2.value}, t = {t2[-1]:.2f}s")

# =====================================================================
#  Run 3: EqF INS closed-loop
# =====================================================================
print("\n" + "=" * 60)
print("  Run 3: EqF INS (Right-IEKF) closed-loop")
print("=" * 60)

vehicle3 = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
vehicle3.initial_state = copy.deepcopy(x0)
imu3, mocap3 = make_sensors(cfg)

eqf = EqFINS(params, dt=dt)

(t3, s3, c3, f3, imu3_meas, _, mocap3_meas, est3, exit3) = simulate(
    world, copy.deepcopy(x0), vehicle3, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu3, mocap3, eqf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False, use_estimator=True)
print(f"  Exit: {exit3.value}, t = {t3[-1]:.2f}s")

# =====================================================================
#  Compute metrics
# =====================================================================
def pos_err(s, f, t):
    return np.linalg.norm(s['x'] - f['x'], axis=1)

e1 = pos_err(s1, f1, t1)
e2 = pos_err(s2, f2, t2)
e3 = pos_err(s3, f3, t3)

# =====================================================================
#  Extract EqF estimation errors
# =====================================================================
N3 = len(t3)
est_pos_err = np.zeros(N3)
est_att_err = np.zeros(N3)
est_vel_err = np.zeros(N3)
for k in range(N3):
    fs = est3['filter_state'][k]
    if len(fs) >= 15:
        R_est = fs[0:9].reshape(3, 3)
        v_est = fs[9:12]
        p_est = fs[12:15]
        est_pos_err[k] = np.linalg.norm(p_est - s3['x'][k])
        est_vel_err[k] = np.linalg.norm(v_est - s3['v'][k])
        R_true = Rotation.from_quat(s3['q'][k]).as_matrix()
        est_att_err[k] = np.degrees(np.linalg.norm(
            Rotation.from_matrix(R_est.T @ R_true).as_rotvec()))

# =====================================================================
#  Plotting
# =====================================================================
print("\n" + "=" * 60)
print("  Generating plots...")
print("=" * 60)

WIND_VEC = np.array(cfg.wind_vec)

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Closed-Loop EqF INS vs Ground Truth vs MoCap\n'
             f'Circular traj (r={cfg.radius}m), wind = {WIND_VEC} m/s',
             fontsize=13, fontweight='bold')

# (0,0) XY trajectory
ax = axes[0, 0]
ax.plot(f1['x'][:, 0], f1['x'][:, 1], 'k--', alpha=0.4, label='Desired')
ax.plot(s1['x'][:, 0], s1['x'][:, 1], 'b', alpha=0.7, label='GT ctrl')
ax.plot(s2['x'][:, 0], s2['x'][:, 1], 'g', alpha=0.7, label='MoCap ctrl')
ax.plot(s3['x'][:, 0], s3['x'][:, 1], 'r', alpha=0.7, label='EqF ctrl')
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Position tracking error
ax = axes[0, 1]
ax.plot(t1, e1, 'b', alpha=0.7, label='GT ctrl')
ax.plot(t2, e2, 'g', alpha=0.7, label='MoCap ctrl')
ax.plot(t3, e3, 'r', alpha=0.7, label='EqF ctrl')
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(); ax.grid(True, alpha=0.3)

# (1,0) EqF position estimation error
ax = axes[1, 0]
ax.plot(t3, est_pos_err, 'r', alpha=0.7)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||p_est - p_true|| [m]')
ax.set_title('EqF Position Estimation Error'); ax.grid(True, alpha=0.3)

# (1,1) EqF attitude estimation error
ax = axes[1, 1]
ax.plot(t3, est_att_err, 'r', alpha=0.7)
ax.set_xlabel('Time [s]'); ax.set_ylabel('Attitude error [deg]')
ax.set_title('EqF Attitude Estimation Error'); ax.grid(True, alpha=0.3)

# (2,0) EqF velocity estimation error
ax = axes[2, 0]
ax.plot(t3, est_vel_err, 'r', alpha=0.7)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||v_est - v_true|| [m/s]')
ax.set_title('EqF Velocity Estimation Error'); ax.grid(True, alpha=0.3)

# (2,1) Covariance trace
ax = axes[2, 1]
cov_trace = np.array([np.trace(est3['covariance'][k])
                       if np.ndim(est3['covariance'][k]) == 2 else 0
                       for k in range(N3)])
ax.plot(t3, cov_trace, 'r', alpha=0.7)
ax.set_xlabel('Time [s]'); ax.set_ylabel('tr(P)')
ax.set_title('EqF Covariance Trace'); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
out_path = os.path.join(ROOT, 'eqf_closedloop_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")

# =====================================================================
#  Summary
# =====================================================================
print(f"\n{'='*60}")
print(f"  Tracking Performance Summary (t > 3s)")
print(f"{'='*60}")

for label, t, e, ex in [('GT ctrl',    t1, e1, exit1),
                          ('MoCap ctrl', t2, e2, exit2),
                          ('EqF ctrl',   t3, e3, exit3)]:
    idx = t > 3.0
    if np.sum(idx) > 0:
        print(f"  {label:<12}  mean={e[idx].mean():.4f}m  std={e[idx].std():.4f}m  "
              f"max={e[idx].max():.4f}m  exit={ex.value}")
    else:
        print(f"  {label:<12}  (crashed before t=3s)  exit={ex.value}")

if N3 > 0:
    idx3 = t3 > 3.0
    if np.sum(idx3) > 0:
        print(f"\n  EqF estimation (t>3s):")
        print(f"    Position: mean={est_pos_err[idx3].mean():.4f}m, max={est_pos_err[idx3].max():.4f}m")
        print(f"    Attitude: mean={est_att_err[idx3].mean():.2f}°, max={est_att_err[idx3].max():.2f}°")
        print(f"    Velocity: mean={est_vel_err[idx3].mean():.4f}m/s, max={est_vel_err[idx3].max():.4f}m/s")

print("\nDone.")
