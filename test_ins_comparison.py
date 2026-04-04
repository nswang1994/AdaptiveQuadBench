"""
Closed-loop comparison: EKF vs UKF vs L-IEKF vs R-IEKF (EqF) for INS.

All estimators have the SAME 15-dim state:
    [attitude(3), velocity(3), position(3), gyro_bias(3), accel_bias(3)]

Runs:
  1. Ground truth baseline
  2. EKF INS  (Euler-angle parameterization)
  3. UKF INS  (Euler-angle, sigma-point)
  4. L-IEKF   (Left-Invariant EKF on SE_2(3), world-frame error)
  5. R-IEKF   (Right-Invariant EKF / EqF on SE_2(3), body-frame error)

Vehicle: 0.826 kg X-config quadrotor
Trajectory: circular (r=2 m), Wind: 3 m/s in +x, Duration: 15 s
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

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.simulate import simulate
from rotorpy.world import World

from rotorpy.estimators.ekf_ins import EKFINS
from rotorpy.estimators.ukf_ins import UKFINS
from rotorpy.estimators.liekf_ins import LIEKFINS
from rotorpy.estimators.eqf_ins import EqFINS

# =====================================================================
#  Configuration
# =====================================================================
cfg = get_config(
    sim_rate    = 100,
    t_final     = 15.0,
    radius      = 2.0,
    wind_mode   = 'constant',
    wind_vec    = [3.0, 0.0, 0.0],
    aero        = 'std',
)

dt = 1.0 / cfg.sim_rate
params = copy.deepcopy(get_quad_params(cfg))
params['use_bem'] = False
x0 = make_x0(cfg)

world = World.empty((-5, 5, -5, 5, -5, 5))
wind_profile = make_wind(cfg)

# =====================================================================
#  Run simulations
# =====================================================================
results = {}

# --- GT baseline ---
print("=" * 60)
print("  Run 1/5: Ground truth baseline")
print("=" * 60)
v = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
v.initial_state = copy.deepcopy(x0)
imu_s, mc_s = make_sensors(cfg)
(t, s, c, f, _, _, _, est, ex) = simulate(
    world, copy.deepcopy(x0), v, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu_s, mc_s, NullEstimator(),
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False)
results['GT'] = {'t': t, 's': s, 'f': f, 'est': est, 'exit': ex}
print(f"  Exit: {ex.value}, t = {t[-1]:.2f}s")

# --- EKF INS ---
print("\n" + "=" * 60)
print("  Run 2/5: EKF INS closed-loop")
print("=" * 60)
v = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
v.initial_state = copy.deepcopy(x0)
imu_s, mc_s = make_sensors(cfg)
ekf = EKFINS(params, dt=dt)
(t, s, c, f, _, _, _, est, ex) = simulate(
    world, copy.deepcopy(x0), v, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu_s, mc_s, ekf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False, use_estimator=True)
results['EKF'] = {'t': t, 's': s, 'f': f, 'est': est, 'exit': ex}
print(f"  Exit: {ex.value}, t = {t[-1]:.2f}s")

# --- UKF INS ---
print("\n" + "=" * 60)
print("  Run 3/5: UKF INS closed-loop")
print("=" * 60)
v = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
v.initial_state = copy.deepcopy(x0)
imu_s, mc_s = make_sensors(cfg)
ukf = UKFINS(params, dt=dt)
(t, s, c, f, _, _, _, est, ex) = simulate(
    world, copy.deepcopy(x0), v, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu_s, mc_s, ukf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False, use_estimator=True)
results['UKF'] = {'t': t, 's': s, 'f': f, 'est': est, 'exit': ex}
print(f"  Exit: {ex.value}, t = {t[-1]:.2f}s")

# --- L-IEKF ---
print("\n" + "=" * 60)
print("  Run 4/5: L-IEKF (Left-Invariant EKF) closed-loop")
print("=" * 60)
v = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
v.initial_state = copy.deepcopy(x0)
imu_s, mc_s = make_sensors(cfg)
liekf = LIEKFINS(params, dt=dt)
(t, s, c, f, _, _, _, est, ex) = simulate(
    world, copy.deepcopy(x0), v, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu_s, mc_s, liekf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False, use_estimator=True)
results['LIEKF'] = {'t': t, 's': s, 'f': f, 'est': est, 'exit': ex}
print(f"  Exit: {ex.value}, t = {t[-1]:.2f}s")

# --- R-IEKF (EqF) ---
print("\n" + "=" * 60)
print("  Run 5/5: R-IEKF (Right-Invariant EKF / EqF) closed-loop")
print("=" * 60)
v = Multirotor(copy.deepcopy(params), control_abstraction='cmd_motor_speeds')
v.initial_state = copy.deepcopy(x0)
imu_s, mc_s = make_sensors(cfg)
eqf = EqFINS(params, dt=dt)
(t, s, c, f, _, _, _, est, ex) = simulate(
    world, copy.deepcopy(x0), v, SE3Control(params),
    CircularTraj(radius=cfg.radius), wind_profile,
    imu_s, mc_s, eqf,
    cfg.t_final, dt, safety_margin=0.25, use_mocap=False, use_estimator=True)
results['RIEKF'] = {'t': t, 's': s, 'f': f, 'est': est, 'exit': ex}
print(f"  Exit: {ex.value}, t = {t[-1]:.2f}s")

# =====================================================================
#  Extract estimation errors
# =====================================================================
def extract_errors(r):
    """Extract position, attitude, velocity estimation errors."""
    t = r['t']; s = r['s']; est = r['est']
    N = len(t)
    pos_err = np.zeros(N)
    att_err = np.zeros(N)
    vel_err = np.zeros(N)
    for k in range(N):
        fs = est['filter_state'][k]
        if len(fs) >= 15:
            R_est = fs[0:9].reshape(3, 3)
            v_est = fs[9:12]
            p_est = fs[12:15]
            pos_err[k] = np.linalg.norm(p_est - s['x'][k])
            vel_err[k] = np.linalg.norm(v_est - s['v'][k])
            R_true = Rotation.from_quat(s['q'][k]).as_matrix()
            att_err[k] = np.degrees(np.linalg.norm(
                Rotation.from_matrix(R_est.T @ R_true).as_rotvec()))
    return pos_err, att_err, vel_err

def tracking_error(r):
    return np.linalg.norm(r['s']['x'] - r['f']['x'], axis=1)

# =====================================================================
#  Plotting
# =====================================================================
print("\n" + "=" * 60)
print("  Generating plots...")
print("=" * 60)

all_filters = ['GT', 'EKF', 'UKF', 'LIEKF', 'RIEKF']
est_filters = ['EKF', 'UKF', 'LIEKF', 'RIEKF']
colors = {'GT': 'k', 'EKF': 'b', 'UKF': 'g', 'LIEKF': 'm', 'RIEKF': 'r'}
labels = {'GT': 'GT baseline', 'EKF': 'EKF INS', 'UKF': 'UKF INS',
          'LIEKF': 'L-IEKF', 'RIEKF': 'R-IEKF'}

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('EKF vs UKF vs L-IEKF vs R-IEKF — Closed-Loop INS Control\n'
             f'Circular traj (r={cfg.radius}m), wind = {cfg.wind_vec} m/s',
             fontsize=13, fontweight='bold')

# (0,0) XY trajectory
ax = axes[0, 0]
r0 = results['GT']
ax.plot(r0['f']['x'][:, 0], r0['f']['x'][:, 1], 'k--', alpha=0.3, label='Desired', lw=1)
for name in all_filters:
    r = results[name]
    ax.plot(r['s']['x'][:, 0], r['s']['x'][:, 1], colors[name],
            alpha=0.7, label=labels[name])
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Position tracking error
ax = axes[0, 1]
for name in all_filters:
    r = results[name]
    ax.plot(r['t'], tracking_error(r), colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,0) Position estimation error
ax = axes[1, 0]
for name in est_filters:
    r = results[name]
    pe, ae, ve = extract_errors(r)
    ax.plot(r['t'], pe, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||p_est - p_true|| [m]')
ax.set_title('Position Estimation Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,1) Attitude estimation error
ax = axes[1, 1]
for name in est_filters:
    r = results[name]
    pe, ae, ve = extract_errors(r)
    ax.plot(r['t'], ae, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('Attitude error [deg]')
ax.set_title('Attitude Estimation Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (2,0) Velocity estimation error
ax = axes[2, 0]
for name in est_filters:
    r = results[name]
    pe, ae, ve = extract_errors(r)
    ax.plot(r['t'], ve, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||v_est - v_true|| [m/s]')
ax.set_title('Velocity Estimation Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (2,1) Covariance trace
ax = axes[2, 1]
for name in est_filters:
    r = results[name]
    N = len(r['t'])
    tr_P = np.array([np.trace(r['est']['covariance'][k])
                      if np.ndim(r['est']['covariance'][k]) == 2 else 0
                      for k in range(N)])
    ax.plot(r['t'], tr_P, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('tr(P)')
ax.set_title('Filter Covariance Trace'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
out_path = os.path.join(ROOT, 'ekf_ukf_liekf_riekf_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")

# =====================================================================
#  Summary Table
# =====================================================================
print(f"\n{'='*75}")
print(f"  Closed-Loop Tracking Performance (t > 3s)")
print(f"{'='*75}")
print(f"{'Filter':<16} {'Track err mean':>14} {'Track err max':>14} {'Exit':>25}")
print("-" * 75)
for name in all_filters:
    r = results[name]
    idx = r['t'] > 3.0
    e = tracking_error(r)
    if np.sum(idx) > 0:
        print(f"{labels[name]:<16} {e[idx].mean():>14.4f} m {e[idx].max():>13.4f} m {r['exit'].value:>25}")
    else:
        print(f"{labels[name]:<16} {'(crashed)':>14} {'':>14} {r['exit'].value:>25}")

print(f"\n{'='*60}")
print(f"  State Estimation Accuracy (t > 3s)")
print(f"{'='*60}")
print(f"{'Filter':<16} {'Pos [mm]':>10} {'Att [deg]':>10} {'Vel [m/s]':>10}")
print("-" * 50)
for name in est_filters:
    r = results[name]
    idx = r['t'] > 3.0
    pe, ae, ve = extract_errors(r)
    if np.sum(idx) > 0:
        print(f"{labels[name]:<16} {pe[idx].mean()*1000:>10.1f} {ae[idx].mean():>10.2f} {ve[idx].mean():>10.4f}")

print("\nDone.")
