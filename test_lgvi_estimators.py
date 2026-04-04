"""
LGVI integrator + all estimators closed-loop comparison.

Plant: Lee-Leok-McClamroch LGVI on SE(3) (integrator='lgvi')
Estimators:
  1. Ground truth baseline
  2. EKF INS  (Euler-angle)
  3. UKF INS  (Euler-angle, sigma-point)
  4. L-IEKF   (Left-Invariant EKF, world-frame error)
  5. R-IEKF   (Right-Invariant EKF / EqF, body-frame error)
"""

import sys, os, copy, time as timer
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
cfg = get_config(
    sim_rate    = 500,
    t_final     = 15.0,
    radius      = 2.0,
    wind_mode   = 'constant',
    wind_vec    = [3.0, 0.0, 0.0],
    aero        = 'std',
    integrator  = 'lgvi',
)
dt = 1.0 / cfg.sim_rate

world = World.empty((-5, 5, -5, 5, -5, 5))
wind_profile = make_wind(cfg)
x0 = make_x0(cfg)
quad_params = get_quad_params(cfg)

# =====================================================================
#  Run all configurations
# =====================================================================
configs = [
    ('GT',    None,     False),
    ('EKF',   EKFINS,   True),
    ('UKF',   UKFINS,   True),
    ('LIEKF', LIEKFINS, True),
    ('RIEKF', EqFINS,   True),
]

results = {}

for i, (name, EstClass, use_est) in enumerate(configs):
    print(f"{'='*60}")
    print(f"  Run {i+1}/{len(configs)}: {name} {'(LGVI plant)' if use_est else '(GT baseline, LGVI plant)'}")
    print(f"{'='*60}")

    params = copy.deepcopy(quad_params)
    params['use_bem'] = False
    v = Multirotor(params, control_abstraction='cmd_motor_speeds', integrator='lgvi')
    v.initial_state = copy.deepcopy(x0)
    imu_s, mc_s = make_sensors(cfg)

    if use_est:
        est = EstClass(quad_params, dt=dt)
    else:
        est = NullEstimator()

    t0 = timer.perf_counter()
    (t, s, c, f, _, _, _, est_out, ex) = simulate(
        world, copy.deepcopy(x0), v, SE3Control(quad_params),
        CircularTraj(radius=cfg.radius), wind_profile,
        imu_s, mc_s, est,
        cfg.t_final, dt, safety_margin=0.25,
        use_mocap=False, use_estimator=use_est)
    wall = timer.perf_counter() - t0

    results[name] = {'t': t, 's': s, 'f': f, 'est': est_out, 'exit': ex, 'wall': wall}
    print(f"  Exit: {ex.value}, t={t[-1]:.2f}s, wall={wall:.2f}s\n")

# =====================================================================
#  Metrics
# =====================================================================
def tracking_error(r):
    return np.linalg.norm(r['s']['x'] - r['f']['x'], axis=1)

def extract_errors(r):
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

# =====================================================================
#  Plotting
# =====================================================================
print(f"{'='*60}\n  Generating plots...\n{'='*60}")

all_filters = ['GT', 'EKF', 'UKF', 'LIEKF', 'RIEKF']
est_filters = ['EKF', 'UKF', 'LIEKF', 'RIEKF']
colors = {'GT': 'k', 'EKF': 'b', 'UKF': 'g', 'LIEKF': 'm', 'RIEKF': 'r'}
labels = {'GT': 'GT baseline', 'EKF': 'EKF INS', 'UKF': 'UKF INS',
          'LIEKF': 'L-IEKF', 'RIEKF': 'R-IEKF (EqF)'}

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('LGVI Plant + EKF / UKF / L-IEKF / R-IEKF Closed-Loop\n'
             f'Circular traj (r={cfg.radius}m), wind = {cfg.wind_vec} m/s',
             fontsize=13, fontweight='bold')

# (0,0) XY trajectory
ax = axes[0, 0]
r0 = results['GT']
ax.plot(r0['f']['x'][:, 0], r0['f']['x'][:, 1], 'k--', alpha=0.3, label='Desired', lw=1)
for name in all_filters:
    r = results[name]
    ax.plot(r['s']['x'][:, 0], r['s']['x'][:, 1], colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Tracking error
ax = axes[0, 1]
for name in all_filters:
    r = results[name]
    ax.plot(r['t'], tracking_error(r), colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,0) Position estimation error
ax = axes[1, 0]
for name in est_filters:
    pe, _, _ = extract_errors(results[name])
    ax.plot(results[name]['t'], pe, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||p_est - p_true|| [m]')
ax.set_title('Position Estimation Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,1) Attitude estimation error
ax = axes[1, 1]
for name in est_filters:
    _, ae, _ = extract_errors(results[name])
    ax.plot(results[name]['t'], ae, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('Attitude error [deg]')
ax.set_title('Attitude Estimation Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (2,0) Velocity estimation error
ax = axes[2, 0]
for name in est_filters:
    _, _, ve = extract_errors(results[name])
    ax.plot(results[name]['t'], ve, colors[name], alpha=0.7, label=labels[name])
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
out_path = os.path.join(ROOT, 'lgvi_estimators_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")

# =====================================================================
#  Summary Tables
# =====================================================================
print(f"\n{'='*75}")
print(f"  LGVI Plant — Closed-Loop Tracking Performance (t > 3s)")
print(f"{'='*75}")
print(f"{'Filter':<16} {'Mean [m]':>10} {'Max [m]':>10} {'Wall [s]':>10} {'Exit':>25}")
print("-" * 75)
for name in all_filters:
    r = results[name]
    idx = r['t'] > 3.0
    e = tracking_error(r)
    if np.sum(idx) > 0:
        print(f"{labels[name]:<16} {e[idx].mean():>10.4f} {e[idx].max():>10.4f} {r['wall']:>10.2f} {r['exit'].value:>25}")
    else:
        print(f"{labels[name]:<16} {'crashed':>10} {'':>10} {r['wall']:>10.2f} {r['exit'].value:>25}")

print(f"\n{'='*60}")
print(f"  LGVI Plant — State Estimation Accuracy (t > 3s)")
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
