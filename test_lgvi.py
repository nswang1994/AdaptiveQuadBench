"""
Compare RK45 vs LGVI (Lie-Störmer-Verlet) integrators.

Both run the SAME scenario: circular trajectory with 3 m/s wind.
Metrics:
  1. Tracking performance (how well the controller tracks)
  2. SO(3) constraint violation: det(R)-1 and R^T R - I  (LGVI should be ~eps)
  3. Energy-like metric (kinetic + potential)
  4. Computational cost
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

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import ConstantWind
from rotorpy.simulate import simulate
from rotorpy.world import World

from sim_config import get_config, make_x0, make_sensors, get_quad_params, NullEstimator

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

world = World.empty((-5, 5, -5, 5, -5, 5))
wind_profile = ConstantWind(*np.array(cfg.wind_vec))
quad_params = get_quad_params(cfg)
x0 = make_x0(cfg)

results = {}

for integ_name in ['rk45', 'lgvi']:
    print(f"{'='*60}\n  Running {integ_name.upper()} integrator\n{'='*60}")
    params = copy.deepcopy(quad_params)
    params['use_bem'] = False

    v = Multirotor(params, control_abstraction='cmd_motor_speeds', integrator=integ_name)
    v.initial_state = copy.deepcopy(x0)
    imu, mc = make_sensors(cfg)

    t0 = timer.perf_counter()
    (t, s, c, f, _, _, _, _, ex) = simulate(
        world, copy.deepcopy(x0), v, SE3Control(quad_params),
        CircularTraj(radius=cfg.radius), wind_profile,
        imu, mc, NullEstimator(),
        cfg.t_final, dt, safety_margin=0.25, use_mocap=False)
    wall_time = timer.perf_counter() - t0

    results[integ_name] = {'t': t, 's': s, 'f': f, 'exit': ex, 'wall': wall_time}
    print(f"  Exit: {ex.value}, t={t[-1]:.2f}s, wall={wall_time:.2f}s")

# =====================================================================
#  Compute metrics
# =====================================================================
def so3_violation(s):
    """det(R)-1 and ||R^T R - I||_F for each timestep."""
    N = len(s['q'])
    det_err = np.zeros(N)
    orth_err = np.zeros(N)
    for k in range(N):
        R = Rotation.from_quat(s['q'][k]).as_matrix()
        det_err[k] = abs(np.linalg.det(R) - 1.0)
        orth_err[k] = np.linalg.norm(R.T @ R - np.eye(3), 'fro')
    return det_err, orth_err

def energy(s, mass, g, inertia):
    """Translational KE + rotational KE + gravitational PE."""
    N = len(s['x'])
    E = np.zeros(N)
    for k in range(N):
        v = s['v'][k]
        w = s['w'][k]
        h = s['x'][k][2]
        E[k] = 0.5 * mass * np.dot(v, v) + 0.5 * w @ inertia @ w + mass * g * h
    return E

def tracking_error(s, f):
    return np.linalg.norm(s['x'] - f['x'], axis=1)

J = np.array([[quad_params['Ixx'], quad_params['Ixy'], quad_params['Ixz']],
              [quad_params['Ixy'], quad_params['Iyy'], quad_params['Iyz']],
              [quad_params['Ixz'], quad_params['Iyz'], quad_params['Izz']]])

# =====================================================================
#  Plotting
# =====================================================================
print(f"\n{'='*60}\n  Generating plots...\n{'='*60}")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('RK45 vs LGVI (Lie-Störmer-Verlet) Integrator Comparison\n'
             f'Circular traj (r={cfg.radius}m), wind = {cfg.wind_vec} m/s, dt = {dt}s',
             fontsize=13, fontweight='bold')
colors = {'rk45': 'b', 'lgvi': 'r'}
labels = {'rk45': 'RK45 (scipy)', 'lgvi': 'LGVI (Lie-SV)'}

# (0,0) XY trajectory
ax = axes[0, 0]
r0 = results['rk45']
ax.plot(r0['f']['x'][:, 0], r0['f']['x'][:, 1], 'k--', alpha=0.3, label='Desired', lw=1)
for name in ['rk45', 'lgvi']:
    r = results[name]
    ax.plot(r['s']['x'][:, 0], r['s']['x'][:, 1], colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Position tracking error
ax = axes[0, 1]
for name in ['rk45', 'lgvi']:
    r = results[name]
    ax.plot(r['t'], tracking_error(r['s'], r['f']), colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (1,0) SO(3) det violation
ax = axes[1, 0]
for name in ['rk45', 'lgvi']:
    det_err, _ = so3_violation(results[name]['s'])
    ax.plot(results[name]['t'], det_err, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('|det(R) - 1|')
ax.set_title('SO(3) Determinant Violation'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (1,1) SO(3) orthogonality violation
ax = axes[1, 1]
for name in ['rk45', 'lgvi']:
    _, orth_err = so3_violation(results[name]['s'])
    ax.plot(results[name]['t'], orth_err, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||R^T R - I||_F')
ax.set_title('SO(3) Orthogonality Violation'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# (2,0) Mechanical energy
ax = axes[2, 0]
for name in ['rk45', 'lgvi']:
    r = results[name]
    E = energy(r['s'], quad_params['mass'], 9.81, J)
    ax.plot(r['t'], E, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('E [J]')
ax.set_title('Total Mechanical Energy (KE + PE)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# (2,1) Angular velocity magnitude
ax = axes[2, 1]
for name in ['rk45', 'lgvi']:
    r = results[name]
    w_mag = np.linalg.norm(r['s']['w'], axis=1)
    ax.plot(r['t'], w_mag, colors[name], alpha=0.7, label=labels[name])
ax.set_xlabel('Time [s]'); ax.set_ylabel('||w|| [rad/s]')
ax.set_title('Angular Velocity Magnitude'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ROOT, 'rk45_vs_lgvi_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")

# =====================================================================
#  Summary
# =====================================================================
print(f"\n{'='*65}")
print(f"  RK45 vs LGVI Summary")
print(f"{'='*65}")
print(f"{'Metric':<35} {'RK45':>12} {'LGVI':>12}")
print("-" * 65)

for name in ['rk45', 'lgvi']:
    r = results[name]
    idx = r['t'] > 3.0
    e = tracking_error(r['s'], r['f'])
    det_err, orth_err = so3_violation(r['s'])
    results[name]['track_mean'] = e[idx].mean()
    results[name]['det_max'] = det_err.max()
    results[name]['orth_max'] = orth_err.max()

print(f"{'Track err mean (t>3s) [m]':<35} {results['rk45']['track_mean']:>12.4f} {results['lgvi']['track_mean']:>12.4f}")
print(f"{'Track err max  (t>3s) [m]':<35} {tracking_error(results['rk45']['s'], results['rk45']['f'])[results['rk45']['t']>3].max():>12.4f} {tracking_error(results['lgvi']['s'], results['lgvi']['f'])[results['lgvi']['t']>3].max():>12.4f}")
print(f"{'SO(3) det violation max':<35} {results['rk45']['det_max']:>12.2e} {results['lgvi']['det_max']:>12.2e}")
print(f"{'SO(3) orth violation max':<35} {results['rk45']['orth_max']:>12.2e} {results['lgvi']['orth_max']:>12.2e}")
print(f"{'Wall time [s]':<35} {results['rk45']['wall']:>12.2f} {results['lgvi']['wall']:>12.2f}")

print("\nDone.")
