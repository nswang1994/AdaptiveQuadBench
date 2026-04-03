"""
Controller benchmark under:  LGVI + BEM + L-IEKF (500 Hz)

Controllers:
  1. SE3Control            — RotorPy baseline PD on SE(3)
  2. GeoControl            — Lee geometric tracking (T. Lee, Automatica 2011)
  3. GeometricAdaptive     — geometric + adaptive uncertainty compensation
  4. L1_GeoControl         — Lee geometric + L1 adaptive augmentation
  5. INDIAdaptive          — Incremental Nonlinear Dynamic Inversion + adaptive
  6. MPC                   — Nonlinear Model Predictive Control (CasADi)
  7. L1_MPC                — MPC + L1 adaptive augmentation

Plant : LGVI integrator, BEM aero (quadrotor_with_bem)
Est.  : L-IEKF (Left-Invariant EKF) closed-loop
Traj  : circular r=2m
Wind  : 3 m/s in +x
Rate  : 500 Hz
"""

import sys, os, copy, time as timer, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
sys.path.insert(0, os.path.join(ROOT, 'controller'))

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import ConstantWind
from rotorpy.sensors.imu import Imu
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.simulate import simulate
from rotorpy.world import World
from rotorpy.estimators.liekf_ins import LIEKFINS

# Controllers
from rotorpy.controllers.quadrotor_control import SE3Control
from geometric_control import GeoControl
from geometric_adaptive_controller import GeometricAdaptiveController
from geometric_control_l1 import L1_GeoControl
from indi_adaptive_controller import INDIAdaptiveController
try:
    from quadrotor_control_mpc import ModelPredictiveControl
    from quadrotor_control_mpc_l1 import L1_ModelPredictiveControl
    HAS_MPC = True
except ImportError as _e:
    print(f"  [WARN] MPC unavailable ({_e}), skipping MPC controllers.")
    HAS_MPC = False

# Vehicle with BEM
from quad_param.quadrotor_with_bem import quad_params as bem_params

# =====================================================================
SIM_RATE = 100
T_FINAL  = 15
RADIUS   = 2.0
WIND_VEC = np.array([3.0, 0.0, 0.0])
dt = 1.0 / SIM_RATE

world = World.empty((-10, 10, -10, 10, -10, 10))
wind_profile = ConstantWind(*WIND_VEC)

hover_omega = np.sqrt(bem_params['mass'] * 9.81 / (4 * bem_params['k_eta']))
x0 = {'x': np.array([RADIUS, 0, 0]),
      'v': np.zeros(3),
      'q': np.array([0, 0, 0, 1]),
      'w': np.zeros(3),
      'wind': np.array([0, 0, 0]),
      'rotor_speeds': np.array([hover_omega]*4)}

mocap_params = {
    'pos_noise_density':  0.0005 * np.ones(3),
    'vel_noise_density':  0.0010 * np.ones(3),
    'att_noise_density':  0.0005 * np.ones(3),
    'rate_noise_density': 0.0005 * np.ones(3),
    'vel_artifact_max': 5, 'vel_artifact_prob': 0.001,
    'rate_artifact_max': 1, 'rate_artifact_prob': 0.0002
}

def make_sensors():
    imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=SIM_RATE)
    mc = MotionCapture(sampling_rate=SIM_RATE, mocap_params=mocap_params, with_artifacts=False)
    return imu, mc

traj = CircularTraj(radius=RADIUS)

# Build controller list: (name, constructor_fn)
controllers = [
    ('SE3',         lambda p: SE3Control(p)),
    ('Geo',         lambda p: GeoControl(p)),
    ('GeoAdaptive', lambda p: GeometricAdaptiveController(p, dt=dt)),
    ('L1-Geo',      lambda p: L1_GeoControl(p)),
    ('INDI',        lambda p: INDIAdaptiveController(p, dt=dt)),
]
if HAS_MPC:
    controllers.append(('MPC', lambda p: ModelPredictiveControl(p, trajectory=traj, sim_rate=SIM_RATE, t_final=T_FINAL, t_horizon=0.5, n_nodes=10)))
    controllers.append(('L1-MPC', lambda p: L1_ModelPredictiveControl(p, trajectory=traj, sim_rate=SIM_RATE, t_final=T_FINAL, t_horizon=0.5, n_nodes=10)))

# =====================================================================
#  Run all controllers
# =====================================================================
results = {}

for i, (name, ctrl_fn) in enumerate(controllers):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(controllers)}] {name}  (LGVI + BEM + L-IEKF @ {SIM_RATE} Hz)")
    print(f"{'='*60}")

    try:
        params = copy.deepcopy(bem_params)
        vehicle = Multirotor(params, control_abstraction='cmd_motor_speeds', integrator='lgvi')
        vehicle.initial_state = copy.deepcopy(x0)
        imu_s, mc_s = make_sensors()
        est = LIEKFINS(bem_params, dt=dt)
        ctrl = ctrl_fn(bem_params)

        t0 = timer.perf_counter()
        (t, s, c, f, _, _, _, est_out, ex) = simulate(
            world, copy.deepcopy(x0), vehicle, ctrl,
            CircularTraj(radius=RADIUS), wind_profile,
            imu_s, mc_s, est,
            T_FINAL, dt, safety_margin=0.25,
            use_mocap=False, use_estimator=True)
        wall = timer.perf_counter() - t0

        results[name] = {'t': t, 's': s, 'f': f, 'est': est_out, 'exit': ex, 'wall': wall}
        print(f"  Exit: {ex.value}, t={t[-1]:.2f}s, wall={wall:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results[name] = None

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
print(f"\n{'='*60}\n  Generating plots...\n{'='*60}")

valid = {k: v for k, v in results.items() if v is not None}
n_valid = len(valid)
cmap = plt.cm.tab10
colors = {name: cmap(i) for i, name in enumerate(valid.keys())}

fig, axes = plt.subplots(3, 2, figsize=(15, 13))
fig.suptitle(f'Controller Benchmark — LGVI + BEM + L-IEKF @ {SIM_RATE} Hz\n'
             f'Circular traj (r={RADIUS}m), wind = {WIND_VEC} m/s',
             fontsize=13, fontweight='bold')

# (0,0) XY trajectory
ax = axes[0, 0]
r0 = list(valid.values())[0]
ax.plot(r0['f']['x'][:, 0], r0['f']['x'][:, 1], 'k--', alpha=0.3, label='Desired', lw=1)
for name, r in valid.items():
    ax.plot(r['s']['x'][:, 0], r['s']['x'][:, 1], color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=6, ncol=2); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Tracking error
ax = axes[0, 1]
for name, r in valid.items():
    ax.plot(r['t'], tracking_error(r), color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

# (1,0) Z tracking
ax = axes[1, 0]
for name, r in valid.items():
    ax.plot(r['t'], r['s']['x'][:, 2] - r['f']['x'][:, 2], color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('Time [s]'); ax.set_ylabel('z error [m]')
ax.set_title('Altitude Error'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

# (1,1) XY error
ax = axes[1, 1]
for name, r in valid.items():
    xy_err = np.linalg.norm(r['s']['x'][:, :2] - r['f']['x'][:, :2], axis=1)
    ax.plot(r['t'], xy_err, color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_xy|| [m]')
ax.set_title('Horizontal Tracking Error'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

# (2,0) Angular velocity
ax = axes[2, 0]
for name, r in valid.items():
    ax.plot(r['t'], np.linalg.norm(r['s']['w'], axis=1), color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||w|| [rad/s]')
ax.set_title('Angular Velocity Magnitude'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

# (2,1) Motor speeds (mean)
ax = axes[2, 1]
for name, r in valid.items():
    mean_rpm = np.mean(r['s']['rotor_speeds'], axis=1)
    ax.plot(r['t'], mean_rpm, color=colors[name], alpha=0.7, label=name)
ax.set_xlabel('Time [s]'); ax.set_ylabel('Mean rotor speed [rad/s]')
ax.set_title('Mean Rotor Speed'); ax.legend(fontsize=6, ncol=2); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ROOT, 'controller_benchmark_lgvi_bem.png')
plt.savefig(out_path, dpi=150)
print(f"Plot saved to: {out_path}")

# =====================================================================
#  Summary
# =====================================================================
print(f"\n{'='*80}")
print(f"  Controller Benchmark — LGVI + BEM + L-IEKF @ {SIM_RATE} Hz  (t > 3s)")
print(f"{'='*80}")
print(f"{'Controller':<16} {'Track mean':>11} {'Track max':>11} {'Track std':>11} {'Wall [s]':>9} {'Exit':>18}")
print("-" * 80)
for name in results:
    r = results[name]
    if r is None:
        print(f"{name:<16} {'FAILED':>11}")
        continue
    idx = r['t'] > 3.0
    e = tracking_error(r)
    if np.sum(idx) > 0:
        print(f"{name:<16} {e[idx].mean():>10.4f}m {e[idx].max():>10.4f}m {e[idx].std():>10.4f}m {r['wall']:>9.1f} {r['exit'].value[:18]:>18}")
    else:
        print(f"{name:<16} {'crashed':>11} {'':>11} {'':>11} {r['wall']:>9.1f} {r['exit'].value[:18]:>18}")

print(f"\n{'='*60}")
print(f"  L-IEKF Estimation Accuracy (t > 3s)")
print(f"{'='*60}")
print(f"{'Controller':<16} {'Pos [mm]':>10} {'Att [deg]':>10} {'Vel [m/s]':>10}")
print("-" * 50)
for name, r in valid.items():
    idx = r['t'] > 3.0
    pe, ae, ve = extract_errors(r)
    if np.sum(idx) > 0:
        print(f"{name:<16} {pe[idx].mean()*1000:>10.1f} {ae[idx].mean():>10.2f} {ve[idx].mean():>10.4f}")

print("\nDone.")
