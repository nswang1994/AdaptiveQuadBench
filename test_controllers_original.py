"""
Controller benchmark under ORIGINAL RotorPy settings:
  RK45 integrator, standard quad_params (no BEM), GT state feedback, 100 Hz
"""
import sys, os, copy, time as timer, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')

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

# Controllers
from rotorpy.controllers.quadrotor_control import SE3Control
from geometric_control import GeoControl
from geometric_adaptive_controller import GeometricAdaptiveController
from geometric_control_l1 import L1_GeoControl
from indi_adaptive_controller import INDIAdaptiveController

# Standard quad_params (NO BEM)
from quad_param.quadrotor import quad_params

SIM_RATE = 100
T_FINAL  = 15
RADIUS   = 2.0
WIND_VEC = np.array([3.0, 0.0, 0.0])
dt = 1.0 / SIM_RATE

world = World.empty((-10, 10, -10, 10, -10, 10))
wind_profile = ConstantWind(*WIND_VEC)

hover_omega = np.sqrt(quad_params['mass'] * 9.81 / (4 * quad_params['k_eta']))
x0 = {'x': np.array([RADIUS, 0, 0]),
      'v': np.zeros(3),
      'q': np.array([0, 0, 0, 1]),
      'w': np.zeros(3),
      'wind': np.array([0, 0, 0]),
      'rotor_speeds': np.array([hover_omega]*4)}

class NullEstimator:
    def step(self, *a, **kw):
        return {'filter_state': np.zeros(1), 'covariance': np.zeros(1)}
    def get_state_estimate(self):
        return {}

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

controllers = [
    ('SE3',         lambda p: SE3Control(p)),
    ('Geo',         lambda p: GeoControl(p)),
    ('GeoAdaptive', lambda p: GeometricAdaptiveController(p, dt=dt)),
    ('L1-Geo',      lambda p: L1_GeoControl(p)),
    ('INDI',        lambda p: INDIAdaptiveController(p, dt=dt)),
]

results = {}
for i, (name, ctrl_fn) in enumerate(controllers):
    print(f"\n{'='*60}")
    print(f"  [{i+1}/{len(controllers)}] {name}  (RK45 + stdParams + GT @ {SIM_RATE} Hz)")
    print(f"{'='*60}")
    try:
        params = copy.deepcopy(quad_params)
        vehicle = Multirotor(params, control_abstraction='cmd_motor_speeds', integrator='rk45')
        vehicle.initial_state = copy.deepcopy(x0)
        imu_s, mc_s = make_sensors()
        ctrl = ctrl_fn(quad_params)

        t0 = timer.perf_counter()
        (t, s, c, f, _, _, _, _, ex) = simulate(
            world, copy.deepcopy(x0), vehicle, ctrl,
            CircularTraj(radius=RADIUS), wind_profile,
            imu_s, mc_s, NullEstimator(),
            T_FINAL, dt, safety_margin=0.25,
            use_mocap=False, use_estimator=False)
        wall = timer.perf_counter() - t0

        results[name] = {'t': t, 's': s, 'f': f, 'exit': ex, 'wall': wall}
        print(f"  Exit: {ex.value}, t={t[-1]:.2f}s, wall={wall:.1f}s")
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        results[name] = None

def tracking_error(r):
    return np.linalg.norm(r['s']['x'] - r['f']['x'], axis=1)

print(f"\n{'='*80}")
print(f"  Original RotorPy Settings: RK45 + stdParams + GT @ {SIM_RATE} Hz  (t > 3s)")
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

print("\nDone.")
