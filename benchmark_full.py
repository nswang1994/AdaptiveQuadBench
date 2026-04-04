"""
Full controller benchmark across 10 configurations.

Cases:
  C1:  RK45 + Std  + GT   @ 100 Hz   (original RotorPy baseline)
  C2:  RK45 + BEM  + GT   @ 100 Hz
  C3:  RK45 + BEM  + GT   @ 500 Hz
  C4:  RK45 + Std  + GT   @ 500 Hz
  C5:  LGVI + Std  + GT   @ 100 Hz
  C6:  LGVI + BEM  + GT   @ 100 Hz
  C7:  LGVI + Std  + GT   @ 500 Hz
  C8:  LGVI + BEM  + GT   @ 500 Hz
  C9:  LGVI + BEM  + LIEKF @ 100 Hz
  C10: LGVI + BEM  + LIEKF @ 500 Hz

Controllers: SE3, Geo, GeoAdaptive, L1-Geo, INDI
Trajectory : circular r=2m, T = 15 s
"""

import sys, os, copy, time as timer, traceback, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
sys.path.insert(0, os.path.join(ROOT, 'controller'))

from sim_config import get_config, make_wind, make_sensors, NullEstimator

from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.simulate import simulate
from rotorpy.world import World
from rotorpy.estimators.liekf_ins import LIEKFINS

from rotorpy.controllers.quadrotor_control import SE3Control
from geometric_control import GeoControl
from geometric_adaptive_controller import GeometricAdaptiveController
from geometric_control_l1 import L1_GeoControl
from indi_adaptive_controller import INDIAdaptiveController

from quad_param.quadrotor import quad_params as std_params
from quad_param.quadrotor_with_bem import quad_params as bem_params

# =====================================================================
#  Configuration  ← edit here
# =====================================================================
cfg = get_config(
    t_final         = 15,
    radius          = 2.0,

    # Wind
    wind_mode       = 'constant',   # 'nowind' | 'constant' | 'turbulent' | 'cfd'
    wind_vec        = [3.0, 0.0, 0.0],
    turb_mean       = [3.0, 0.0, 0.0],
    turb_std        = [1.5, 0.6, 0.4],
    turb_tau        = 3.5,
    turb_seed       = 0,
    cfd_file        = '/wind field/2019/case6_highTi_UWV_10min_bin1.nc',
    cfd_wake        = 10.0,
    cfd_unit        = 'R',
    cfd_lateral     = 0.0,
    cfd_t_offset    = 0.0,

    results_dir     = os.path.join(ROOT, 'results', 'benchmark'),
)

RESULTS_DIR = cfg.results_dir
os.makedirs(RESULTS_DIR, exist_ok=True)

wind_profile = make_wind(cfg)
print(f"[benchmark] Wind mode: {cfg.wind_mode}")

world = World.empty((-10, 10, -10, 10, -10, 10))

# =====================================================================
#  Case definitions
# =====================================================================
CASES = [
    # (label,          integrator, params_key, estimator, rate)
    ('C1:  RK45+Std+GT@100',    'rk45', 'std', 'gt',    100),
    ('C2:  RK45+BEM+GT@100',    'rk45', 'bem', 'gt',    100),
    ('C3:  RK45+BEM+GT@500',    'rk45', 'bem', 'gt',    500),
    ('C4:  RK45+Std+GT@500',    'rk45', 'std', 'gt',    500),
    ('C5:  LGVI+Std+GT@100',    'lgvi', 'std', 'gt',    100),
    ('C6:  LGVI+BEM+GT@100',    'lgvi', 'bem', 'gt',    100),
    ('C7:  LGVI+Std+GT@500',    'lgvi', 'std', 'gt',    500),
    ('C8:  LGVI+BEM+GT@500',    'lgvi', 'bem', 'gt',    500),
    ('C9:  LGVI+BEM+LIEKF@100', 'lgvi', 'bem', 'liekf', 100),
    ('C10: LGVI+BEM+LIEKF@500', 'lgvi', 'bem', 'liekf', 500),
]

CTRL_NAMES = ['SE3', 'Geo', 'GeoAdaptive', 'L1-Geo', 'INDI']

def get_params(key):
    return std_params if key == 'std' else bem_params

def make_x0(params):
    hover_omega = np.sqrt(params['mass'] * 9.81 / (4 * params['k_eta']))
    return {'x': np.array([cfg.radius, 0, 0]),
            'v': np.zeros(3),
            'q': np.array([0, 0, 0, 1]),
            'w': np.zeros(3),
            'wind': np.array([0, 0, 0]),
            'rotor_speeds': np.array([hover_omega]*4)}

def make_controller(name, params, dt):
    if name == 'SE3':         return SE3Control(params)
    if name == 'Geo':         return GeoControl(params)
    if name == 'GeoAdaptive': return GeometricAdaptiveController(params, dt=dt)
    if name == 'L1-Geo':      return L1_GeoControl(params)
    if name == 'INDI':        return INDIAdaptiveController(params, dt=dt)

def tracking_error(r):
    return np.linalg.norm(r['s']['x'] - r['f']['x'], axis=1)

# =====================================================================
#  Main loop
# =====================================================================
# results[case_label][ctrl_name] = {t, s, f, exit, wall, track_mean, ...} or None
all_results = {}

total_runs = len(CASES) * len(CTRL_NAMES)
run_idx = 0

for case_label, integrator, params_key, est_type, rate in CASES:
    dt = 1.0 / rate
    params = get_params(params_key)
    x0 = make_x0(params)
    case_results = {}

    print(f"\n{'#'*70}")
    print(f"  {case_label}")
    print(f"{'#'*70}")

    for ctrl_name in CTRL_NAMES:
        run_idx += 1
        tag = f"[{run_idx}/{total_runs}]"
        print(f"\n  {tag} {ctrl_name} — {case_label}")

        try:
            p = copy.deepcopy(params)
            vehicle = Multirotor(p, control_abstraction='cmd_motor_speeds',
                                 integrator=integrator)
            vehicle.initial_state = copy.deepcopy(x0)

            case_cfg = get_config(**{**cfg, 'sim_rate': rate})
            imu_s, mc_s = make_sensors(case_cfg)

            use_est = (est_type == 'liekf')
            est = LIEKFINS(params, dt=dt) if use_est else NullEstimator()
            ctrl = make_controller(ctrl_name, params, dt)

            t0 = timer.perf_counter()
            result_tuple = simulate(
                world, copy.deepcopy(x0), vehicle, ctrl,
                CircularTraj(radius=cfg.radius), wind_profile,
                imu_s, mc_s, est,
                cfg.t_final, dt, safety_margin=0.25,
                use_mocap=False, use_estimator=use_est)
            wall = timer.perf_counter() - t0

            t_arr = result_tuple[0]
            s_arr = result_tuple[1]
            f_arr = result_tuple[3]
            ex = result_tuple[8]

            r = {'t': t_arr, 's': s_arr, 'f': f_arr, 'exit': ex, 'wall': wall}

            # Compute metrics (t > 3s)
            e = tracking_error(r)
            idx = r['t'] > 3.0
            if np.sum(idx) > 0 and t_arr[-1] > 5.0:
                r['track_mean'] = float(e[idx].mean())
                r['track_max']  = float(e[idx].max())
                r['track_std']  = float(e[idx].std())
                r['survived']   = True
            else:
                r['track_mean'] = None
                r['survived']   = False

            case_results[ctrl_name] = r
            status = f"t={t_arr[-1]:.1f}s, track={r['track_mean']:.3f}m" if r['survived'] else f"CRASHED t={t_arr[-1]:.1f}s"
            print(f"    {status}  ({wall:.1f}s wall)")

        except Exception as exc:
            print(f"    EXCEPTION: {exc}")
            traceback.print_exc()
            case_results[ctrl_name] = None

    all_results[case_label] = case_results

# =====================================================================
#  Summary table
# =====================================================================
print("\n\n")
print("=" * 110)
print("  FULL BENCHMARK SUMMARY — Mean Tracking Error [m] (t > 3s)")
print("  'X' = crashed / unstable,  '-' = exception")
print("=" * 110)

header = f"{'Case':<28}" + "".join(f"{c:>14}" for c in CTRL_NAMES)
print(header)
print("-" * 110)

# Also build a machine-readable summary
summary_data = {}

for case_label, _, _, _, _ in CASES:
    cr = all_results[case_label]
    row = f"{case_label:<28}"
    case_summary = {}
    for cn in CTRL_NAMES:
        r = cr.get(cn)
        if r is None:
            row += f"{'  -':>14}"
            case_summary[cn] = '-'
        elif not r['survived']:
            crash_t = f"X({r['t'][-1]:.1f}s)"
            row += f"{crash_t:>14}"
            case_summary[cn] = crash_t
        else:
            val = f"{r['track_mean']:.4f}"
            row += f"{val:>14}"
            case_summary[cn] = r['track_mean']
    print(row)
    summary_data[case_label] = case_summary

print("=" * 110)

# =====================================================================
#  Save JSON summary
# =====================================================================
json_path = os.path.join(RESULTS_DIR, 'benchmark_summary.json')
with open(json_path, 'w') as f:
    json.dump(summary_data, f, indent=2, default=str)
print(f"\nJSON summary saved to: {json_path}")

# =====================================================================
#  Plot: heatmap of tracking errors
# =====================================================================
case_labels = [c[0] for c in CASES]
n_cases = len(case_labels)
n_ctrls = len(CTRL_NAMES)

mat = np.full((n_cases, n_ctrls), np.nan)
crashed = np.zeros((n_cases, n_ctrls), dtype=bool)

for i, cl in enumerate(case_labels):
    cr = all_results[cl]
    for j, cn in enumerate(CTRL_NAMES):
        r = cr.get(cn)
        if r is not None and r['survived']:
            mat[i, j] = r['track_mean']
        else:
            crashed[i, j] = True

fig, ax = plt.subplots(figsize=(12, 8))

# Use a colormap: lower = better (green), higher = worse (red)
cmap = plt.cm.RdYlGn_r
masked = np.ma.masked_where(np.isnan(mat), mat)
vmin, vmax = 0.0, min(np.nanmax(mat) if np.any(~np.isnan(mat)) else 1.0, 2.0)

im = ax.imshow(masked, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

# Annotate cells
for i in range(n_cases):
    for j in range(n_ctrls):
        if crashed[i, j]:
            r = all_results[case_labels[i]].get(CTRL_NAMES[j])
            if r is not None:
                ax.text(j, i, f"X\n({r['t'][-1]:.1f}s)", ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#c0392b', alpha=0.85))
            else:
                ax.text(j, i, "-", ha='center', va='center', fontsize=10, color='gray')
        else:
            val = mat[i, j]
            color = 'white' if val > (vmax * 0.6) else 'black'
            ax.text(j, i, f"{val:.3f}", ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)

ax.set_xticks(range(n_ctrls))
ax.set_xticklabels(CTRL_NAMES, fontsize=10, fontweight='bold')
ax.set_yticks(range(n_cases))
ax.set_yticklabels(case_labels, fontsize=9)
ax.set_xlabel('Controller', fontsize=12)
ax.set_ylabel('Configuration', fontsize=12)
ax.set_title('Mean Position Tracking Error [m] (t > 3s)\n'
             'Circular traj (r=2m), wind = [3,0,0] m/s, T = 15s\n'
             'Red X = crashed before completion',
             fontsize=12, fontweight='bold')

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('Tracking Error [m]', fontsize=10)

plt.tight_layout()
out_path = os.path.join(RESULTS_DIR, 'benchmark_full_heatmap.png')
plt.savefig(out_path, dpi=150)
print(f"Heatmap saved to: {out_path}")

# =====================================================================
#  Plot: bar chart per controller showing survival across cases
# =====================================================================
fig2, axes2 = plt.subplots(1, n_ctrls, figsize=(18, 6), sharey=True)
fig2.suptitle('Controller Performance Across Configurations\n'
              'Mean Tracking Error [m] (t > 3s) — missing bars = crashed',
              fontsize=12, fontweight='bold')

colors_case = plt.cm.tab10(np.linspace(0, 1, n_cases))

for j, cn in enumerate(CTRL_NAMES):
    ax = axes2[j]
    vals = []
    cols = []
    labels = []
    for i, cl in enumerate(case_labels):
        r = all_results[cl].get(cn)
        if r is not None and r['survived']:
            vals.append(r['track_mean'])
            cols.append(colors_case[i])
            labels.append(cl.split(':')[0])  # C1, C2, ...
        else:
            vals.append(0)
            cols.append('#dddddd')
            labels.append(cl.split(':')[0])

    bars = ax.bar(range(n_cases), vals, color=cols, edgecolor='black', linewidth=0.5)
    # Mark crashed bars
    for i in range(n_cases):
        r = all_results[case_labels[i]].get(cn)
        if r is None or not r['survived']:
            ax.text(i, 0.01, 'X', ha='center', va='bottom', fontsize=10,
                    fontweight='bold', color='red')

    ax.set_xticks(range(n_cases))
    ax.set_xticklabels(labels, fontsize=7, rotation=45)
    ax.set_title(cn, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    if j == 0:
        ax.set_ylabel('Tracking Error [m]')

plt.tight_layout()
out_path2 = os.path.join(RESULTS_DIR, 'benchmark_full_bars.png')
plt.savefig(out_path2, dpi=150)
print(f"Bar chart saved to: {out_path2}")

print("\n  BENCHMARK COMPLETE.")
