"""
test_cfd_wind.py — SE3Control across four wind disturbance modes.

Wind mode (set wind_mode in cfg below)
---------------------------------------
  'nowind'    — no wind  (controller ideal baseline)
  'constant'  — ConstantWind([3, 0, 0]) m/s  (no LES file needed)
  'turbulent' — mean [3,0,0] + Gauss-Markov colored-noise turbulence
  'cfd'       — all of the above  +  van der Laan LES at each wake
                position in WAKE_POSITIONS

Setup
-----
- Controller : SE3Control
- Integrator : RK45 + standard quad_params + GT feedback @ 100 Hz
- Trajectory : circular r=2 m, T=15 s

Usage
-----
1. Edit cfg / WAKE_POSITIONS below as needed.
2. python test_cfd_wind.py

Outputs: results/cfd_wind/test_cfd_wind_positions.png
         results/cfd_wind/test_cfd_wind_stats.json
"""

import sys, os, copy, json, time as timer
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
sys.path.insert(0, os.path.join(ROOT, 'controller'))

# =====================================================================
#  Configuration — edit here
# =====================================================================
from sim_config import get_config, make_x0, make_sensors, get_quad_params, NullEstimator

cfg = get_config(
    sim_rate        = 100,
    t_final         = 15.0,
    radius          = 2.0,
    aero            = 'std',

    # Wind
    wind_mode       = 'cfd',        # 'nowind' | 'constant' | 'turbulent' | 'cfd'
    wind_vec        = [3.0, 0.0, 0.0],
    turb_mean       = [3.0, 0.0, 0.0],
    turb_std        = [1.5, 0.6, 0.4],
    turb_tau        = 3.5,
    turb_seed       = 0,
    cfd_file        = r"C:\Users\wnsx2\University of Michigan Dropbox\Ningshan Wang\wind field\2019\case6_highTi_UWV_10min_bin1.nc",
    cfd_unit        = 'R',
    cfd_lateral     = 0.0,

    # Output
    results_dir     = os.path.join(ROOT, 'results', 'cfd_wind'),
)

# CFD-specific: wake positions to test (in cfg.cfd_unit)
WAKE_POSITIONS = [6.0, 10.0, 16.0]

os.makedirs(cfg.results_dir, exist_ok=True)

# =====================================================================
#  Imports
# =====================================================================
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.simulate import simulate
from rotorpy.world import World
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.wind.default_winds import NoWind, ConstantWind, TurbulentWind

# ── Derived quantities ────────────────────────────────────────────────────────
DT     = 1.0 / cfg.sim_rate
params = get_quad_params(cfg)
x0     = make_x0(cfg)
world  = World.empty((-10, 10, -10, 10, -10, 10))

# ── Build wind case list from cfg.wind_mode ──────────────────────────────────
D = 126.0   # NREL 5MW rotor diameter [m]
R = 63.0    # NREL 5MW rotor radius   [m]
_unit_scale = {'R': R, 'D': D, 'm': 1.0}

wind_cases = []   # list of (label, wind_object)

if cfg.wind_mode == 'nowind':
    wind_cases.append(('NoWind', NoWind()))

elif cfg.wind_mode == 'constant':
    wind_cases.append(('Constant_3ms', ConstantWind(*np.array(cfg.wind_vec))))

elif cfg.wind_mode == 'turbulent':
    wind_cases.append(('Turbulent', TurbulentWind(mean=np.array(cfg.turb_mean),
                                                   std=np.array(cfg.turb_std),
                                                   corr_time=cfg.turb_tau,
                                                   seed=cfg.turb_seed)))

elif cfg.wind_mode == 'cfd':
    # All four cases together for direct comparison
    from scipy.interpolate import RegularGridInterpolator
    import xarray as xr

    print("=" * 60)
    print(f"Loading LES: {os.path.basename(cfg.cfd_file)} …")
    _ds      = xr.open_dataset(cfg.cfd_file)
    _t_arr   = _ds['t'].values.astype(np.float64)
    _x_arr   = _ds['x'].values.astype(np.float64)
    _y_arr   = _ds['y'].values.astype(np.float64)
    _U_data  = _ds['U'].values.astype(np.float32)
    _V_data  = _ds['V'].values.astype(np.float32)
    _W_data  = _ds['W'].values.astype(np.float32)
    _ds.close()
    mem_mb = (_U_data.nbytes + _V_data.nbytes + _W_data.nbytes) / 1e6
    print(f"  shape={_U_data.shape}  dt={_t_arr[1]-_t_arr[0]:.3f}s  "
          f"dx={_x_arr[1]-_x_arr[0]:.1f}m  ({mem_mb:.0f} MB)")

    _pts = (_t_arr, _x_arr, _y_arr)
    _kw  = dict(method='linear', bounds_error=False, fill_value=None)
    _iU  = RegularGridInterpolator(_pts, _U_data, **_kw)
    _iV  = RegularGridInterpolator(_pts, _V_data, **_kw)
    _iW  = RegularGridInterpolator(_pts, _W_data, **_kw)

    _scale = _unit_scale.get(cfg.cfd_unit)
    if _scale is None:
        raise ValueError(f"cfd_unit must be 'R', 'D', or 'm', got {cfg.cfd_unit!r}")

    def _make_cfd_wind_from_cache(pos_val):
        x0_m = pos_val * _scale
        y0_m = cfg.cfd_lateral * _scale
        _orig = np.array([x0_m, y0_m])
        _tmin, _tmax = _t_arr[0], _t_arr[-1]

        class _Obj:
            origin = _orig
            t_min, t_max = _tmin, _tmax
            iU, iV, iW = _iU, _iV, _iW

            def update(self, t, position):
                xl = float(position[0]) + self.origin[0]
                yl = float(position[1]) + self.origin[1]
                tl = np.clip(float(t), self.t_min, self.t_max)
                pt = np.array([[tl, xl, yl]])
                wx = float(self.iU(pt)[0])
                wy = float(self.iV(pt)[0])
                wz = float(self.iW(pt)[0])
                if not np.isfinite(wx): wx = 8.0
                if not np.isfinite(wy): wy = 0.0
                if not np.isfinite(wz): wz = 0.0
                return np.array([wx, wy, wz])

        return _Obj()

    wind_cases = [
        ('NoWind',       NoWind()),
        ('Constant_3ms', ConstantWind(*np.array(cfg.wind_vec))),
        ('Turbulent',    TurbulentWind(mean=np.array(cfg.turb_mean),
                                       std=np.array(cfg.turb_std),
                                       corr_time=cfg.turb_tau,
                                       seed=cfg.turb_seed)),
    ] + [(f"CFD_{p:.0f}{cfg.cfd_unit}", _make_cfd_wind_from_cache(p))
         for p in WAKE_POSITIONS]

    print(f"  LES wake positions: "
          f"{[f'{p}{cfg.cfd_unit} ({p*_scale:.0f}m)' for p in WAKE_POSITIONS]}")
    print()

else:
    raise ValueError(f"Unknown wind_mode={cfg.wind_mode!r}. "
                     "Choose 'nowind', 'constant', 'turbulent', or 'cfd'.")

# ── Run simulations ───────────────────────────────────────────────────────────
results = {}

print(f"{'Case':<22}  {'Result'}")
print("-" * 55)

for label, wind_obj in wind_cases:
    try:
        p = copy.deepcopy(params)
        vehicle = Multirotor(p, control_abstraction='cmd_motor_speeds', integrator='rk45')
        vehicle.initial_state = copy.deepcopy(x0)
        imu_s, mc_s = make_sensors(cfg)
        ctrl  = SE3Control(params)
        est   = NullEstimator()

        t0 = timer.perf_counter()
        res = simulate(world, copy.deepcopy(x0), vehicle, ctrl,
                       CircularTraj(radius=cfg.radius), wind_obj,
                       imu_s, mc_s, est,
                       cfg.t_final, DT, safety_margin=0.25,
                       use_mocap=False, use_estimator=False)
        wall = timer.perf_counter() - t0

        t_arr, s_arr, _, f_arr = res[0], res[1], res[2], res[3]
        err = np.linalg.norm(s_arr['x'] - f_arr['x'], axis=1)
        idx = t_arr > 3.0
        survived = (t_arr[-1] > 5.0) and np.sum(idx) > 0
        if survived:
            track_mean = float(err[idx].mean())
            track_max  = float(err[idx].max())
            track_std  = float(err[idx].std())
            status = f"t={t_arr[-1]:.1f}s  err={track_mean:.3f}±{track_std:.3f}m  max={track_max:.3f}m"
        else:
            track_mean = track_max = track_std = None
            status = f"CRASHED  t={t_arr[-1]:.1f}s"

        results[label] = {'t': t_arr, 's': s_arr, 'f': f_arr,
                          'err': err, 'survived': survived,
                          'track_mean': track_mean, 'track_max': track_max,
                          'track_std': track_std, 'wall': wall}
        print(f"  {label:<22}  {status}  ({wall:.1f}s wall)")

    except Exception as exc:
        import traceback
        print(f"  {label:<22}  EXCEPTION: {exc}")
        traceback.print_exc()
        results[label] = None

# ── Plot ──────────────────────────────────────────────────────────────────────
print("\nGenerating plots …")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

colors = plt.cm.Set1(np.linspace(0, 0.9, len(wind_cases)))

for (label, _), color in zip(wind_cases, colors):
    r = results.get(label)
    if r is None:
        continue
    t_arr = r['t']
    err   = r['err']
    linestyle = '--' if 'Constant' in label else '-'
    axes[0].plot(t_arr, err, label=label, color=color, lw=1.5, ls=linestyle)

axes[0].axvline(3.0, color='gray', ls=':', lw=1, label='t=3s (metric start)')
axes[0].set_ylabel('Position error [m]')
axes[0].set_title('SE3Control — CFDWind (van der Laan high-TI) vs ConstantWind\n'
                  f'RK45 + std params + GT @ {cfg.sim_rate} Hz, circle r={cfg.radius}m, T={cfg.t_final}s')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(bottom=0)

# Bar chart of mean errors
labels_ok = [l for l, _ in wind_cases if results.get(l) and results[l]['survived']]
means_ok  = [results[l]['track_mean'] for l in labels_ok]
stds_ok   = [results[l]['track_std']  for l in labels_ok]
clrs_ok   = [colors[i] for i, (l, _) in enumerate(wind_cases)
             if results.get(l) and results[l]['survived']]

bars = axes[1].bar(range(len(labels_ok)), means_ok,
                   yerr=stds_ok, color=clrs_ok,
                   edgecolor='black', linewidth=0.8,
                   capsize=5, error_kw={'elinewidth': 1.5})
axes[1].set_xticks(range(len(labels_ok)))
axes[1].set_xticklabels(labels_ok, rotation=20, ha='right', fontsize=9)
axes[1].set_ylabel('Mean tracking error [m] (t>3s)')
axes[1].set_title('Mean tracking error ± 1σ')
axes[1].grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, means_ok):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
out = os.path.join(cfg.results_dir, 'test_cfd_wind_positions.png')
plt.savefig(out, dpi=150)
print(f"Plot saved: {out}")

# ── JSON summary ──────────────────────────────────────────────────────────────
summary = {}
for label, _ in wind_cases:
    r = results.get(label)
    if r is None:
        summary[label] = 'exception'
    elif not r['survived']:
        summary[label] = {'crashed': True, 't_end': float(r['t'][-1])}
    else:
        summary[label] = {
            'track_mean': r['track_mean'],
            'track_max':  r['track_max'],
            'track_std':  r['track_std'],
            'wall_s':     r['wall'],
        }

json_out = os.path.join(cfg.results_dir, 'test_cfd_wind_stats.json')
with open(json_out, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Stats saved: {json_out}")

# ── Print summary table ───────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SE3Control — tracking error summary")
print("="*55)
print(f"  {'Case':<22}  {'Mean [m]':>10}  {'Max [m]':>10}  {'Std [m]':>10}")
print("-"*55)
for label, _ in wind_cases:
    r = results.get(label)
    if r is None:
        print(f"  {label:<22}  {'exception':>32}")
    elif not r['survived']:
        print(f"  {label:<22}  {'CRASHED':>32}")
    else:
        print(f"  {label:<22}  {r['track_mean']:>10.3f}  {r['track_max']:>10.3f}  {r['track_std']:>10.3f}")
print("="*55)
