# CLAUDE.md — AdaptiveQuadBench

Project context for Claude Code. Always read this file before making changes.

---

## What this repo is

A benchmark framework for adaptive quadrotor controllers, built for the AIAA
2026 paper on SOC-adaptive ESO wind estimation.  The codebase is a fork of
[AdaptiveQuadBench](https://github.com/Dz298/AdaptiveQuadBench) with `rotorpy`
vendored directly (no submodule) so all local extensions are tracked here.

---

## Directory layout

```
AdaptiveQuadBench/
├── rotorpy/                  ← vendored simulator (all local edits live here)
│   ├── vehicles/
│   │   ├── multirotor.py     ← Multirotor dynamics; integrator='rk45'|'lgvi'
│   │   └── bem_rotor.py      ← BEM rotor aerodynamics (Davoudi et al.)
│   ├── estimators/
│   │   ├── ekf_ins.py        ← 15-state EKF (Euler-angle INS)
│   │   ├── ukf_ins.py        ← UKF INS (Merwe scaled sigma points)
│   │   ├── liekf_ins.py      ← Left-Invariant EKF on SE_2(3)
│   │   └── eqf_ins.py        ← Right-Invariant EKF / Equivariant Filter
│   └── simulate.py           ← simulate(); use_estimator=True for closed-loop
├── controller/               ← all controller implementations
│   ├── geometric_control.py
│   ├── geometric_adaptive_controller.py
│   ├── geometric_control_l1.py
│   ├── indi_adaptive_controller.py
│   └── quadrotor_control_mpc.py   ← requires acados (not installed)
├── quad_param/
│   ├── quadrotor.py          ← standard (simplified) quad params
│   ├── quadrotor_bem.py      ← BEM-only params
│   └── quadrotor_with_bem.py ← full params used with BEM integrator
├── benchmark_full.py         ← main 10-case × 5-controller benchmark
├── benchmark_summary.json    ← last run results (machine-readable)
└── test_*.py                 ← individual comparison scripts
```

---

## Key design decisions

### State feedback
- **All estimators** expose the same interface: `step(imu, mocap)` + `get_state_estimate()` returning `{x, v, q, w}`.
- `simulate()` with `use_estimator=True` feeds the estimator output to the controller.  It supplements with `rotor_speeds`, `wind`, `accel`, `gyro` from ground truth for controllers that need them (INDI).
- **Use EKF-fused outputs** (attitude, pos, vel, ang vel) as feedback — do not use raw sensor readings directly.

### Physics integrator
- `integrator='rk45'` — SciPy adaptive RK45 (original RotorPy default).
- `integrator='lgvi'` — Lee-Leok-McClamroch Lie-group variational integrator on SO(3). Uses Cayley map + Newton iteration (10 steps, tol 1e-12) to solve the discrete Euler-Poincaré equation. Rotation: `R_{k+1} = R_k F_k`. Translation: Störmer-Verlet. Rotor dynamics: forward Euler (decoupled). LGVI is ~2-3× faster in wall time than RK45 for the same step size, and preserves SO(3) exactly.

### Aerodynamic model
- **Standard** (`quad_param/quadrotor.py`): lumped thrust/drag coefficients `k_eta`, `k_m`.
- **BEM** (`quad_param/quadrotor_with_bem.py`): blade-element momentum model via `bem_rotor.py`. More realistic; increases tracking error 10–30% and destabilises some controllers at low rates.

### Sample rate
- **100 Hz** is sufficient for RK45 + standard params + GT feedback.
- **500 Hz** is required for LGVI + BEM + closed-loop estimator (L-IEKF). At 100 Hz with estimator, discrete error amplification causes most controllers to crash.

---

## Controllers

| Name | Class | Notes |
|------|-------|-------|
| SE3 | `SE3Control` (rotorpy) | Most robust; survives all 10 benchmark cases |
| Geo | `GeoControl` | Lee geometric; crashes at LGVI+BEM+100 Hz |
| GeoAdaptive | `GeometricAdaptiveController` | Gains tuned for circle traj; fragile with BEM |
| L1-Geo | `L1_GeoControl` | L1 filter tuned for 100 Hz; crashes at 500 Hz |
| INDI | `INDIAdaptiveController` | Best at ideal conditions; crashes with BEM+estimator |
| MPC / L1-MPC | — | Require acados — **not installed**, skip |

All controllers live under `controller/` and must be imported with `sys.path` pointing there.

---

## Running the benchmark

```bash
# Full 10-case × 5-controller benchmark
python benchmark_full.py

# Single-case scripts
python test_controllers_lgvi_bem.py   # LGVI + BEM + L-IEKF
python test_controllers_original.py  # RK45 + std params + GT
python test_ins_comparison.py         # Estimator comparison (4 estimators)
python test_lgvi.py                   # RK45 vs LGVI integrator
```

Outputs land in the repo root: `benchmark_full_heatmap.png`, `benchmark_full_bars.png`, `benchmark_summary.json`.

---

## Benchmark results summary (last run)

Mean position tracking error [m], t > 3 s.  **X** = crashed.

| Case | SE3 | Geo | GeoAdaptive | L1-Geo | INDI |
|------|-----|-----|-------------|--------|------|
| C1: RK45+Std+GT@100 | 0.147 | 0.255 | 0.281 | 0.240 | **0.137** |
| C2: RK45+BEM+GT@100 | **0.164** | 0.268 | 0.375 | 0.313 | 0.179 |
| C3: RK45+BEM+GT@500 | **0.163** | 0.278 | 0.371 | X(2.1s) | 0.181 |
| C4: RK45+Std+GT@500 | 0.149 | 0.262 | 0.283 | unstable | **0.134** |
| C5: LGVI+Std+GT@100 | **0.145** | 0.234 | 1.763 | 0.191 | 0.175 |
| C6: LGVI+BEM+GT@100 | **0.167** | X(2.3s) | X(4.4s) | 0.308 | 3.626 |
| C7: LGVI+Std+GT@500 | 0.151 | 0.237 | 0.282 | unstable | **0.138** |
| C8: LGVI+BEM+GT@500 | **0.164** | 0.248 | 0.376 | X(2.1s) | 0.184 |
| C9: LGVI+BEM+LIEKF@100 | 6.689 | X | X | X | X |
| C10: LGVI+BEM+LIEKF@500 | **0.163** | 0.282 | 0.380 | X(2.1s) | 0.527 |

---

## Code conventions

- `quad_params` dicts are always `copy.deepcopy()`-ed before passing to `Multirotor` or controller constructors — they mutate the dict.
- Estimator `dt` must match `1/SIM_RATE` passed to `simulate()`.
- Sensors (`Imu`, `MotionCapture`) must be re-instantiated for each run; they carry internal state.
- MPC controllers are guarded with `try/except ImportError` — do not remove this guard.
- Never import from `rotorpy.vehicles.hummingbird_params` in new code; use `quad_param/quadrotor.py` or `quad_param/quadrotor_with_bem.py`.

---

## Dependencies

```
numpy, scipy, matplotlib          # standard
filterpy                          # UKF (pip install filterpy)
pyquaternion                      # GeometricAdaptiveController
acados_template                   # MPC — NOT installed, controllers skipped
```

Install extras: `pip install filterpy pyquaternion`
