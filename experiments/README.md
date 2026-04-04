# Experiments

Experiment scripts for the AIAA 2026 paper on SOC-adaptive ESO wind estimation.
All scripts live in the **repo root** (one level up); this folder is the documentation index.

> `test/` (sibling folder) contains the original RotorPy unit tests — unrelated.

---

## Quick reference

| Script | What it tests | Key outputs |
|--------|---------------|-------------|
| [`benchmark_full.py`](#benchmark_fullpy) | 10 cases × 5 controllers | heatmap, bar chart, JSON |
| [`test_controllers_original.py`](#test_controllers_originalpy) | Original RotorPy settings (ideal baseline) | console table |
| [`test_controllers_lgvi_bem.py`](#test_controllers_lgvi_bempy) | LGVI + BEM + L-IEKF @ 100 Hz | console table |
| [`test_lgvi.py`](#test_lgvipy) | RK45 vs LGVI integrator | trajectory plots |
| [`test_lgvi_estimators.py`](#test_lgvi_estimatorspy) | LGVI + 4 estimators | error plots |
| [`test_ins_comparison.py`](#test_ins_comparisonpy) | EKF / UKF / L-IEKF / R-IEKF comparison | error plots |
| [`test_bem_comparison.py`](#test_bem_comparisonpy) | Standard aero vs BEM aero | error plots |
| [`test_eqf_comparison.py`](#test_eqf_comparisonpy) | EqF vs other estimators (open-loop) | error plots |
| [`test_eqf_closedloop.py`](#test_eqf_closedlooppy) | EqF closed-loop with SE3 controller | trajectory plots |
| [`test_cfd_wind.py`](#test_cfd_windpy) | Four wind modes incl. LES wake | position plots, JSON |

---

## benchmark_full.py

**Purpose:** Full factorial benchmark — 10 simulation configurations × 5 controllers = 50 runs.

**Cases:**

| ID | Integrator | Aero | Estimator | Rate |
|----|-----------|------|-----------|------|
| C1 | RK45 | Std | GT | 100 Hz |
| C2 | RK45 | BEM | GT | 100 Hz |
| C3 | RK45 | BEM | GT | 500 Hz |
| C4 | RK45 | Std | GT | 500 Hz |
| C5 | LGVI | Std | GT | 100 Hz |
| C6 | LGVI | BEM | GT | 100 Hz |
| C7 | LGVI | Std | GT | 500 Hz |
| C8 | LGVI | BEM | GT | 500 Hz |
| C9 | LGVI | BEM | L-IEKF | 100 Hz |
| C10 | LGVI | BEM | L-IEKF | 500 Hz |

**Controllers:** SE3, Geo, GeoAdaptive, L1-Geo, INDI

**Wind mode** (set `WIND_MODE` at top of file):

| Value | Description |
|-------|-------------|
| `'nowind'` | No disturbance |
| `'constant'` | Uniform 3 m/s (default) |
| `'turbulent'` | Mean 3 m/s + Gauss-Markov colored noise |
| `'cfd'` | van der Laan LES wake (requires NetCDF file) |

**Outputs:** `benchmark_full_heatmap.png`, `benchmark_full_bars.png`, `benchmark_summary.json`

---

## test_controllers_original.py

**Purpose:** Reproduce the original AdaptiveQuadBench results — RK45 + standard params + ground-truth feedback @ 100 Hz. Establishes the ideal upper bound for all controllers.

**Finding:** All 5 controllers survive; INDI achieves the best tracking (0.132 m). This is the baseline that gets degraded in subsequent experiments.

---

## test_controllers_lgvi_bem.py

**Purpose:** Stress-test — LGVI integrator + BEM aerodynamics + L-IEKF estimator @ 100 Hz. Reveals which controllers are robust to realistic physics + noisy state feedback.

**Finding:** Only L1-Geo survives (0.302 m). SE3, Geo, GeoAdaptive, INDI all crash. Confirms 100 Hz is insufficient for LGVI + estimator except for L1-Geo (whose internal filter is tuned for 100 Hz).

---

## test_lgvi.py

**Purpose:** Isolate the integrator effect. Side-by-side comparison of RK45 vs LGVI with identical controller (SE3) and params. Verifies LGVI is a drop-in replacement and quantifies any accuracy difference.

---

## test_lgvi_estimators.py

**Purpose:** LGVI dynamics + four estimators (EKF, UKF, L-IEKF, R-IEKF) with SE3 controller. Measures how estimator choice affects closed-loop tracking under realistic (BEM) aerodynamics.

---

## test_ins_comparison.py

**Purpose:** Open-loop estimator accuracy comparison. All four estimators (EKF-INS, UKF-INS, L-IEKF-INS, EqF) run against the same simulated IMU + mocap sensor stream. Reports position, velocity, and attitude RMSE.

---

## test_bem_comparison.py

**Purpose:** Isolate the aerodynamic model effect. Standard lumped-coefficient model vs BEM rotor model, with identical integrator (RK45) and controller (SE3). Quantifies how much BEM increases tracking error.

---

## test_eqf_comparison.py

**Purpose:** Benchmark the Equivariant Filter (EqF) against EKF/UKF/L-IEKF in open-loop estimation (no controller feedback). Evaluates the geometric estimator on SE_2(3).

---

## test_eqf_closedloop.py

**Purpose:** Closed-loop validation of EqF — SE3 controller driven by EqF state estimates, LGVI dynamics, BEM aero. Checks whether EqF is stable and competitive in closed-loop vs L-IEKF.

---

## test_cfd_wind.py

**Purpose:** SE3 controller under four wind disturbance conditions, including LES wake data from van der Laan & Andersen (2018). Intended as a stepping stone toward ESO validation in realistic wind fields.

**Wind mode** (set `WIND_MODE` at top of file):

| Value | Cases run |
|-------|-----------|
| `'nowind'` | No disturbance only |
| `'constant'` | Uniform 3 m/s only |
| `'turbulent'` | Gauss-Markov colored noise only |
| `'cfd'` | All of the above + LES at each `WAKE_POSITIONS` |

**CFD settings:**

```python
WAKE_POS_UNIT   = 'R'              # 'R' rotor radius (63 m) | 'D' diameter | 'm' metres
WAKE_POSITIONS  = [6.0, 10.0, 16.0]  # = 3D, 5D, 8D downstream
WAKE_LAT_OFFSET = 0.0              # lateral offset from wake centreline
```

UAV world-frame origin is placed at the specified downstream position behind the turbine (LES x = 0 is hub location, +x is downstream).

**Outputs:** `test_cfd_wind_positions.png`, `test_cfd_wind_stats.json`

---

## LES dataset

van der Laan & Andersen (2018), single NREL-5MW wake, hub-height horizontal plane.

| Property | Value |
|----------|-------|
| Rotor diameter D | 126 m |
| Rotor radius R | 63 m |
| Hub-height inflow U₀ | 8.0 m/s |
| Spatial resolution | dx = dy = 4.2 m |
| Time step | dt = 0.24 s |
| Domain | x ∈ [−126, 1071] m, y ∈ [−252, 252] m |
| Duration | 600 s (10 min), 2500 snapshots |
| Cases | `case5_lowTi`, `case6_highTi` × 6 random seeds |

Files expected at:
```
C:\Users\wnsx2\University of Michigan Dropbox\Ningshan Wang\wind field\2019\
    case5_lowTi_UWV_10min_bin{1-6}.nc
    case6_highTi_UWV_10min_bin{1-6}.nc
```
