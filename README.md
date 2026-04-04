# AdaptiveQuadBench

![System Overview](media/system_overview.png)

*Figure from our paper: [A Simulation Evaluation Suite for Robust Adaptive Quadcopter Control](https://arxiv.org/abs/2510.03471)*

AdaptiveQuadBench is a standarized testbed for adaptive quadrotor controllers, built on top of the RotorPy simulator. It provides implementations of various adaptive control strategies and tools to evaluate their performance under different disturbances and conditions.

## Features

- Multiple adaptive controller implementations (Geometric, L1, MPC, INDI, etc.)
- Standardized evaluation framework for controller comparison
- Support for various disturbances (wind, force, torque, parameter uncertainty)
- Parallel simulation capabilities for efficient benchmarking
- Visualization tools for performance analysis

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda package manager
- Git

### Installation Steps

1. Clone the repository and initialize submodules:
   ```bash
   git clone https://github.com/Dz298/AdaptiveQuadBench.git
   cd AdaptiveQuadBench
   git submodule update --init --recursive
   ```

2. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate quadbench
   ```

3. If you plan to use MPC controllers, install acados following these steps:

   **Step 1: Clone acados repository**
   ```bash
   git clone https://github.com/acados/acados.git
   cd acados
   git submodule update --recursive --init
   ```

   **Step 2: Build and install acados**
   ```bash
   mkdir -p build
   cd build
   cmake -DACADOS_WITH_QPOASES=ON ..
   make install -j4
   ```

   **Step 3: Install Python interface**
   ```bash
   pip install -e <acados_root>/interfaces/acados_template
   ```
   Note: Replace `<acados_root>` with the actual path to your acados directory.

   **Step 4: Set environment variables**
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
   export ACADOS_SOURCE_DIR="<acados_root>"
   ```
   Note: On MacOS, use `DYLD_LIBRARY_PATH` instead of `LD_LIBRARY_PATH`.
   
   **Step 5: Verify installation**
   ```bash
   cd <acados_root>/examples/acados_python/getting_started/
   python minimal_example_ocp.py
   ```
   If this runs without errors, your acados installation is working correctly.

   **Step 6: Make environment variables permanent (optional)**
   Add the export commands to your `~/.bashrc` or `~/.zshrc` file to avoid setting them every time you open a new terminal.

## Usage

### Running Experiments

To run a basic experiment comparing different controllers:

```bash
python run_eval.py --controller geo geo-a l1geo --experiment wind --num_trials 100
```

Available options:
- `--controller`: Controller types to evaluate (geo, geo-a, l1geo, l1mpc, indi-a, xadap, mpc, all)
- `--experiment`: Experiment type (wind, uncertainty, force, torque, rotoreff)
- `--num_trials`: Number of trials to run
- `--trajectory`: Trajectory type (random, hover, circle)
- `--save_trials`: Save individual trial data
- `--serial`: Run in serial mode (default is parallel)
- `--vis`: Visualize a single trial without saving data

### Finding Controller Limits

To find the failure point of controllers by gradually increasing disturbance intensity:

```bash
python run_eval.py --controller geo geo-a --experiment wind --when2fail --max_intensity 10.0
```

### Finding Delay Margin

To evaluate the robustness of controllers to time delays, you can run the delay margin analysis:

```bash
python run_eval.py --controller geo geo-a l1geo --delay_margin
```

### Visualizing Results

Results are saved in the `data` directory and can be visualized using the included visualization tools.


## Citation

If you use AdaptiveQuadBench in your research, please cite:

```bibtex
@misc{zhang2025simulationevaluationsuiterobust,
      title={A Simulation Evaluation Suite for Robust Adaptive Quadcopter Control}, 
      author={Dingqi Zhang and Ran Tao and Sheng Cheng and Naira Hovakimyan and Mark W. Mueller},
      year={2025},
      eprint={2510.03471},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.03471}, 
}
```

Additionally, please cite the specific controller implementations you use:

* Baseline Controllers
  * Geometric Control described in [Lee CDC'10](https://doi.org/10.1109/CDC.2010.5717652)
  * Model Predictive Control described in [Sun TRO'22](https://arxiv.org/abs/2109.01365)

* Adaptive Controllers
  * Geometric Adaptive Controller described in [Goodarzi JDSMC'15](https://doi.org/10.1115/1.4030419)
  * L1 Adaptive Geometric Controller described in [Wu arXiv'23](https://arxiv.org/abs/2302.07208)
  * L1 Adaptive MPC described in [Tao ACC'24](https://doi.org/10.23919/ACC60939.2024.10644611)
  * Adaptive INDI described in [Smeur JGCD'15](https://doi.org/10.2514/1.G001490) and [Tal TCST'21](https://doi.org/10.1109/TCST.2020.3001117)
  * Learning-based Extreme Adaptation Controller described in [Zhang TRO'25](https://doi.org/10.1109/TRO.2025.3577037)


## Extended Features (AIAA 2026 Fork)

This fork adds physics-fidelity modules, wind field models, and a unified
configuration system for benchmarking controllers under realistic conditions.

### New Modules

| Module | File | Description |
|--------|------|-------------|
| **LGVI Integrator** | `rotorpy/vehicles/multirotor.py` | Lie-group variational integrator (Lee-Leok-McClamroch) on SO(3) via Cayley map + implicit Newton solve.  Replaces RK45 for exact angular-momentum and SO(3) preservation. Translation uses Stormer-Verlet. |
| **BEM Rotor Aero** | `rotorpy/vehicles/bem_rotor.py` | Blade-element momentum model (Davoudi et al.). Wind-sensitive thrust — increases tracking error 10-30%. |
| **EKF INS** | `rotorpy/estimators/ekf_ins.py` | 15-state extended Kalman filter (Euler-angle parameterization). IMU propagation + MoCap correction. |
| **UKF INS** | `rotorpy/estimators/ukf_ins.py` | Unscented Kalman filter, same 15-state Euler-angle formulation. Uses Merwe scaled sigma points via `filterpy`. |
| **L-IEKF INS** | `rotorpy/estimators/liekf_ins.py` | Left-Invariant EKF on SE_2(3) x R^6. World-frame error convention; constant gravity block in the A matrix. |
| **R-IEKF (EqF) INS** | `rotorpy/estimators/eqf_ins.py` | Right-Invariant EKF / Equivariant Filter. Body-frame error convention. |
| **Turbulent Wind** | `rotorpy/wind/default_winds.py` | First-order Gauss-Markov (Ornstein-Uhlenbeck) colored-noise turbulence model. Configurable mean, std, correlation time. |
| **CFD Wind** | `rotorpy/wind/cfd_wind.py` | van der Laan LES NetCDF wind field (NREL-5MW wake). Spatially and temporally varying 2D hub-height flow. |
| **Estimator-in-the-loop** | `rotorpy/simulate.py` | `use_estimator=True` flag feeds estimated (not true) state to the controller, closing the estimation-control loop. |

### Unified Configuration (`sim_config.py`)

All simulation scripts share a single configuration system. Edit parameters at
the top of each file — no command-line arguments needed:

```python
from sim_config import get_config, make_x0, make_wind, make_sensors, get_quad_params, NullEstimator

cfg = get_config(
    integrator  = 'lgvi',       # 'rk45' | 'lgvi'
    aero        = 'std',        # 'std' | 'bem'
    sim_rate    = 500,          # Hz
    t_final     = 15.0,         # s
    controller  = 'SE3',        # 'SE3' | 'Geo' | 'GeoAdaptive' | 'L1-Geo' | 'INDI'
    estimator   = 'gt',         # 'gt' | 'liekf'
    trajectory  = 'circular',   # 'circular' | 'hover' | 'lissajous'
    radius      = 2.0,          # m
    circle_freq = 0.2,          # Hz (0.2 = 5s per lap)
    wind_mode   = 'constant',   # 'nowind' | 'constant' | 'turbulent' | 'cfd'
    wind_vec    = [3.0, 0.0, 0.0],
)
```

Global defaults live in `sim_config.py`; each script overrides only what it needs.

### Wind Models

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| `nowind` | No disturbance | — |
| `constant` | Uniform steady wind | `wind_vec = [wx, wy, wz]` m/s |
| `turbulent` | Mean + Gauss-Markov colored noise | `turb_mean`, `turb_std`, `turb_tau` (correlation time) |
| `cfd` | van der Laan LES wake field (NREL-5MW) | `cfd_file`, `cfd_wake` (downstream distance), `cfd_unit` ('R'/'D'/'m') |

CFD wind example — UAV flying in near-wake:
```python
cfg = get_config(
    wind_mode   = 'cfd',
    cfd_file    = 'wind field/2019/case6_highTi_UWV_10min_bin1.nc',
    cfd_wake    = 5.0,      # 5R = 315m downstream of turbine
    cfd_unit    = 'R',      # R=63m (rotor radius), D=126m (diameter), m=metres
)
```

### Single-Case Simulation (`run_single.py`)

Quick simulation with configurable physics, controller, trajectory, and wind.
Edit the `cfg = get_config(...)` block at the top, then run:

```bash
python run_single.py
```

Outputs:
- `run_single_result.png` — 3D trajectory, tracking error, position components, rotor speeds
- `run_single_result_cfd_wind.png` — (CFD only) global wake view + zoomed UAV region with wind field contour
- `run_single_result_cfd_wind_ts.png` — (CFD only) wind time series experienced by UAV

### Full Benchmark (`benchmark_full.py`)

Ten configurations spanning three fidelity axes — integrator, aerodynamic model,
and state-feedback source — at two sample rates:

| # | Integrator | Aero Model | State Feedback | Rate |
|---|-----------|-----------|---------------|------|
| C1 | RK45 | Standard | GT (noisy) | 100 Hz |
| C2 | RK45 | BEM | GT (noisy) | 100 Hz |
| C3 | RK45 | BEM | GT (noisy) | 500 Hz |
| C4 | RK45 | Standard | GT (noisy) | 500 Hz |
| C5 | LGVI | Standard | GT (noisy) | 100 Hz |
| C6 | LGVI | BEM | GT (noisy) | 100 Hz |
| C7 | LGVI | Standard | GT (noisy) | 500 Hz |
| C8 | LGVI | BEM | GT (noisy) | 500 Hz |
| C9 | LGVI | BEM | L-IEKF | 100 Hz |
| C10 | LGVI | BEM | L-IEKF | 500 Hz |

All cases: circular trajectory (r = 2 m), constant wind [3, 0, 0] m/s, T = 15 s.

### Results — Mean Position Tracking Error [m] (t > 3 s)

| Case | SE3 | Geo | GeoAdaptive | L1-Geo | INDI |
|------|-----|-----|-------------|--------|------|
| C1:  RK45+Std+GT@100 | 0.147 | 0.255 | 0.281 | 0.240 | **0.137** |
| C2:  RK45+BEM+GT@100 | **0.164** | 0.268 | 0.375 | 0.313 | 0.179 |
| C3:  RK45+BEM+GT@500 | **0.163** | 0.278 | 0.371 | X (2.1s) | 0.181 |
| C4:  RK45+Std+GT@500 | 0.149 | 0.262 | 0.283 | unstable | **0.134** |
| C5:  LGVI+Std+GT@100 | **0.145** | 0.234 | 1.763 | 0.191 | 0.175 |
| C6:  LGVI+BEM+GT@100 | **0.167** | X (2.3s) | X (4.4s) | 0.308 | 3.626 |
| C7:  LGVI+Std+GT@500 | 0.151 | 0.237 | 0.282 | unstable | **0.138** |
| C8:  LGVI+BEM+GT@500 | **0.164** | 0.248 | 0.376 | X (2.1s) | 0.184 |
| C9:  LGVI+BEM+LIEKF@100 | 6.689 | X (3.0s) | X (3.4s) | X (4.4s) | X (3.1s) |
| C10: LGVI+BEM+LIEKF@500 | **0.163** | 0.282 | 0.380 | X (2.1s) | 0.527 |

**X** = crashed / diverged before completion.  **Bold** = best in row.

### Key Findings

1. **SE3Control is the most robust controller overall** — it survives every
   single configuration including the hardest case (C9), though with degraded
   accuracy (6.7 m) at 100 Hz + L-IEKF.
2. **INDI is the best performer under ideal conditions** (C1: 0.137 m) but
   degrades sharply once BEM aerodynamics and estimation noise are introduced
   (C6: 3.6 m, C9: crashed).
3. **L1-Geo is fragile at 500 Hz** — its L1 adaptive filter bandwidth is tuned
   for 100 Hz; at 500 Hz the filter becomes too aggressive and causes
   instability across all aero models.  At 100 Hz + BEM it performs well (C6:
   0.308 m).
4. **Estimation-in-the-loop at 100 Hz is catastrophic** (C9) — only SE3
   survives, and poorly.  At 500 Hz (C10) most controllers recover to near-GT
   performance.
5. **BEM aerodynamics increases tracking error by 10-30%** compared to the
   standard model (C1 vs C2, C5 vs C6), and destabilizes Geo/GeoAdaptive
   at low rates.
6. **LGVI vs RK45** makes little difference in tracking accuracy when both
   converge, but LGVI is 2-3x faster in wall-clock time due to its
   fixed-step nature (no adaptive step-size overhead).

### Comparison Scripts

| Script | What it compares |
|--------|-----------------|
| `run_single.py` | Single configurable simulation with plots |
| `benchmark_full.py` | 10-case x 5-controller full benchmark |
| `test_lgvi.py` | RK45 vs LGVI integrator (SO(3) preservation, energy, speed) |
| `test_ins_comparison.py` | EKF vs UKF vs L-IEKF vs R-IEKF closed-loop |
| `test_lgvi_estimators.py` | LGVI + all 4 estimators at 500 Hz |
| `test_eqf_closedloop.py` | EqF INS vs GT vs MoCap feedback |
| `test_eqf_comparison.py` | EKF vs EqF wind estimation |
| `test_cfd_wind.py` | SE3 across NoWind / Constant / Turbulent / CFD wake positions |
| `test_bem_comparison.py` | Lumped vs BEM rotor model |

Running the full benchmark:
```bash
python benchmark_full.py
```

Outputs land in `results/benchmark/`:
- `benchmark_summary.json` — machine-readable results
- `benchmark_full_heatmap.png` — color-coded tracking-error matrix
- `benchmark_full_bars.png` — per-controller bar charts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [RotorPy](https://github.com/spencerfolk/rotorpy) for the quadrotor simulation framework
- Contributors to the various control algorithms implemented in this repository