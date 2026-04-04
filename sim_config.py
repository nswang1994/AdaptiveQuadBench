"""
sim_config.py — Shared simulation configuration for all scripts.

Every simulation script imports from here, then overrides what it needs:

    from sim_config import CFG, make_x0, make_wind, make_sensors, NullEstimator

Edit the CONFIG dict below to change defaults globally.
"""

import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))

# ╔════════════════════════════════════════════════════════════════════╗
# ║  MASTER CONFIGURATION — edit this dict to change defaults        ║
# ╚════════════════════════════════════════════════════════════════════╝
CONFIG = dict(

    # ── Simulation ────────────────────────────────────────────────
    sim_rate        = 100,          # Hz
    t_final         = 15.0,         # s
    integrator      = 'rk45',       # 'rk45' | 'lgvi'
    aero            = 'std',        # 'std' | 'bem'

    # ── Trajectory ────────────────────────────────────────────────
    trajectory      = 'circular',   # 'circular' | 'hover' | 'lissajous'
    radius          = 2.0,          # m  (for circular)
    circle_freq     = 0.2,          # Hz (circular: 0.2 = 5s per lap, omega = 2*pi*freq)

    # ── Controller ────────────────────────────────────────────────
    controller      = 'SE3',        # 'SE3' | 'Geo' | 'GeoAdaptive' |
                                    # 'L1-Geo' | 'INDI' | 'Xadap-NN' |
                                    # 'MPC' | 'L1-MPC'

    # ── Estimator ─────────────────────────────────────────────────
    estimator       = 'gt',         # 'gt' | 'liekf'

    # ── Wind ──────────────────────────────────────────────────────
    wind_mode       = 'constant',   # 'nowind' | 'constant' | 'turbulent' | 'cfd'
    wind_vec        = [3.0, 0.0, 0.0],   # constant wind [m/s]

    # turbulent wind
    turb_mean       = [3.0, 0.0, 0.0],
    turb_std        = [1.5, 0.6, 0.4],
    turb_tau        = 3.5,          # correlation time [s]
    turb_seed       = 0,            # -1 or None → random

    # CFD wind (van der Laan LES)
    cfd_file        = os.path.join(ROOT, 'wind field', '2019',
                                   'case6_highTi_UWV_10min_bin1.nc'),
    cfd_wake        = 5.0,          # downstream position
    cfd_unit        = 'R',          # 'R' | 'D' | 'm'
    cfd_lateral     = 0.0,          # lateral offset
    cfd_t_offset    = 0.0,          # time offset [s]

    # ── Sensor noise ──────────────────────────────────────────────
    mocap_params    = dict(
        pos_noise_density   = [0.0005]*3,
        vel_noise_density   = [0.0010]*3,
        att_noise_density   = [0.0005]*3,
        rate_noise_density  = [0.0005]*3,
        vel_artifact_max    = 5,
        vel_artifact_prob   = 0.001,
        rate_artifact_max   = 1,
        rate_artifact_prob  = 0.0002,
    ),

    # ── Output ────────────────────────────────────────────────────
    results_dir     = os.path.join(ROOT, 'results'),
)


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Helper class — dot-access, copy-on-write                        ║
# ╚════════════════════════════════════════════════════════════════════╝
class SimConfig(dict):
    """Dict subclass with attribute access:  cfg.sim_rate == cfg['sim_rate']"""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)
    def __setattr__(self, key, val):
        self[key] = val
    def copy(self):
        return SimConfig(super().copy())

def get_config(**overrides):
    """Return a fresh SimConfig with any overrides applied."""
    cfg = SimConfig(CONFIG)
    cfg.update(overrides)
    return cfg

# Convenience: default config
CFG = get_config()


# ╔════════════════════════════════════════════════════════════════════╗
# ║  Factory helpers — build objects from config                      ║
# ╚════════════════════════════════════════════════════════════════════╝

def make_x0(cfg):
    """Build initial state dict from config."""
    import sys
    sys.path.insert(0, ROOT)
    if cfg.aero == 'bem':
        from quad_param.quadrotor_with_bem import quad_params
    else:
        from quad_param.quadrotor import quad_params

    hover_omega = np.sqrt(quad_params['mass'] * 9.81 / (4 * quad_params['k_eta']))
    pos = np.array([cfg.radius, 0, 0]) if cfg.trajectory == 'circular' else np.zeros(3)
    return {
        'x': pos,
        'v': np.zeros(3),
        'q': np.array([0, 0, 0, 1]),
        'w': np.zeros(3),
        'wind': np.zeros(3),
        'rotor_speeds': np.array([hover_omega] * 4),
    }


def make_wind(cfg):
    """Build wind object from config."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
    from rotorpy.wind.default_winds import NoWind, ConstantWind

    if cfg.wind_mode == 'nowind':
        return NoWind()
    elif cfg.wind_mode == 'constant':
        return ConstantWind(*np.array(cfg.wind_vec))
    elif cfg.wind_mode == 'turbulent':
        from rotorpy.wind.default_winds import TurbulentWind
        seed = cfg.turb_seed if cfg.turb_seed and cfg.turb_seed >= 0 else None
        return TurbulentWind(mean=np.array(cfg.turb_mean),
                             std=np.array(cfg.turb_std),
                             corr_time=cfg.turb_tau, seed=seed)
    elif cfg.wind_mode == 'cfd':
        from rotorpy.wind.cfd_wind import CFDWind
        return CFDWind.from_params(cfg.cfd_file,
                                   wake_position=cfg.cfd_wake,
                                   unit=cfg.cfd_unit,
                                   lateral_offset=cfg.cfd_lateral,
                                   t_offset=cfg.cfd_t_offset)
    else:
        raise ValueError(f"Unknown wind_mode: {cfg.wind_mode!r}")


def make_sensors(cfg):
    """Return (imu, mocap) sensor pair."""
    import sys
    sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
    from rotorpy.sensors.imu import Imu
    from rotorpy.sensors.external_mocap import MotionCapture

    mp = {k: np.array(v) if isinstance(v, list) else v
          for k, v in cfg.mocap_params.items()}
    imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=cfg.sim_rate)
    mc  = MotionCapture(sampling_rate=cfg.sim_rate, mocap_params=mp,
                        with_artifacts=False)
    return imu, mc


def get_quad_params(cfg):
    """Return the right quad_params dict for the aero model."""
    import sys
    sys.path.insert(0, ROOT)
    if cfg.aero == 'bem':
        from quad_param.quadrotor_with_bem import quad_params
    else:
        from quad_param.quadrotor import quad_params
    return quad_params


class NullEstimator:
    """Placeholder when not using a real estimator."""
    def step(self, *a, **kw):
        return {'filter_state': np.zeros(1), 'covariance': np.zeros(1)}
    def get_state_estimate(self):
        return {}


def print_config(cfg, title='Simulation Configuration'):
    """Pretty-print a config."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(f"  Integrator : {cfg.integrator}")
    print(f"  Aero model : {cfg.aero}")
    print(f"  Estimator  : {cfg.estimator}")
    print(f"  Rate       : {cfg.sim_rate} Hz  (dt = {1/cfg.sim_rate:.4f} s)")
    print(f"  Controller : {cfg.controller}")
    freq = cfg.get('circle_freq', 0.2)
    if cfg.trajectory == 'circular':
        print(f"  Trajectory : circular  r={cfg.radius}m  freq={freq}Hz  "
              f"(T={1/freq:.1f}s/lap, v={2*3.1416*cfg.radius*freq:.2f}m/s)")
    else:
        print(f"  Trajectory : {cfg.trajectory}")
    if cfg.wind_mode == 'constant':
        print(f"  Wind       : constant {cfg.wind_vec} m/s")
    elif cfg.wind_mode == 'turbulent':
        print(f"  Wind       : turbulent  mean={cfg.turb_mean}  "
              f"std={cfg.turb_std}  tau={cfg.turb_tau}s")
    elif cfg.wind_mode == 'cfd':
        print(f"  Wind       : CFD  {cfg.cfd_wake}{cfg.cfd_unit} downstream")
    else:
        print(f"  Wind       : {cfg.wind_mode}")
    print(f"  T_final    : {cfg.t_final} s")
    print(f"{'='*60}\n")
