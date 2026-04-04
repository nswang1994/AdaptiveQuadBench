"""
Single-case simulation.  Edit the CONFIG overrides below, then:

    python run_single.py
"""

import sys, os, copy, time as timer, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))
sys.path.insert(0, os.path.join(ROOT, 'controller'))

from sim_config import (get_config, make_x0, make_wind, make_sensors,
                         get_quad_params, NullEstimator, print_config)

# ╔════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — edit here                                       ║
# ╠════════════════════════════════════════════════════════════════════╣
# ║  Anything not listed uses the default from sim_config.py          ║
# ╚════════════════════════════════════════════════════════════════════╝
cfg = get_config(
    # ── physics ───────────────────────────────────────────────────
    integrator      = 'lgvi',       # 'rk45' | 'lgvi'
    aero            = 'std',        # 'std' | 'bem'
    sim_rate        = 100,          # Hz
    t_final         = 120.0,         # s

    # ── controller ────────────────────────────────────────────────
    controller      = 'GeoAdaptive',# 'SE3' | 'Geo' | 'GeoAdaptive' |
                                    # 'L1-Geo' | 'INDI' | 'Xadap-NN' |
                                    # 'MPC' | 'L1-MPC'

    # ── estimator ─────────────────────────────────────────────────
    estimator       = 'gt',         # 'gt' | 'liekf'

    # ── trajectory ────────────────────────────────────────────────
    trajectory      = 'circular',   # 'circular' | 'hover' | 'lissajous'
    radius          = 30.0,          # m  (circular)
    circle_freq     = 1.0/60.0,     # Hz (circular, 0.2 = 5s/round, omega=2*pi*freq)

    # ── wind ──────────────────────────────────────────────────────
    wind_mode       = 'cfd',   # 'nowind' | 'constant' | 'turbulent' | 'cfd'
    # wind_vec        = [3.0, 0.0, 0.0],

    # turbulent
    # turb_mean     = [3.0, 0.0, 0.0],
    # turb_std      = [1.5, 0.6, 0.4],
    # turb_tau      = 3.5,
    # turb_seed     = 0,

    # cfd
    cfd_file      = 'wind field/2019/case6_highTi_UWV_10min_bin1.nc',
    cfd_wake      = 5.0,
    cfd_unit      = 'R',
    cfd_lateral   = 0.0,
    cfd_t_offset  = 0.0,

    # ── output ────────────────────────────────────────────────────
    save_fig        = None,         # e.g. 'results/run.png'  (None → auto)
    no_plot         = False,
)
# ╔════════════════════════════════════════════════════════════════════╗
# ║  Or use a preset:  uncomment ONE line below to override above     ║
# ╠════════════════════════════════════════════════════════════════════╣
# ║  C1: RK45+Std+GT@100   C5: LGVI+Std+GT@100   C9:  LGVI+BEM+LIEKF@100  ║
# ║  C2: RK45+BEM+GT@100   C6: LGVI+BEM+GT@100   C10: LGVI+BEM+LIEKF@500  ║
# ║  C3: RK45+BEM+GT@500   C7: LGVI+Std+GT@500                       ║
# ║  C4: RK45+Std+GT@500   C8: LGVI+BEM+GT@500                       ║
# ╚════════════════════════════════════════════════════════════════════╝
PRESETS = {
    'C1':  dict(integrator='rk45', aero='std', estimator='gt',    sim_rate=100),
    'C2':  dict(integrator='rk45', aero='bem', estimator='gt',    sim_rate=100),
    'C3':  dict(integrator='rk45', aero='bem', estimator='gt',    sim_rate=500),
    'C4':  dict(integrator='rk45', aero='std', estimator='gt',    sim_rate=500),
    'C5':  dict(integrator='lgvi', aero='std', estimator='gt',    sim_rate=100),
    'C6':  dict(integrator='lgvi', aero='bem', estimator='gt',    sim_rate=100),
    'C7':  dict(integrator='lgvi', aero='std', estimator='gt',    sim_rate=500),
    'C8':  dict(integrator='lgvi', aero='bem', estimator='gt',    sim_rate=500),
    'C9':  dict(integrator='lgvi', aero='bem', estimator='liekf', sim_rate=100),
    'C10': dict(integrator='lgvi', aero='bem', estimator='liekf', sim_rate=500),
}
# PRESET = 'C6'          # ← uncomment & set to use a preset
# cfg.update(PRESETS[PRESET])

# =====================================================================
#  Controller registry
# =====================================================================
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.simulate import simulate
from rotorpy.world import World
from rotorpy.estimators.liekf_ins import LIEKFINS

from rotorpy.controllers.quadrotor_control import SE3Control
from geometric_control import GeoControl
from geometric_adaptive_controller import GeometricAdaptiveController
from geometric_control_l1 import L1_GeoControl
from indi_adaptive_controller import INDIAdaptiveController

try:
    from Xadap_NN_control import Xadap_NN_control
    _HAS_XADAP = True
except Exception as _e:
    _HAS_XADAP = False
    print(f"[warn] Xadap-NN unavailable: {_e}")

try:
    from quadrotor_control_mpc import ModelPredictiveControl
    from quadrotor_control_mpc_l1 import L1_ModelPredictiveControl
    _HAS_MPC = True
except ImportError:
    _HAS_MPC = False

CONTROLLERS = {
    'SE3':         lambda p, dt, **kw: SE3Control(p),
    'Geo':         lambda p, dt, **kw: GeoControl(p),
    'GeoAdaptive': lambda p, dt, **kw: GeometricAdaptiveController(p, dt=dt),
    'L1-Geo':      lambda p, dt, **kw: L1_GeoControl(p),
    'INDI':        lambda p, dt, **kw: INDIAdaptiveController(p, dt=dt),
}
if _HAS_XADAP:
    CONTROLLERS['Xadap-NN'] = lambda p, dt, **kw: Xadap_NN_control(p)
if _HAS_MPC:
    CONTROLLERS['MPC']    = lambda p, dt, **kw: ModelPredictiveControl(
        p, trajectory=kw['traj'], sim_rate=kw['rate'], t_final=kw['t_final'])
    CONTROLLERS['L1-MPC'] = lambda p, dt, **kw: L1_ModelPredictiveControl(
        p, trajectory=kw['traj'], sim_rate=kw['rate'], t_final=kw['t_final'])


def build_trajectory(cfg):
    if cfg.trajectory == 'circular':
        return CircularTraj(radius=cfg.radius, freq=cfg.get('circle_freq', 0.2))
    elif cfg.trajectory == 'hover':
        return HoverTraj(x0=np.array([0, 0, 0]))
    elif cfg.trajectory == 'lissajous':
        return TwoDLissajous(A=2, B=2, a=1, b=2, delta=np.pi/2, height=0)
    else:
        raise ValueError(f"Unknown trajectory: {cfg.trajectory}")


# =====================================================================
#  Main
# =====================================================================
def main():
    # Report optional controllers
    if not _HAS_XADAP:
        print("[info] Xadap-NN unavailable (needs onnxruntime)")
    if not _HAS_MPC:
        print("[info] MPC / L1-MPC unavailable (needs acados)")

    print_config(cfg)

    dt = 1.0 / cfg.sim_rate
    params = get_quad_params(cfg)
    x0 = make_x0(cfg)

    p = copy.deepcopy(params)
    vehicle = Multirotor(p, control_abstraction='cmd_motor_speeds',
                         integrator=cfg.integrator)
    vehicle.initial_state = copy.deepcopy(x0)

    imu_s, mc_s = make_sensors(cfg)

    use_est = (cfg.estimator == 'liekf')
    est  = LIEKFINS(params, dt=dt) if use_est else NullEstimator()
    traj = build_trajectory(cfg)
    ctrl = CONTROLLERS[cfg.controller](copy.deepcopy(params), dt,
                                       traj=traj, rate=cfg.sim_rate,
                                       t_final=cfg.t_final)
    wind = make_wind(cfg)
    world = World.empty((-10, 10, -10, 10, -10, 10))

    # --- Run simulation ---
    print("Running simulation...")
    t0 = timer.perf_counter()
    try:
        result = simulate(
            world, copy.deepcopy(x0), vehicle, ctrl, traj, wind,
            imu_s, mc_s, est,
            cfg.t_final, dt, safety_margin=0.25,
            use_mocap=False, use_estimator=use_est)
    except Exception as exc:
        print(f"\nSIMULATION FAILED: {exc}")
        traceback.print_exc()
        return
    wall = timer.perf_counter() - t0

    t_arr = result[0]
    s_arr = result[1]
    f_arr = result[3]
    ex    = result[8]

    # --- Metrics ---
    e_pos = np.linalg.norm(s_arr['x'] - f_arr['x'], axis=1)
    idx = t_arr > 3.0
    if np.sum(idx) > 0 and t_arr[-1] > 5.0:
        mean_e = e_pos[idx].mean()
        max_e  = e_pos[idx].max()
        std_e  = e_pos[idx].std()
        print(f"\n  Sim ended at t = {t_arr[-1]:.2f} s  (exit: {ex})")
        print(f"  Wall time    : {wall:.2f} s")
        print(f"  Tracking (t>3s): mean={mean_e:.4f} m  max={max_e:.4f} m  std={std_e:.4f} m")
    else:
        mean_e = None
        print(f"\n  CRASHED at t = {t_arr[-1]:.2f} s  (exit: {ex})")
        print(f"  Wall time: {wall:.2f} s")

    if cfg.get('no_plot', False):
        return

    # --- Figure 1: main 2x2 performance plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f'{cfg.controller} | {cfg.integrator}+{cfg.aero}+{cfg.estimator}@{cfg.sim_rate}Hz | '
                 f'traj={cfg.trajectory} | wind={cfg.wind_mode}',
                 fontsize=12, fontweight='bold')

    # 1) 3D trajectory
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    axes[0, 0].set_visible(False)
    ax.plot(s_arr['x'][:, 0], s_arr['x'][:, 1], s_arr['x'][:, 2], 'b-', label='actual', linewidth=0.8)
    ax.plot(f_arr['x'][:, 0], f_arr['x'][:, 1], f_arr['x'][:, 2], 'r--', label='desired', linewidth=0.8)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.legend(fontsize=8); ax.set_title('3D Trajectory')

    # 2) Position tracking error
    ax2 = axes[0, 1]
    ax2.plot(t_arr, e_pos, 'b-', linewidth=0.8)
    ax2.axhline(y=0, color='k', linewidth=0.3)
    if mean_e is not None:
        ax2.axhline(y=mean_e, color='r', linestyle='--', linewidth=0.8, label=f'mean={mean_e:.3f}m')
        ax2.legend(fontsize=8)
    ax2.set_xlabel('Time [s]'); ax2.set_ylabel('Pos Error [m]')
    ax2.set_title('Position Tracking Error'); ax2.grid(True, alpha=0.3)

    # 3) Position components
    ax3 = axes[1, 0]
    for i, label in enumerate(['x', 'y', 'z']):
        ax3.plot(t_arr, s_arr['x'][:, i], linewidth=0.8, label=f'{label} actual')
        ax3.plot(t_arr, f_arr['x'][:, i], '--', linewidth=0.8, label=f'{label} desired')
    ax3.set_xlabel('Time [s]'); ax3.set_ylabel('Position [m]')
    ax3.set_title('Position Components'); ax3.legend(fontsize=7, ncol=2); ax3.grid(True, alpha=0.3)

    # 4) Rotor speeds
    ax4 = axes[1, 1]
    for i in range(4):
        ax4.plot(t_arr, s_arr['rotor_speeds'][:, i], linewidth=0.6, label=f'rotor {i+1}')
    ax4.set_xlabel('Time [s]'); ax4.set_ylabel('Rotor Speed [rad/s]')
    ax4.set_title('Rotor Speeds'); ax4.legend(fontsize=7); ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_path = cfg.get('save_fig') or 'run_single_result.png'
    if cfg.get('save_fig'):
        os.makedirs(os.path.dirname(fig1_path) or '.', exist_ok=True)
    plt.savefig(fig1_path, dpi=150)
    print(f"  Figure saved to: {fig1_path}")

    # --- Figure 2 (CFD only): wind field + trajectory overlay ---
    is_cfd = (cfg.wind_mode == 'cfd')
    if is_cfd:
        from matplotlib.patches import Rectangle

        turbine_x_w = -wind.origin_les[0]
        turbine_y_w = -wind.origin_les[1]
        rotor_r = wind.R

        les_x_min_w = wind.x_arr[0]  - wind.origin_les[0]
        les_x_max_w = wind.x_arr[-1] - wind.origin_les[0]
        les_y_min_w = wind.y_arr[0]  - wind.origin_les[1]
        les_y_max_w = wind.y_arr[-1] - wind.origin_les[1]

        # UAV trajectory bounding box (adaptive to any trajectory size)
        traj_xmin = min(s_arr['x'][:, 0].min(), f_arr['x'][:, 0].min())
        traj_xmax = max(s_arr['x'][:, 0].max(), f_arr['x'][:, 0].max())
        traj_ymin = min(s_arr['x'][:, 1].min(), f_arr['x'][:, 1].min())
        traj_ymax = max(s_arr['x'][:, 1].max(), f_arr['x'][:, 1].max())
        traj_cx = 0.5 * (traj_xmin + traj_xmax)
        traj_cy = 0.5 * (traj_ymin + traj_ymax)
        traj_half_x = (traj_xmax - traj_xmin) / 2
        traj_half_y = (traj_ymax - traj_ymin) / 2
        traj_half = max(traj_half_x, traj_half_y) + 1.0
        # Scale factor: how big the trajectory is relative to the domain
        traj_scale = max(traj_half, 5.0)

        def sample_wind_grid(gx, gy, t_query):
            GX, GY = np.meshgrid(gx, gy)
            n = GX.size
            x_les = GX.ravel() + wind.origin_les[0]
            y_les = GY.ravel() + wind.origin_les[1]
            t_les = t_query + wind.t_offset
            if not wind.mean_only:
                if wind.loop_time:
                    span = wind.t_max - wind.t_min
                    t_les = wind.t_min + (t_les - wind.t_min) % span
                else:
                    t_les = np.clip(t_les, wind.t_min, wind.t_max)
                pts = np.column_stack([np.full(n, t_les), x_les, y_les])
            else:
                pts = np.column_stack([x_les, y_les])
            U = wind._iU(pts).reshape(GX.shape)
            V = wind._iV(pts).reshape(GX.shape)
            U = np.where(np.isfinite(U), U, wind.U0)
            V = np.where(np.isfinite(V), V, 0.0)
            mag = np.sqrt(U**2 + V**2)
            return GX, GY, U, V, mag

        def draw_turbine(ax):
            ax.plot([turbine_x_w, turbine_x_w],
                    [turbine_y_w - rotor_r, turbine_y_w + rotor_r],
                    'k-', linewidth=3, zorder=10, label='Turbine rotor')
            ax.plot(turbine_x_w, turbine_y_w, 'k^', ms=9, zorder=11)

        t_snap = min(cfg.t_final / 2.0, 300.0)  # cap at LES record half
        dist_m = wind.origin_les[0]

        fig2, (ax_g, ax_z) = plt.subplots(1, 2, figsize=(18, 7))
        fig2.suptitle(f'CFD Wind Field  |  {os.path.basename(cfg.cfd_file)}  |  '
                      f'UAV at {cfg.cfd_wake}{cfg.cfd_unit} downstream  |  '
                      f't = {t_snap:.1f} s',
                      fontsize=12, fontweight='bold')

        # Left: global wake view (turbine → UAV)
        gx_min = max(les_x_min_w, turbine_x_w - 1.0 * wind.D)
        gx_max = min(les_x_max_w, traj_cx + traj_half + 1.0 * wind.D)
        gy_need = max(2.0 * wind.D, abs(traj_cy) + traj_half + wind.D)
        gy_min = max(les_y_min_w, -gy_need)
        gy_max = min(les_y_max_w,  gy_need)
        # Adaptive quiver scale based on global domain span
        global_span = max(gx_max - gx_min, gy_max - gy_min)
        q_scale_g = max(40, global_span * 0.3)

        ng = 150
        GX_g, GY_g, U_g, V_g, mag_g = sample_wind_grid(
            np.linspace(gx_min, gx_max, ng), np.linspace(gy_min, gy_max, ng), t_snap)
        cf_g = ax_g.contourf(GX_g, GY_g, mag_g, levels=30, cmap='coolwarm', alpha=0.85)
        plt.colorbar(cf_g, ax=ax_g, label='$|\\mathbf{w}|$ [m/s]', shrink=0.85)
        skip = max(1, ng // 20)
        ax_g.quiver(GX_g[::skip, ::skip], GY_g[::skip, ::skip],
                    U_g[::skip, ::skip], V_g[::skip, ::skip],
                    color='k', alpha=0.3, scale=q_scale_g, width=0.003)
        draw_turbine(ax_g)
        ax_g.annotate('Wind Turbine', (turbine_x_w, turbine_y_w + rotor_r),
                      textcoords='offset points', xytext=(8, 8), fontsize=9, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color='k', lw=1.2))
        # UAV region box — margin scales with trajectory
        box_margin = max(traj_half * 0.5, 5.0)
        box_half = traj_half + box_margin
        rect = Rectangle((traj_cx - box_half, traj_cy - box_half),
                          2*box_half, 2*box_half, lw=2, edgecolor='lime',
                          facecolor='none', ls='--', zorder=12)
        ax_g.add_patch(rect)
        ax_g.annotate('UAV region', (traj_cx + box_half, traj_cy + box_half),
                      textcoords='offset points', xytext=(6, 6), fontsize=9,
                      fontweight='bold', color='lime',
                      arrowprops=dict(arrowstyle='->', color='lime', lw=1.2))
        # Distance annotation — offset scales with domain
        ann_offset = max(10, (gy_max - gy_min) * 0.03)
        ax_g.annotate('', xy=(traj_cx, gy_min + ann_offset),
                      xytext=(turbine_x_w, gy_min + ann_offset),
                      arrowprops=dict(arrowstyle='<->', color='white', lw=1.5))
        ax_g.text(0.5*(turbine_x_w + traj_cx), gy_min + ann_offset * 1.5,
                  f'{dist_m:.0f} m ({cfg.cfd_wake}{cfg.cfd_unit})',
                  ha='center', fontsize=8, fontweight='bold', color='white',
                  bbox=dict(facecolor='black', alpha=0.5, pad=2))
        ax_g.set_xlabel('X (streamwise) [m]'); ax_g.set_ylabel('Y (lateral) [m]')
        ax_g.set_title('Global Wake View'); ax_g.legend(fontsize=7, loc='upper right')
        ax_g.set_aspect('equal'); ax_g.grid(True, alpha=0.3)

        # Right: zoomed UAV region — margin scales with trajectory
        zoom_margin = max(traj_scale * 0.3, 3.0)
        zh_x = traj_half_x + zoom_margin
        zh_y = traj_half_y + zoom_margin
        GX_z, GY_z, U_z, V_z, mag_z = sample_wind_grid(
            np.linspace(traj_cx - zh_x, traj_cx + zh_x, 100),
            np.linspace(traj_cy - zh_y, traj_cy + zh_y, 100), t_snap)
        cf_z = ax_z.contourf(GX_z, GY_z, mag_z, levels=30, cmap='coolwarm', alpha=0.85)
        plt.colorbar(cf_z, ax=ax_z, label='$|\\mathbf{w}|$ [m/s]', shrink=0.85)
        # Adaptive quiver for zoom
        zoom_span = max(2*zh_x, 2*zh_y)
        q_scale_z = max(20, zoom_span * 0.8)
        sk = max(1, 100 // 15)
        ax_z.quiver(GX_z[::sk, ::sk], GY_z[::sk, ::sk],
                    U_z[::sk, ::sk], V_z[::sk, ::sk],
                    color='k', alpha=0.4, scale=q_scale_z, width=0.004)
        ax_z.plot(f_arr['x'][:, 0], f_arr['x'][:, 1], 'r--', lw=1.5, label='desired')
        ax_z.plot(s_arr['x'][:, 0], s_arr['x'][:, 1], 'b-', lw=1.2, label='actual')
        ax_z.plot(s_arr['x'][0, 0], s_arr['x'][0, 1], 'go', ms=7, zorder=11, label='start')
        ax_z.plot(s_arr['x'][-1, 0], s_arr['x'][-1, 1], 'rs', ms=7, zorder=11, label='end')
        # Turbine direction annotation — position scales
        ax_z.annotate(f'Turbine\n{dist_m:.0f} m',
                      xy=(traj_cx - zh_x * 0.9, traj_cy), fontsize=8, fontweight='bold',
                      color='k', ha='left', arrowprops=dict(arrowstyle='<-', color='k', lw=1.5))
        ax_z.set_xlabel('X [m]'); ax_z.set_ylabel('Y [m]')
        ax_z.set_title('UAV Region (zoomed)'); ax_z.legend(fontsize=7, loc='upper right')
        ax_z.set_aspect('equal'); ax_z.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_path = fig1_path.replace('.png', '_cfd_wind.png')
        plt.savefig(fig2_path, dpi=150)
        print(f"  CFD wind figure saved to: {fig2_path}")

        # Figure 3: wind time series
        fig3, ax_ts = plt.subplots(figsize=(10, 4))
        wind_hist = np.array([wind.update(t_arr[k], s_arr['x'][k])
                              for k in range(len(t_arr))])
        ax_ts.plot(t_arr, wind_hist[:, 0], lw=0.8, label='$w_x$ (streamwise)')
        ax_ts.plot(t_arr, wind_hist[:, 1], lw=0.8, label='$w_y$ (lateral)')
        ax_ts.plot(t_arr, wind_hist[:, 2], lw=0.8, label='$w_z$ (vertical)')
        ax_ts.axhline(y=wind.U0, color='gray', ls=':', lw=0.8,
                      label=f'freestream $U_0$={wind.U0} m/s')
        ax_ts.set_xlabel('Time [s]'); ax_ts.set_ylabel('Wind [m/s]')
        ax_ts.set_title('Wind Experienced by UAV Along Trajectory')
        ax_ts.legend(fontsize=8); ax_ts.grid(True, alpha=0.3)
        plt.tight_layout()
        fig3_path = fig1_path.replace('.png', '_cfd_wind_ts.png')
        plt.savefig(fig3_path, dpi=150)
        print(f"  CFD wind time series saved to: {fig3_path}")


if __name__ == '__main__':
    main()
