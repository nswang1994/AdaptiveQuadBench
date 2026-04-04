"""
Compare lumped-parameter vs. BEM rotor models on the SAME vehicle.

Both simulations use:
  - Same 0.826 kg X-config quadrotor
  - Same SE3 geometric controller with same gains
  - Same circular trajectory (r=2m)
  - Same constant wind (3 m/s in +x)

The ONLY difference is the rotor aerodynamics:
  1. Lumped: T = k_eta * omega^2  (wind-insensitive)
  2. BEM:   Blade Element Momentum theory (wind-sensitive, Davoudi et al.)

BEM thrust_scale is auto-calibrated so hover thrust matches exactly.
"""

import sys, os
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'rotorpy'))

from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.circular_traj import CircularTraj
from rotorpy.wind.default_winds import ConstantWind, NoWind

SIM_RATE = 100
T_FINAL  = 10
RADIUS   = 2.0

# Initial state (same for all runs)
from quad_param.quadrotor import quad_params as base_params
k_eta = base_params['k_eta']
mass  = base_params['mass']
hover_omega = np.sqrt(mass * 9.81 / (4 * k_eta))

x0 = {'x': np.array([RADIUS, 0, 0]),
      'v': np.zeros(3),
      'q': np.array([0, 0, 0, 1]),
      'w': np.zeros(3),
      'wind': np.array([0, 0, 0]),
      'rotor_speeds': np.array([hover_omega]*4)}

def run_sim(label, params, wind_profile):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    p = copy.deepcopy(params)
    vehicle = Multirotor(p, control_abstraction='cmd_motor_speeds')
    ctrl    = SE3Control(p)
    vehicle.initial_state = copy.deepcopy(x0)
    env = Environment(vehicle=vehicle, controller=ctrl,
                      trajectory=CircularTraj(radius=RADIUS),
                      wind_profile=wind_profile, sim_rate=SIM_RATE)
    res = env.run(t_final=T_FINAL, use_mocap=False, terminate=False,
                  plot=False, animate_bool=False, verbose=True)
    return res

# ---- Run 1: Lumped, no wind ----
lp_nw = copy.deepcopy(base_params); lp_nw['use_bem'] = False
res_nw = run_sim("Lumped-parameter, NO wind", lp_nw, NoWind())

# ---- Run 2: Lumped + wind ----
wind = ConstantWind(3.0, 0.0, 0.0)
lp_w = copy.deepcopy(base_params); lp_w['use_bem'] = False
res_lp = run_sim("Lumped-parameter + 3 m/s wind", lp_w, wind)

# ---- Run 3: BEM + wind ----
from quad_param.quadrotor_with_bem import quad_params as bem_params
res_bem = run_sim("BEM (Davoudi) + 3 m/s wind", bem_params, wind)

# ---- Run 4: BEM, no wind ----
res_bem_nw = run_sim("BEM (Davoudi), NO wind", bem_params, NoWind())

# =====================================================================
#  Plotting
# =====================================================================
print(f"\n{'='*60}\n  Generating plots...\n{'='*60}")

datasets = [
    (res_nw,     'Lumped, no wind', 'k--', 0.4),
    (res_lp,     'Lumped + wind',   'b-',  1.0),
    (res_bem_nw, 'BEM, no wind',    'g--', 0.5),
    (res_bem,    'BEM + wind',      'r-',  1.0),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Lumped vs BEM Rotor Model — Same Vehicle (0.826 kg)\nCircular trajectory, 3 m/s wind in +x',
             fontsize=13, fontweight='bold')

# helpers
def pos_err(res):
    return np.linalg.norm(res['state']['x'] - res['flat']['x'], axis=1)

def speed(res):
    return np.linalg.norm(res['state']['v'], axis=1)

def euler(res):
    return np.array([Rotation.from_quat(res['state']['q'][k]).as_euler('xyz', degrees=True)
                     for k in range(len(res['time']))])

# (0,0) XY trajectory
ax = axes[0,0]
for res, lbl, ls, a in datasets:
    ax.plot(res['state']['x'][:,0], res['state']['x'][:,1], ls, alpha=a, label=lbl)
ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
ax.set_title('XY Trajectory'); ax.legend(fontsize=7); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

# (0,1) Position error
ax = axes[0,1]
for res, lbl, ls, a in datasets:
    ax.plot(res['time'], pos_err(res), ls, alpha=a, label=lbl)
ax.set_xlabel('Time [s]'); ax.set_ylabel('||e_pos|| [m]')
ax.set_title('Position Tracking Error'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,0) Speed
ax = axes[1,0]
for res, lbl, ls, a in datasets:
    ax.plot(res['time'], speed(res), ls, alpha=a, label=lbl)
ax.set_xlabel('Time [s]'); ax.set_ylabel('Speed [m/s]')
ax.set_title('Vehicle Speed'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (1,1) Rotor speed R1
ax = axes[1,1]
for res, lbl, ls, a in datasets:
    ax.plot(res['time'], res['state']['rotor_speeds'][:,0], ls, alpha=a, label=lbl+' R1')
ax.set_xlabel('Time [s]'); ax.set_ylabel('Rotor speed [rad/s]')
ax.set_title('Rotor 1 Speed (front-left)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (2,0) Pitch angle
ax = axes[2,0]
for res, lbl, ls, a in datasets:
    e = euler(res)
    ax.plot(res['time'], e[:,1], ls, alpha=a, label=lbl)
ax.set_xlabel('Time [s]'); ax.set_ylabel('Pitch [deg]')
ax.set_title('Pitch Angle'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

# (2,1) Altitude (z)
ax = axes[2,1]
for res, lbl, ls, a in datasets:
    ax.plot(res['time'], res['state']['x'][:,2], ls, alpha=a, label=lbl)
ax.set_xlabel('Time [s]'); ax.set_ylabel('z [m]')
ax.set_title('Altitude'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(ROOT, 'bem_vs_lumped_comparison.png')
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved to: {out_path}")

# Summary
print(f"\n{'='*60}")
print(f"  Summary (t > 2s)")
print(f"{'='*60}")
print(f"{'Model':<25} {'Mean err [m]':>12} {'Max err [m]':>12} {'Exit':>15}")
print("-"*65)
for res, lbl, _, _ in datasets:
    t = res['time']
    idx = t > 2.0
    e = pos_err(res)
    if np.sum(idx) > 0:
        print(f"{lbl:<25} {e[idx].mean():>12.4f} {e[idx].max():>12.4f} {res['exit'].name:>15}")
    else:
        print(f"{lbl:<25} {'N/A':>12} {'N/A':>12} {res['exit'].name:>15}")
