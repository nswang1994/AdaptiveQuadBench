"""
Same physical quadrotor as quadrotor.py (0.826 kg, X-config, d=0.166 m),
but with BEM rotor parameters added.

The BEM parameters are estimated for a rotor with:
  - radius 0.10 m (matching rotor_radius in original params)
  - 2 blades, chord ~1.5 cm
  - twist from 20 deg (root) to 5 deg (tip)
  - Cl_alpha calibrated so that BEM hover matches k_eta hover

This allows us to run the SAME vehicle with SAME controller gains
using either lumped or BEM dynamics, isolating the aero model effect.
"""

import numpy as np

d = 0.166  # Arm length from original quadrotor.py

# First, compute the target hover conditions from the original k_eta
_k_eta_orig = 7.64e-6   # N/(rad/s)^2
_mass = 0.826            # kg
_g = 9.81
_T_hover_per_rotor = _mass * _g / 4.0   # ~ 2.025 N
_omega_hover = np.sqrt(_T_hover_per_rotor / _k_eta_orig)  # ~ 514.9 rad/s

# BEM blade parameters for a 10cm-radius rotor
_R = 0.10             # rotor radius [m]
_N_b = 2              # blades
_c_bar = 0.015        # chord [m] (1.5 cm typical for this class)
_theta_root = np.radians(20)
_theta_tip  = np.radians(5)
_rho = 1.225

# We'll calibrate thrust_scale so BEM hover thrust matches k_eta hover thrust.
# This is done automatically below.

quad_params = {
    # Inertial properties — SAME as original
    'mass': 0.826,
    'Ixx':  0.0047,
    'Iyy':  0.005,
    'Izz':  0.0074,
    'Ixy':  0.0,
    'Iyz':  0.0,
    'Ixz':  0.0,
    'arm_length': d,
    'com': np.array([0.0, 0.0, 0.0]),

    # Geometry — SAME as original (X-config)
    'num_rotors': 4,
    'rotor_radius': _R,
    'rotor_pos': {
        'r1': d*np.array([ np.sin(np.pi/4),   np.cos(np.pi/4), 0]),
        'r2': d*np.array([ np.sin(np.pi/4),  -np.cos(np.pi/4), 0]),
        'r3': d*np.array([-np.sin(np.pi/4),  -np.cos(np.pi/4), 0]),
        'r4': d*np.array([-np.sin(np.pi/4),   np.cos(np.pi/4), 0]),
    },
    'rotor_directions': np.array([1,-1,1,-1]),
    'rotor_efficiency': np.array([1.0,1.0,1.0,1.0]),
    'rI': np.array([0,0,0]),

    # Drag — SAME as original
    'cd1_x': 0.62,
    'cd1_y': 0.62,
    'cd1_z': 0.62,
    'cdz_h': 0.00,
    'c_Dx': 0.00,
    'c_Dy': 0.00,
    'c_Dz': 0.00,

    # Rotor — SAME as original (used by controller for allocation)
    'k_eta': 7.64e-6,
    'k_m': 7.64e-6 * 0.0140,
    'k_d': 1.19e-04,
    'k_z': 2.32e-04,
    'k_flap': 0.0,

    # Motor — SAME as original
    'tau_m': 0.005,
    'rotor_speed_min': 0.0,
    'rotor_speed_max': 1000.0,
    'motor_noise_std': 50,

    # BEM parameters
    'use_bem': True,
    'bem_params': {
        'R':          _R,
        'R_min':      0.1 * _R,
        'N_b':        _N_b,
        'c_bar':      _c_bar,
        'c_func':     None,
        'theta_root': _theta_root,
        'theta_tip':  _theta_tip,
        'Cl_alpha':   5.7,          # 2*pi for thin airfoil
        'alpha_L0':   np.radians(2),
        'Cd0':        0.02,
        'rho':        _rho,
        'c_d_bar':    0.04,
        'f_over_A':   0.005,
        'n_elements': 30,
        'thrust_scale': 1.0,       # will be calibrated below
    },
}

# ── Auto-calibrate thrust_scale so BEM hover = k_eta hover ──
def _calibrate():
    from rotorpy.vehicles.bem_rotor import BEMRotor
    bem = BEMRotor(quad_params['bem_params'])
    T_bem, _, _ = bem.compute_thrust_torque(_omega_hover, np.array([0, 0, 0]))
    if T_bem > 0:
        scale = _T_hover_per_rotor / T_bem
        quad_params['bem_params']['thrust_scale'] = scale
        # print(f"[BEM calibration] T_bem_raw={T_bem:.4f} N, T_target={_T_hover_per_rotor:.4f} N, scale={scale:.4f}")

_calibrate()
