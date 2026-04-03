"""
Quadrotor parameters with BEM (Blade Element Momentum) rotor aerodynamics.

Based on the vehicle in:
    Davoudi et al., "Quad-rotor Flight Simulation in Realistic Atmospheric Conditions,"
    arXiv:1902.01465, 2019.

Vehicle: 0.69 kg quadrotor, arm length 0.225 m, rotor radius 7.62 cm.
Rotor:   2-blade, chord ~1.10 cm, twist from 25 deg (root) to 5 deg (tip).
Airfoil: Cl_alpha = 1.7059,  alpha_L0 = 4 deg.
"""

import numpy as np

d = 0.225  # Arm length [m]  (Davoudi: l = 0.225 m, half of the rotor-to-rotor distance)

quad_params = {
    # =====================================================================
    #  Inertial properties  (Davoudi Section VI, p.25)
    # =====================================================================
    'mass': 0.69,        # kg
    'Ixx':  0.0469,      # kg*m^2
    'Iyy':  0.0358,      # kg*m^2
    'Izz':  0.0673,      # kg*m^2
    'Ixy':  0.0,
    'Iyz':  0.0,
    'Ixz':  0.0,
    'arm_length': d,
    'com': np.array([0.0, 0.0, 0.0]),

    # =====================================================================
    #  Geometric properties — "+" configuration  (Davoudi Fig. 9)
    # =====================================================================
    #           x_B (front)
    #            ^
    #       mot4 | mot1
    #            |
    #  y_B <-----+-----
    #            |
    #       mot3 | mot2
    #
    'num_rotors': 4,
    'rotor_radius': 0.0762,   # 7.62 cm  (Davoudi Fig. 8)
    'rotor_pos': {
        'r1': np.array([ d,  0, 0]),   # front
        'r2': np.array([ 0, -d, 0]),   # right
        'r3': np.array([-d,  0, 0]),   # back
        'r4': np.array([ 0,  d, 0]),   # left
    },
    'rotor_directions': np.array([-1, 1, -1, 1]),  # alternating for yaw balance
    'rotor_efficiency': np.array([1.0, 1.0, 1.0, 1.0]),

    'rI': np.array([0, 0, 0]),

    # =====================================================================
    #  Frame aerodynamic properties
    #  When BEM is active, frame parasitic drag is handled by the lumped
    #  drag model (Davoudi Eq. 12), so we set c_Dx/y/z = 0.
    # =====================================================================
    'c_Dx': 0.0,
    'c_Dy': 0.0,
    'c_Dz': 0.0,

    # =====================================================================
    #  Rotor properties — lumped-parameter fallback
    #  These are used ONLY when aero_model != 'bem'.
    #  Values from Davoudi Fig. 8 curve fit:
    #    b = 1.5652e-8  N/RPM^2 -> k_eta = b*(60/(2*pi))^2 = 1.428e-6 N/(rad/s)^2
    #    k = 2.0862e-10 Nm/RPM^2 -> k_m = k*(60/(2*pi))^2 = 1.903e-8 Nm/(rad/s)^2
    #    Note: T = b*RPM^2 = k_eta*omega^2, RPM = omega*60/(2pi)
    #          k_eta = b * (60/(2pi))^2 = b * 91.19
    # =====================================================================
    'k_eta': 1.428e-6,
    'k_m':   1.903e-8,
    'k_d':   1.19e-04,
    'k_z':   2.32e-04,
    'k_flap': 0.0,

    # =====================================================================
    #  Motor properties
    # =====================================================================
    'tau_m': 0.005,
    'rotor_speed_min': 0.0,
    'rotor_speed_max': 2100.0,   # ~20000 RPM
    'motor_noise_std': 50,

    # =====================================================================
    #  BEM rotor parameters  (Davoudi Section III)
    # =====================================================================
    'use_bem': True,
    'bem_params': {
        'R':          0.0762,           # rotor radius [m]
        'R_min':      0.00762,          # hub cutout = 0.1 * R [m]
        'N_b':        2,                # number of blades
        'c_bar':      0.0110,           # mean chord [m]  (Davoudi: 1.10 cm)
        'c_func':     None,             # constant chord (set callable for variable chord)
        'theta_root': np.radians(25),   # blade pitch at root [rad]
        'theta_tip':  np.radians(5),    # blade pitch at tip  [rad]
        'Cl_alpha':   1.7059,           # 2-D lift curve slope [1/rad]
        'alpha_L0':   np.radians(4),    # zero-lift AoA [rad]
        'Cd0':        0.02,             # profile drag coefficient
        'rho':        1.225,            # air density [kg/m^3]
        'c_d_bar':    0.04,             # lumped drag coefficient (Davoudi Eq. 12)
        'f_over_A':   0.005,            # equivalent flat plate area ratio
        'n_elements': 30,               # number of radial integration stations
        'thrust_scale': 2.29,           # calibration factor to match experimental
                                        # hover data (Davoudi Fig. 8):
                                        # BEM hover = 15735 RPM, expt = 10395 RPM
                                        # scale = (15735/10395)^2 = 2.29
    },
}
