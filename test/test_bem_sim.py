"""
Integration test: run a short hover simulation with BEM vs lumped-parameter model.
Compares the two models under the same wind condition.

Run:  python -m test.test_bem_sim   (from AdaptiveQuadBench root)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rotorpy'))

import numpy as np
from rotorpy.vehicles.multirotor import Multirotor

# ---- Load both parameter sets ----
from quad_param.quadrotor import quad_params as params_lumped
from quad_param.quadrotor_bem import quad_params as params_bem

# ---- Hover initial state ----
def make_initial_state(quad_params):
    """Create hover initial state with correct rotor speeds."""
    mass = quad_params['mass']
    g = 9.81
    k_eta = quad_params['k_eta']
    T_hover = mass * g / quad_params['num_rotors']
    omega_hover = np.sqrt(T_hover / k_eta)
    return {
        'x': np.array([0.0, 0.0, 0.0]),
        'v': np.zeros(3),
        'q': np.array([0, 0, 0, 1]),
        'w': np.zeros(3),
        'wind': np.array([0.0, 0.0, 0.0]),
        'rotor_speeds': np.full(4, omega_hover),
        'ext_force': np.zeros(3),
        'ext_torque': np.zeros(3),
    }

# ---- Test: instantiate BEM multirotor ----
print("=" * 60)
print("BEM Simulation Integration Test")
print("=" * 60)

print("\n--- Instantiating Multirotor with BEM ---")
state_bem = make_initial_state(params_bem)
quad_bem = Multirotor(params_bem, initial_state=state_bem,
                      control_abstraction='cmd_motor_speeds', aero=True)
print(f"  aero_model: {quad_bem.aero_model}")
print(f"  use_bem:    {quad_bem.use_bem}")
print(f"  num_rotors: {quad_bem.num_rotors}")

print("\n--- Instantiating Multirotor with lumped params ---")
state_lump = make_initial_state(params_lumped)
quad_lump = Multirotor(params_lumped, initial_state=state_lump,
                       control_abstraction='cmd_motor_speeds', aero=True)
print(f"  aero_model: {quad_lump.aero_model}")
print(f"  use_bem:    {quad_lump.use_bem}")

# ---- Compare compute_body_wrench ----
print("\n--- Body wrench comparison (hover, no wind) ---")
omega_hover_bem = state_bem['rotor_speeds']
omega_hover_lump = state_lump['rotor_speeds']

F_bem, M_bem = quad_bem.compute_body_wrench(np.zeros(3), omega_hover_bem, np.zeros(3))
F_lump, M_lump = quad_lump.compute_body_wrench(np.zeros(3), omega_hover_lump, np.zeros(3))

print(f"  BEM   F_body = [{F_bem[0]:.4f}, {F_bem[1]:.4f}, {F_bem[2]:.4f}] N")
print(f"  Lump  F_body = [{F_lump[0]:.4f}, {F_lump[1]:.4f}, {F_lump[2]:.4f}] N")
print(f"  BEM   M_body = [{M_bem[0]:.6f}, {M_bem[1]:.6f}, {M_bem[2]:.6f}] Nm")
print(f"  Lump  M_body = [{M_lump[0]:.6f}, {M_lump[1]:.6f}, {M_lump[2]:.6f}] Nm")

# ---- Compare with 5 m/s headwind ----
print("\n--- Body wrench comparison (hover, 5 m/s headwind in body x) ---")
V_wind = np.array([5.0, 0.0, 0.0])

F_bem_w, M_bem_w = quad_bem.compute_body_wrench(np.zeros(3), omega_hover_bem, V_wind)
F_lump_w, M_lump_w = quad_lump.compute_body_wrench(np.zeros(3), omega_hover_lump, V_wind)

print(f"  BEM   F_body = [{F_bem_w[0]:.4f}, {F_bem_w[1]:.4f}, {F_bem_w[2]:.4f}] N")
print(f"  Lump  F_body = [{F_lump_w[0]:.4f}, {F_lump_w[1]:.4f}, {F_lump_w[2]:.4f}] N")
print(f"  BEM  dF_wind = [{F_bem_w[0]-F_bem[0]:.4f}, {F_bem_w[1]-F_bem[1]:.4f}, {F_bem_w[2]-F_bem[2]:.4f}] N")
print(f"  Lump dF_wind = [{F_lump_w[0]-F_lump[0]:.4f}, {F_lump_w[1]-F_lump[1]:.4f}, {F_lump_w[2]-F_lump[2]:.4f}] N")

# ---- Single time step ----
print("\n--- Single time step (dt=0.01s, no wind) ---")
dt = 0.01
control = {'cmd_motor_speeds': omega_hover_bem}
state_next = quad_bem.step(state_bem, control, dt)
print(f"  x_next = {state_next['x']}")
print(f"  v_next = {state_next['v']}")
print(f"  q_next = {state_next['q']}")

print("\n" + "=" * 60)
print("Integration test completed successfully.")
print("=" * 60)
