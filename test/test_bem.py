"""
Test script for BEM rotor integration.

Compares:
  1. BEM thrust vs. lumped-parameter thrust at hover
  2. BEM thrust sensitivity to wind (axial & lateral)
  3. BEM torque decomposition
  4. Hover RPM consistency check with Davoudi Fig. 8

Run:  python -m test.test_bem   (from AdaptiveQuadBench root)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'rotorpy'))

import numpy as np
from rotorpy.vehicles.bem_rotor import BEMRotor

# ---------- Load BEM parameters ----------
from quad_param.quadrotor_bem import quad_params
bem_params = quad_params['bem_params']
bem = BEMRotor(bem_params)

mass = quad_params['mass']
g = 9.81
T_hover_per_rotor = mass * g / 4.0  # target thrust per rotor at hover

print("=" * 60)
print("BEM Rotor Model — Integration Test")
print("=" * 60)

# ---------- Test 1: Hover thrust ----------
print("\n--- Test 1: Hover thrust ---")
# Find omega for hover (no wind)
V_no_wind = np.array([0.0, 0.0, 0.0])

# Scan omega range to find hover
omegas = np.linspace(100, 2100, 200)
thrusts = []
for w in omegas:
    T, Q, D = bem.compute_thrust_torque(w, V_no_wind)
    thrusts.append(T)
thrusts = np.array(thrusts)

# Find hover omega by interpolation
idx = np.argmin(np.abs(thrusts - T_hover_per_rotor))
omega_hover = omegas[idx]
T_hover, Q_hover, _ = bem.compute_thrust_torque(omega_hover, V_no_wind)
rpm_hover = omega_hover * 60 / (2 * np.pi)

print(f"  Vehicle mass:       {mass:.3f} kg")
print(f"  Target T/rotor:     {T_hover_per_rotor:.4f} N")
print(f"  BEM T @ hover:      {T_hover:.4f} N")
print(f"  Hover omega:        {omega_hover:.1f} rad/s  ({rpm_hover:.0f} RPM)")
print(f"  Hover torque:       {Q_hover:.6f} Nm")

# Compare with lumped parameter
k_eta = quad_params['k_eta']
omega_hover_lumped = np.sqrt(T_hover_per_rotor / k_eta)
rpm_hover_lumped = omega_hover_lumped * 60 / (2 * np.pi)
print(f"\n  Lumped-param hover: {omega_hover_lumped:.1f} rad/s  ({rpm_hover_lumped:.0f} RPM)")
print(f"  RPM difference:     {abs(rpm_hover - rpm_hover_lumped):.0f} RPM ({abs(rpm_hover - rpm_hover_lumped)/rpm_hover*100:.1f}%)")

# ---------- Test 2: Wind sensitivity ----------
print("\n--- Test 2: Wind sensitivity (at hover omega) ---")
wind_speeds = [0, 2, 4, 6, 8]

print(f"  {'Wind (m/s)':>12} | {'BEM T (N)':>10} | {'Lumped T (N)':>12} | {'BEM Q (Nm)':>10} | {'Drag_x (N)':>10}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

for vw in wind_speeds:
    V_wind = np.array([vw, 0, 0])  # headwind in body x
    T_bem, Q_bem, D_bem = bem.compute_thrust_torque(omega_hover, V_wind)
    T_lumped = k_eta * omega_hover ** 2  # lumped: wind-insensitive!
    print(f"  {vw:>12.1f} | {T_bem:>10.4f} | {T_lumped:>12.4f} | {Q_bem:>10.6f} | {D_bem[0]:>10.4f}")

# ---------- Test 3: Axial wind (climb/descent) ----------
print("\n--- Test 3: Axial wind (climb vs descent at hover omega) ---")
for vz in [-4, -2, 0, 2, 4]:
    V_axial = np.array([0, 0, vz])
    T_bem, Q_bem, _ = bem.compute_thrust_torque(omega_hover, V_axial)
    print(f"  Vz = {vz:+.1f} m/s  ->  T = {T_bem:.4f} N,  Q = {Q_bem:.6f} Nm")

# ---------- Test 4: Inverse problem ----------
print("\n--- Test 4: Inverse problem (omega_from_thrust) ---")
for vw in [0, 4, 8]:
    V_wind = np.array([vw, 0, 0])
    omega_inv = bem.omega_from_thrust(T_hover_per_rotor, V_wind)
    T_check, _, _ = bem.compute_thrust_torque(omega_inv, V_wind)
    rpm_inv = omega_inv * 60 / (2 * np.pi)
    print(f"  Wind = {vw} m/s  ->  omega = {omega_inv:.1f} rad/s ({rpm_inv:.0f} RPM),  T_check = {T_check:.4f} N  (target: {T_hover_per_rotor:.4f} N)")

print("\n" + "=" * 60)
print("All tests completed.")
print("=" * 60)
