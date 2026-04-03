"""
Blade Element Momentum (BEM) rotor aerodynamics model.

Implements the radial inflow model from:
    Davoudi et al., "Quad-rotor Flight Simulation in Realistic Atmospheric Conditions,"
    arXiv:1902.01465, 2019.

Key features vs. the lumped-parameter model (T = k_eta * omega^2):
    1. Thrust depends on inflow velocity (wind-sensitive)
    2. Torque is decomposed into induced, profile, parasite, and climb power
    3. Lumped drag couples airframe drag with rotor thrust
    4. Prandtl tip/root loss correction is included
    5. Blade geometry (chord, twist, airfoil Cl_alpha) is explicitly modeled

All quantities are in SI units unless noted otherwise.
"""

import numpy as np
from scipy.optimize import brentq


class BEMRotor:
    """
    Single-rotor BEM model.  Call ``compute_thrust_torque(omega, V_rel_body)``
    to get the thrust (scalar, along +z_body) and aerodynamic torque (scalar,
    about z_body) for one rotor.

    Parameters (passed via a dict ``bem_params``):
        R           : rotor radius [m]
        R_min       : hub cutout radius [m]  (typically 0.1*R)
        N_b         : number of blades
        c_bar       : mean chord [m]  (constant chord assumed if c_func is None)
        c_func      : callable c(r) returning chord at radial station r [m]
        theta_tip   : blade pitch at tip [rad]
        theta_root  : blade pitch at root [rad]
        Cl_alpha    : 2-D lift curve slope [1/rad]
        alpha_L0    : zero-lift angle of attack [rad]
        Cd0         : profile drag coefficient
        rho         : air density [kg/m^3]
        c_d_bar     : lumped drag coefficient  (Davoudi Eq. 12, default 0.04)
        f_over_A    : equivalent flat-plate area ratio (default 0.005)
        n_elements  : number of radial integration stations (default 30)
    """

    def __init__(self, bem_params):
        self.R       = bem_params['R']
        self.R_min   = bem_params.get('R_min', 0.1 * self.R)
        self.N_b     = bem_params['N_b']
        self.rho     = bem_params.get('rho', 1.225)

        # Blade geometry
        self.c_bar       = bem_params['c_bar']
        self.c_func      = bem_params.get('c_func', None)   # callable or None
        self.theta_tip   = bem_params['theta_tip']           # rad
        self.theta_root  = bem_params['theta_root']          # rad

        # Airfoil
        self.Cl_alpha = bem_params['Cl_alpha']
        self.alpha_L0 = bem_params.get('alpha_L0', np.radians(4.0))
        self.Cd0      = bem_params.get('Cd0', 0.02)

        # Drag
        self.c_d_bar  = bem_params.get('c_d_bar', 0.04)
        self.f_over_A = bem_params.get('f_over_A', 0.005)

        # Calibration scale factor:  T_calibrated = thrust_scale * T_bem
        # Use this to match BEM output to experimental hover data.
        # Set to 1.0 for pure BEM (no calibration).
        self.thrust_scale = bem_params.get('thrust_scale', 1.0)

        # Integration
        self.n_elements = bem_params.get('n_elements', 30)

        # Pre-compute non-dimensional radial stations (normalised by R)
        self._r_nd = np.linspace(self.R_min / self.R, 1.0, self.n_elements)
        self._dr_nd = self._r_nd[1] - self._r_nd[0]

        # Blade solidity  sigma = N_b * c_bar / (pi * R)
        self.sigma = self.N_b * self.c_bar / (np.pi * self.R)

        # Disk area
        self.A_disk = np.pi * self.R ** 2

    # ------------------------------------------------------------------
    #  Blade geometry helpers
    # ------------------------------------------------------------------
    def _theta(self, r_nd):
        """Linear twist distribution: pitch angle at non-dimensional station r/R."""
        return self.theta_root + (self.theta_tip - self.theta_root) * r_nd

    def _chord(self, r_nd):
        """Chord at non-dimensional station r/R."""
        if self.c_func is not None:
            return self.c_func(r_nd * self.R)
        return self.c_bar

    # ------------------------------------------------------------------
    #  Prandtl tip / root loss
    # ------------------------------------------------------------------
    def _prandtl_F(self, r_nd, theta):
        """Prandtl tip-loss and root-loss correction factor (Davoudi Eq. 6)."""
        eps = 1e-8
        r_nd_safe = np.clip(r_nd, eps, 1.0 - eps)
        theta_safe = np.clip(np.abs(theta), eps, None)

        f_tip  = self.N_b / 2.0 * (1.0 - r_nd_safe) / (r_nd_safe * theta_safe)
        f_root = self.N_b / 2.0 * r_nd_safe / ((1.0 - r_nd_safe) * theta_safe)

        F_tip  = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_tip),  -1, 1))
        F_root = 2.0 / np.pi * np.arccos(np.clip(np.exp(-f_root), -1, 1))

        return np.clip(F_tip * F_root, eps, 1.0)

    # ------------------------------------------------------------------
    #  Inflow ratio  (Davoudi Eq. 5 — closed-form)
    # ------------------------------------------------------------------
    def _lambda(self, r_nd, theta, lambda_c, F):
        """
        Radial inflow ratio (Davoudi Eq. 5).

        lambda(r) = sqrt( (s*Cla/(16F) - lc/2)^2 + s*Cla*theta*r/(8F) )
                    - ( s*Cla/(16F) - lc/2 )
        """
        a = self.sigma * self.Cl_alpha / (16.0 * F) - lambda_c / 2.0
        b = self.sigma * self.Cl_alpha / (8.0 * F) * theta * r_nd
        return np.sqrt(a ** 2 + b) - a

    # ------------------------------------------------------------------
    #  Core: compute thrust and torque for a single rotor
    # ------------------------------------------------------------------
    def compute_thrust_torque(self, omega, V_rel_body):
        """
        Compute rotor thrust and aerodynamic torque via BEM.

        Parameters
        ----------
        omega : float
            Rotor angular speed [rad/s].  Must be > 0.
        V_rel_body : array (3,)
            Relative airspeed of the rotor hub expressed in the **body frame**,
            defined as  R^T (v_vehicle - v_wind)  evaluated at the rotor hub.
            Convention: +z is the thrust direction (downward for NED).

        Returns
        -------
        T : float
            Thrust magnitude [N] (positive along +z_body).
        Q : float
            Aerodynamic torque magnitude [N·m] (positive = resists rotation).
        D_body : array (3,)
            Lumped aerodynamic drag force in body frame [N].
        """
        omega = np.abs(omega)
        if omega < 1.0:
            return 0.0, 0.0, np.zeros(3)

        V_tip = omega * self.R

        # Climb inflow ratio  (Davoudi Eq. 7)
        # Note: V_rel_body[2] > 0 means air flows INTO the rotor from below (climb)
        # Davoudi convention: positive z is downward, so lambda_c = -V_relBz / V_tip
        lambda_c = -V_rel_body[2] / V_tip

        # Advance ratio  (Davoudi p.18)
        mu = np.sqrt(V_rel_body[0] ** 2 + V_rel_body[1] ** 2) / V_tip

        # ----- Blade-element integration for thrust -----
        r_nd = self._r_nd                   # non-dimensional radial stations
        theta = self._theta(r_nd)           # local blade pitch
        F = self._prandtl_F(r_nd, theta)    # Prandtl correction
        lam = self._lambda(r_nd, theta, lambda_c, F)  # inflow ratio

        # Effective angle of attack at each station
        # Inflow angle  Phi = lambda / r  (small angle approx for low inflow)
        Phi = lam / np.clip(r_nd, 1e-8, None)
        alpha_eff = theta + self.alpha_L0 - Phi
        Cl = self.Cl_alpha * alpha_eff

        # Local velocity squared at each station
        #   V_local^2 = (r*omega)^2 + (lambda*V_tip)^2
        r_dim = r_nd * self.R
        V_local_sq = (r_dim * omega) ** 2 + (lam * V_tip) ** 2

        # Chord at each station
        c = np.array([self._chord(rr) for rr in r_nd])

        # Sectional thrust (integrate from hub to tip)
        dT = 0.5 * self.rho * Cl * V_local_sq * c * self.R * self._dr_nd * self.N_b
        T = np.sum(dT)
        T = max(T, 0.0)

        # Apply calibration scale factor
        T *= self.thrust_scale

        # ----- Torque via power decomposition  (Davoudi Eq. 10) -----
        C_T = T / (self.rho * self.A_disk * V_tip ** 2)

        # Hover inflow ratio (for induced power calculation)
        lambda_0_sq = lam.mean() ** 2    # approximate
        denom = np.sqrt(max(lambda_0_sq + mu ** 2, 1e-12))

        C_P_induced  = 1.15 * C_T ** 2 / (2.0 * denom)
        C_P_profile  = self.sigma * self.Cd0 / 8.0 * (1.0 + 4.6 * mu ** 2)
        C_P_parasite = 0.125 * self.f_over_A * mu ** 3
        C_P_climb    = C_T * lambda_c

        C_P = C_P_induced + C_P_profile + C_P_parasite + C_P_climb
        Q = C_P * self.rho * np.pi * self.R ** 5 * omega ** 2

        # ----- Lumped drag  (Davoudi Eq. 12) -----
        # D = -[[c_d,0,0],[0,c_d,0],[0,0,0]] * T * V_rel_body
        D_body = np.zeros(3)
        D_body[0] = -self.c_d_bar * T * V_rel_body[0]
        D_body[1] = -self.c_d_bar * T * V_rel_body[1]
        # z-component is zero: axial aero already captured in BEM thrust

        return T, Q, D_body

    # ------------------------------------------------------------------
    #  Inverse: given desired thrust, find omega  (for control allocation)
    # ------------------------------------------------------------------
    def omega_from_thrust(self, T_desired, V_rel_body, omega_guess=None,
                          omega_min=10.0, omega_max=3000.0):
        """
        Solve the inverse problem: find omega such that thrust(omega) = T_desired.

        Uses Brent's method on the scalar equation  T(omega) - T_desired = 0.
        """
        if T_desired <= 0:
            return 0.0

        def residual(omega):
            T, _, _ = self.compute_thrust_torque(omega, V_rel_body)
            return T - T_desired

        try:
            omega_sol = brentq(residual, omega_min, omega_max, xtol=1e-3, maxiter=50)
        except ValueError:
            # If the bracket doesn't contain a root, fall back to the static model
            # T ≈ k * omega^2  where k = rho * A * C_T_hover
            k_approx = self.rho * self.A_disk * self.sigma * self.Cl_alpha * \
                       self._theta(0.75) / 16.0   # rough hover estimate
            omega_sol = np.sqrt(max(T_desired / max(k_approx, 1e-10), 0.0))

        return omega_sol
