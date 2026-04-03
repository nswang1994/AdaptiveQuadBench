import numpy as np
from numpy.linalg import inv, norm
import scipy.integrate
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.bem_rotor import BEMRotor

import time

"""
Multirotor models
"""

def quat_dot(quat, omega):
    """
    Parameters:
        quat, [i,j,k,w]
        omega, angular velocity of body in body axes

    Returns
        duat_dot, [i,j,k,w]

    """
    # Adapted from "Quaternions And Dynamics" by Basile Graf.
    (q0, q1, q2, q3) = (quat[0], quat[1], quat[2], quat[3])
    G = np.array([[ q3,  q2, -q1, -q0],
                  [-q2,  q3,  q0, -q1],
                  [ q1, -q0,  q3, -q2]])
    quat_dot = 0.5 * G.T @ omega
    # Augment to maintain unit quaternion.
    quat_err = np.sum(quat**2) - 1
    quat_err_grad = 2 * quat
    quat_dot = quat_dot - quat_err * quat_err_grad
    return quat_dot

class Multirotor(object):
    """
    Multirotor forward dynamics model. 

    states: [position, velocity, attitude, body rates, wind, rotor speeds]

    Parameters:
        quad_params: a dictionary containing relevant physical parameters for the multirotor. 
        initial_state: the initial state of the vehicle. 
        control_abstraction: the appropriate control abstraction that is used by the controller, options are...
                                'cmd_motor_speeds': the controller directly commands motor speeds. 
                                'cmd_motor_thrusts': the controller commands forces for each rotor.
                                'cmd_ctbr': the controller commands a collective thrsut and body rates. 
                                'cmd_ctbm': the controller commands a collective thrust and moments on the x/y/z body axes
                                'cmd_ctatt': the controller commands a collective thrust and attitude (as a quaternion).
                                'cmd_vel': the controller commands a velocity vector in the world frame. 
                                'cmd_acc': the controller commands a mass normalized thrust vector (acceleration) in the world frame.
        aero: boolean, determines whether or not aerodynamic drag forces are computed. 
    """
    def __init__(self, quad_params, initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])},
                       control_abstraction='cmd_motor_speeds',
                       aero = True,
                       integrator = 'rk45',
                ):
        """
        Initialize quadrotor physical parameters.
        """

        # Inertial parameters
        self.mass            = quad_params['mass'] # kg
        self.Ixx             = quad_params['Ixx']  # kg*m^2
        self.Iyy             = quad_params['Iyy']  # kg*m^2
        self.Izz             = quad_params['Izz']  # kg*m^2
        self.Ixy             = quad_params['Ixy']  # kg*m^2
        self.Ixz             = quad_params['Ixz']  # kg*m^2
        self.Iyz             = quad_params['Iyz']  # kg*m^2
        self.arm_length      = quad_params['arm_length']  # meters

        if 'com' in quad_params:
            self.com = quad_params['com']
        else:
            self.com = np.array([0.0, 0.0, 0.0])

        # Payload parameters
        self.payload_mass = 0 # kg
        self.payload_position = np.array([0.0, 0.0, 0.0]) # m

        # Frame parameters
        self.c_Dx            = quad_params['c_Dx']  # drag coeff, N/(m/s)**2
        self.c_Dy            = quad_params['c_Dy']  # drag coeff, N/(m/s)**2
        self.c_Dz            = quad_params['c_Dz']  # drag coeff, N/(m/s)**2

        self.num_rotors      = quad_params['num_rotors']
        self.rotor_pos       = quad_params['rotor_pos']

        self.rotor_dir       = quad_params['rotor_directions']

        self.extract_geometry()

        # Rotor parameters    
        self.rotor_speed_min = quad_params['rotor_speed_min'] # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max'] # rad/s

        self.k_eta           = quad_params['k_eta']     # thrust coeff, N/(rad/s)**2
        self.k_m             = quad_params['k_m']       # yaw moment coeff, Nm/(rad/s)**2
        self.k_d             = quad_params['k_d']       # rotor drag coeff, N/(m/s)
        self.k_z             = quad_params['k_z']       # induced inflow coeff N/(m/s)
        self.k_flap          = quad_params['k_flap']    # Flapping moment coefficient Nm/(m/s)

        # Motor parameters
        self.tau_m           = quad_params['tau_m']     # motor reponse time, seconds
        self.motor_noise     = quad_params['motor_noise_std'] # noise added to the actual motor speed, rad/s / sqrt(Hz)

        # Additional constants.
        self.inertia = np.array([[self.Ixx, self.Ixy, self.Ixz],
                                 [self.Ixy, self.Iyy, self.Iyz],
                                 [self.Ixz, self.Iyz, self.Izz]])
        self.rotor_drag_matrix = np.array([[self.k_d,   0,                 0],
                                           [0,          self.k_d,          0],
                                           [0,          0,          self.k_z]])
        self.drag_matrix = np.array([[self.c_Dx,    0,          0],
                                     [0,            self.c_Dy,  0],
                                     [0,            0,          self.c_Dz]])
        self.aero_model = 'rotorpy'
        
        # Check if cd1_x exists in quad_params, if not set all related values to 0
        if 'cd1_x' in quad_params and quad_params['cd1_x'] is not None:
            self.aero_model = 'other'
            self.cdz_h = quad_params.get('cdz_h', 0)
            self.cd1x = quad_params['cd1_x']
            self.cd1y = quad_params['cd1_y']
            self.cd1z = quad_params['cd1_z']
            self.drag_matrix = np.array([[self.cd1x,    0,          0],
                                         [0,            self.cd1y,  0],
                                         [0,            0,          self.cd1z]])
        else:
            self.cdz_h = 0
            self.cd1x = 0
            self.cd1y = 0
            self.cd1z = 0
        
        # rotor efficiency
        if 'rotor_efficiency' in quad_params:
            self.rotor_efficiency = quad_params['rotor_efficiency']
        else:
            self.rotor_efficiency = np.ones(self.num_rotors)

        self.g = 9.81 # m/s^2

        self.inv_inertia = inv(self.inertia)
        self.weight = np.array([0, 0, -self.mass*self.g])

        # Control allocation
        k = self.k_m/self.k_eta  # Ratio of torque to thrust coefficient. 

        # Below is an automated generation of the control allocator matrix. It assumes that all thrust vectors are aligned
        # with the z axis.
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),np.hstack([np.cross(self.rotor_pos[key],np.array([0,0,1])).reshape(-1,1)[0:2] for key in self.rotor_pos]), (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

        # Set the initial state
        self.initial_state = initial_state

        self.control_abstraction = control_abstraction

        self.k_w = 1                # The body rate P gain        (for cmd_ctbr)
        self.k_v = 10               # The *world* velocity P gain (for cmd_vel)
        self.kp_att = 544           # The attitude P gain (for cmd_vel, cmd_acc, and cmd_ctatt)
        self.kd_att = 46.64         # The attitude D gain (for cmd_vel, cmd_acc, and cmd_ctatt)

        self.aero = aero

        # BEM rotor model (Davoudi et al., arXiv:1902.01465)
        self.use_bem = quad_params.get('use_bem', False)
        if self.use_bem:
            bem_params = quad_params['bem_params']
            self.bem_rotors = [BEMRotor(bem_params) for _ in range(self.num_rotors)]
            self.aero_model = 'bem'

        # Integrator selection: 'rk45' (default) or 'lgvi' (Lie group variational)
        self.integrator = integrator.lower()

    def extract_geometry(self):
        """
        Extracts the geometry in self.rotors for efficient use later on in the computation of 
        wrenches acting on the rigid body.
        The rotor_geometry is an array of length (n,3), where n is the number of rotors. 
        Each row corresponds to the position vector of the rotor relative to the CoM. 
        """
        
        self.rotor_geometry = np.array([]).reshape(0,3)
        for rotor in self.rotor_pos:
            # Adjust rotor position relative to the COM
            r = self.rotor_pos[rotor] - self.com
            self.rotor_geometry = np.vstack([self.rotor_geometry, r])
        
        return

    def statedot(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant cmd_rotor_speeds for time t_step.
        """

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)

        # The true motor speeds can not fall below min and max speeds.
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max) 

        # Form autonomous ODE for constant inputs and integrate one time step.
        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds,state['ext_force'],state['ext_torque'])
        s = Multirotor._pack_state(state)
        
        s_dot = s_dot_fn(0, s)
        v_dot = s_dot[3:6]
        w_dot = s_dot[10:13]

        state_dot = {'vdot': v_dot,'wdot': w_dot}
        return state_dot 


    def step(self, state, control, t_step):
        """
        Integrate dynamics forward from state given constant control for time t_step.
        Dispatches to RK45 (default) or LGVI based on self.integrator.
        """
        if self.integrator == 'lgvi':
            return self._step_lgvi(state, control, t_step)
        else:
            return self._step_rk45(state, control, t_step)

    def _step_rk45(self, state, control, t_step):
        """Original RK45 integrator on flat state vector + quaternion renormalization."""

        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        def s_dot_fn(t, s):
            return self._s_dot_fn(t, s, cmd_rotor_speeds, state['ext_force'], state['ext_torque'])
        s = Multirotor._pack_state(state)

        sol = scipy.integrate.solve_ivp(s_dot_fn, (0, t_step), s, first_step=t_step)
        s = sol['y'][:,-1]

        state_unpacked = Multirotor._unpack_state(s)
        state_unpacked['q'] = state_unpacked['q'] / norm(state_unpacked['q'])

        state_unpacked['rotor_speeds'] += np.random.normal(scale=np.abs(self.motor_noise), size=(self.num_rotors,))
        state_unpacked['rotor_speeds'] = np.clip(state_unpacked['rotor_speeds'], self.rotor_speed_min, self.rotor_speed_max)
        state_unpacked['ext_force'] = state['ext_force']
        state_unpacked['ext_torque'] = state['ext_torque']

        return state_unpacked

    # -----------------------------------------------------------------
    #  Lie Group Variational Integrator (LGVI) on SE(3)
    #
    #  Rigid-body rotation on SO(3) via the discrete Euler-Poincaré
    #  equation with the Cayley map, solved by Newton iteration.
    #  Translation on R^3 via Störmer-Verlet (velocity Verlet).
    #  Rotor dynamics: forward Euler (decoupled).
    #
    #  Ref: T. Lee, M. Leok, N.H. McClamroch,
    #       "Lie group variational integrators for the full body
    #        problem in orbital mechanics," CMAME, 2007.
    #       T. Lee, M. Leok, N.H. McClamroch,
    #       "Global Formulations of Lagrangian and Hamiltonian Dynamics
    #        on Manifolds," Springer, 2018, Chapter 9.
    #
    #  Key equation (implicit, for F_k ∈ SO(3)):
    #
    #      F_k J_d − J_d F_k^T  =  h · hat(Π_k + (h/2) M_k)
    #
    #  where J_d = (tr(J)/2)I − J  is the nonstandard inertia,
    #  F_k = cay(f) is the relative rotation (Cayley map),
    #  and Π_k = J Ω_k is the body angular momentum.
    #
    #  Properties:
    #   • R_{k+1} = R_k F_k  stays exactly on SO(3)
    #   • Symplectic (variational) — near-exact energy over long time
    #   • Preserves angular momentum map (torque-free case)
    # -----------------------------------------------------------------

    @staticmethod
    def _so3_hat(v):
        """R^3 -> so(3) skew-symmetric."""
        return np.array([[    0, -v[2],  v[1]],
                         [ v[2],     0, -v[0]],
                         [-v[1],  v[0],     0]])

    @staticmethod
    def _so3_vee(S):
        """so(3) -> R^3: extract vector from skew-symmetric matrix."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])

    @staticmethod
    def _cayley(f):
        """Cayley map: R^3 -> SO(3).  cay(f) = (I + [f]x/2)(I - [f]x/2)^{-1}."""
        f_hat = Multirotor._so3_hat(f)
        return np.linalg.solve((np.eye(3) - 0.5 * f_hat).T,
                               (np.eye(3) + 0.5 * f_hat).T).T

    def _step_lgvi(self, state, control, t_step):
        """
        Lee-Leok-McClamroch LGVI for rigid-body dynamics on SE(3).

        Rotation lives exactly on SO(3) via the Cayley map — no quaternion,
        no renormalization.  The implicit discrete Euler-Poincaré equation
        is solved by Newton iteration (typically 3-5 iterations to eps).
        """
        h = t_step
        J = self.inertia
        J_inv = self.inv_inertia

        # Nonstandard inertia:  J_d = (tr(J)/2) I - J
        J_d = 0.5 * np.trace(J) * np.eye(3) - J

        # --- Unpack current state ---
        x_k = state['x'].copy()
        v_k = state['v'].copy()
        R_k = Rotation.from_quat(state['q']).as_matrix()
        w_k = state['w'].copy()
        rotor_speeds_k = state['rotor_speeds'].copy()
        wind_k = state['wind'].copy()

        # --- Commanded rotor speeds ---
        cmd_rotor_speeds = self.get_cmd_motor_speeds(state, control)
        cmd_rotor_speeds = np.clip(cmd_rotor_speeds, self.rotor_speed_min, self.rotor_speed_max)

        # --- Body wrench at current state ---
        body_airspeed = R_k.T @ (v_k - wind_k)
        (F_body, M_body) = self.compute_body_wrench(w_k, rotor_speeds_k, body_airspeed)

        ext_force  = state['ext_force']
        ext_torque = state['ext_torque']
        M_total = M_body + ext_torque

        # ============================================================
        #  SO(3):  Lee's LGVI — implicit discrete Euler-Poincaré
        # ============================================================
        Pi_k = J @ w_k                       # body angular momentum
        target = Pi_k + (h / 2.0) * M_total  # RHS vector

        # Residual:  r(f) = vee(F J_d - J_d F^T) - h·target
        def residual(f):
            F = Multirotor._cayley(f)
            S = F @ J_d - J_d @ F.T           # skew-symmetric 3x3
            return Multirotor._so3_vee(S) - h * target

        # Newton iteration  (initial guess: f = h Ω_k)
        f = h * w_k
        for _ in range(10):
            r = residual(f)
            if norm(r) < 1e-12:
                break
            # Numerical Jacobian (3 × 3)
            eps = 1e-7
            Dr = np.empty((3, 3))
            for j in range(3):
                f_p = f.copy()
                f_p[j] += eps
                Dr[:, j] = (residual(f_p) - r) / eps
            f -= np.linalg.solve(Dr, r)

        F_k = Multirotor._cayley(f)

        # Rotation update  (exact on SO(3))
        R_kp1 = R_k @ F_k

        # Angular momentum update (Lee eq. 3.3 / 9.13):
        #   Π_{k+1} = F_k^T (Π_k + (h/2) M_k) + (h/2) M_{k+1}
        # With M_{k+1} ≈ M_k  (constant over step):
        Pi_kp1 = F_k.T @ target + (h / 2.0) * M_total
        w_kp1  = J_inv @ Pi_kp1

        # ============================================================
        #  R^3:  Störmer-Verlet for translation
        # ============================================================
        F_world = R_k @ F_body + self.weight + ext_force
        a_k = F_world / self.mass

        v_half = v_k + (h / 2.0) * a_k
        x_kp1  = x_k + h * v_half

        # Recompute body wrench at new R for the second half-kick
        body_airspeed_new = R_kp1.T @ (v_half - wind_k)
        (F_body_new, _) = self.compute_body_wrench(w_kp1, rotor_speeds_k, body_airspeed_new)
        F_world_new = R_kp1 @ F_body_new + self.weight + ext_force
        a_kp1 = F_world_new / self.mass

        v_kp1 = v_half + (h / 2.0) * a_kp1

        # ============================================================
        #  Rotor dynamics: forward Euler (decoupled)
        # ============================================================
        rotor_accel = (1.0 / self.tau_m) * (cmd_rotor_speeds - rotor_speeds_k)
        rotor_speeds_kp1 = rotor_speeds_k + rotor_accel * h

        # --- Pack output ---
        q_kp1 = Rotation.from_matrix(R_kp1).as_quat()

        state_new = {
            'x': x_kp1,
            'v': v_kp1,
            'q': q_kp1,
            'w': w_kp1,
            'wind': wind_k,
            'rotor_speeds': rotor_speeds_kp1,
        }

        state_new['rotor_speeds'] += np.random.normal(scale=np.abs(self.motor_noise), size=(self.num_rotors,))
        state_new['rotor_speeds'] = np.clip(state_new['rotor_speeds'], self.rotor_speed_min, self.rotor_speed_max)
        state_new['ext_force']  = ext_force
        state_new['ext_torque'] = ext_torque

        return state_new

    def _s_dot_fn(self, t, s, cmd_rotor_speeds, ext_force, ext_torque):
        """
        Compute derivative of state for quadrotor given fixed control inputs as
        an autonomous ODE.
        """

        state = Multirotor._unpack_state(s)

        rotor_speeds = state['rotor_speeds']
        inertial_velocity = state['v']
        wind_velocity = state['wind']
        R = Rotation.from_quat(state['q']).as_matrix()

        # Rotor speed derivative
        rotor_accel = (1/self.tau_m)*(cmd_rotor_speeds - rotor_speeds)

        # Position derivative.
        x_dot = state['v']

        # Orientation derivative.
        q_dot = quat_dot(state['q'], state['w'])

        # Compute airspeed vector in the body frame
        body_airspeed_vector = R.T@(inertial_velocity - wind_velocity)

        # Compute total wrench in the body frame based on the current rotor speeds and their location w.r.t. CoM
        (FtotB, MtotB) = self.compute_body_wrench(state['w'], rotor_speeds, body_airspeed_vector)

        # Rotate the force from the body frame to the inertial frame
        Ftot = R@FtotB

        # Velocity derivative.
        v_dot = (self.weight + Ftot + ext_force) / self.mass

        # Angular velocity derivative.
        w = state['w']
        w_hat = Multirotor.hat_map(w)
        w_dot = self.inv_inertia @ (MtotB + ext_torque - w_hat @ (self.inertia @ w))

        # NOTE: the wind dynamics are currently handled in the wind_profile object. 
        # The line below doesn't do anything, as the wind state is assigned elsewhere. 
        wind_dot = np.zeros(3,)

        # Pack into vector of derivatives.
        s_dot = np.zeros((16+self.num_rotors,))
        s_dot[0:3]   = x_dot
        s_dot[3:6]   = v_dot
        s_dot[6:10]  = q_dot
        s_dot[10:13] = w_dot
        s_dot[13:16] = wind_dot
        s_dot[16:]   = rotor_accel

        return s_dot

    def compute_body_wrench(self, body_rates, rotor_speeds, body_airspeed_vector):
        """
        Computes the wrench acting on the rigid body based on the rotor speeds for thrust and airspeed 
        for aerodynamic forces. 
        The airspeed is represented in the body frame.
        The net force Ftot is represented in the body frame. 
        The net moment Mtot is represented in the body frame. 
        """

        # Get the local airspeeds for each rotor
        local_airspeeds = body_airspeed_vector[:, np.newaxis] + Multirotor.hat_map(body_rates)@(self.rotor_geometry.T)

        # ---- BEM branch (Davoudi et al. arXiv:1902.01465) ----
        if self.use_bem and self.aero:
            T = np.zeros((3, self.num_rotors))
            H = np.zeros((3, self.num_rotors))  # lumped drag per rotor
            M_yaw = np.zeros((3, self.num_rotors))

            for i in range(self.num_rotors):
                V_hub_i = local_airspeeds[:, i]
                eff_i = self.rotor_efficiency[i]

                # BEM: thrust, torque, and lumped drag
                Ti, Qi, Di = self.bem_rotors[i].compute_thrust_torque(
                    rotor_speeds[i], V_hub_i)

                # Scale thrust and torque by rotor efficiency
                Ti *= eff_i
                Qi *= eff_i

                # Thrust along +z body
                T[:, i] = np.array([0.0, 0.0, Ti])

                # Lumped drag at hub (Davoudi Eq. 12)
                H[:, i] = Di * eff_i

                # Yaw torque (reactive, opposes rotation)
                M_yaw[:, i] = np.array([0.0, 0.0, self.rotor_dir[i] * Qi])

            D = np.zeros(3)     # frame parasitic drag already in lumped Di
            M_flap = np.zeros((3, self.num_rotors))  # flapping captured in BEM

            # Moments from thrust + drag forces at rotor hubs
            M_force = -np.einsum('ijk, ik->j', Multirotor.hat_map(self.rotor_geometry), T + H)

            FtotB = np.sum(T + H, axis=1) + D
            MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

            return (FtotB, MtotB)

        # ---- Original lumped-parameter branch ----
        # Scale k_eta for each rotor based on rotor efficiency
        k_eta_scaled = self.k_eta * self.rotor_efficiency
        # Scale k_m for each rotor based on rotor efficiency
        k_m_scaled = self.k_m * self.rotor_efficiency

        # Compute the thrust of each rotor with scaled k_eta
        T = np.array([0, 0, 1])[:, np.newaxis] * (k_eta_scaled * rotor_speeds**2)

        # Add in aero wrenches (if applicable)
        if self.aero:
            if self.aero_model == 'rotorpy':
                # Parasitic drag force acting at the CoM
                D = -Multirotor._norm(body_airspeed_vector)*self.drag_matrix@body_airspeed_vector
            else:
                # Parasitic drag force acting at the CoM
                D = -self.drag_matrix@body_airspeed_vector
                D[-1] += self.cdz_h*(body_airspeed_vector[0]**2 + body_airspeed_vector[1]**2)
            # Rotor drag (aka H force) acting at each propeller hub - scale with rotor efficiency
            H = -(rotor_speeds)*(self.rotor_drag_matrix@local_airspeeds)
            # Pitching flapping moment acting at each propeller hub - scale with rotor efficiency
            M_flap = -self.k_flap*(rotor_speeds)*((Multirotor.hat_map(local_airspeeds.T).transpose(2, 0, 1))@np.array([0,0,1])).T
        else:
            D = np.zeros(3,)
            H = np.zeros((3,self.num_rotors))
            M_flap = np.zeros((3,self.num_rotors))

        # Compute the moments due to the rotor thrusts, rotor drag (if applicable), and rotor drag torques
        M_force = -np.einsum('ijk, ik->j', Multirotor.hat_map(self.rotor_geometry), T+H)
        # Use scaled k_m for yaw moment
        M_yaw = self.rotor_dir*(np.array([0, 0, 1])[:, np.newaxis] * (k_m_scaled * rotor_speeds**2))

        # Sum all elements to compute the total body wrench
        FtotB = np.sum(T + H, axis=1) + D
        MtotB = M_force + np.sum(M_yaw + M_flap, axis=1)

        return (FtotB, MtotB)

    def get_cmd_motor_speeds(self, state, control):
        """
        Computes the commanded motor speeds depending on the control abstraction.
        For higher level control abstractions, we have low-level controllers that will produce motor speeds based on the higher level commmand. 

        """

        if self.control_abstraction == 'cmd_motor_speeds':
            # The controller directly controls motor speeds, so command that. 
            return control['cmd_motor_speeds']

        elif self.control_abstraction == 'cmd_motor_thrusts':
            # The controller commands individual motor forces. 
            cmd_motor_speeds = control['cmd_motor_thrusts'] / self.k_eta                        # Convert to motor speeds from thrust coefficient. 
            return np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        elif self.control_abstraction == 'cmd_ctbm':
            # The controller commands collective thrust and moment on each axis. 
            cmd_thrust = control['cmd_thrust']
            cmd_moment = control['cmd_moment']  

        elif self.control_abstraction == 'cmd_ctbr':
            # The controller commands collective thrust and body rates on each axis. 

            cmd_thrust = control['cmd_thrust']

            # First compute the error between the desired body rates and the actual body rates given by state. 
            w_err = state['w'] - control['cmd_w']

            # Computed commanded moment based on the attitude error and body rate error
            wdot_cmd = -self.k_w*w_err
            cmd_moment = self.inertia@wdot_cmd

            # Now proceed with the cmd_ctbm formulation.

        elif self.control_abstraction == 'cmd_vel':
            # The controller commands a velocity vector. 
            
            # Get the error in the current velocity. 
            v_err = state['v'] - control['cmd_v']

            # Get desired acceleration based on P control of velocity error. 
            a_cmd = -self.k_v*v_err

            # Get desired force from this acceleration. 
            F_des = self.mass*(a_cmd + np.array([0, 0, self.g]))

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Follow rest of SE3 controller to compute cmd moment. 

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])

        elif self.control_abstraction == 'cmd_ctatt':
            # The controller commands the collective thrust and attitude.

            cmd_thrust = control['cmd_thrust']

            # Compute the shape error from the current attitude and the desired attitude. 
            R = Rotation.from_quat(state['q']).as_matrix()
            R_des = Rotation.from_quat(control['cmd_q']).as_matrix()

            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Compute command moment based on attitude error. 
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        
        elif self.control_abstraction == 'cmd_acc':
            # The controller commands an acceleration vector (or thrust vector). This is equivalent to F_des in the SE3 controller. 
            F_des = control['cmd_acc']*self.mass

            R = Rotation.from_quat(state['q']).as_matrix()
            b3 = R @ np.array([0, 0, 1])
            cmd_thrust = np.dot(F_des, b3)

            # Desired orientation to obtain force vector.
            b3_des = F_des/np.linalg.norm(F_des)
            c1_des = np.array([1, 0, 0])
            b2_des = np.cross(b3_des, c1_des)/np.linalg.norm(np.cross(b3_des, c1_des))
            b1_des = np.cross(b2_des, b3_des)
            R_des = np.stack([b1_des, b2_des, b3_des]).T

            # Orientation error.
            S_err = 0.5 * (R_des.T @ R - R.T @ R_des)
            att_err = np.array([-S_err[1,2], S_err[0,2], -S_err[0,1]])

            # Angular control; vector units of N*m.
            cmd_moment = self.inertia @ (-self.kp_att*att_err - self.kd_att*state['w']) + np.cross(state['w'], self.inertia@state['w'])
        else:
            raise ValueError("Invalid control abstraction selected. Options are: cmd_motor_speeds, cmd_motor_thrusts, cmd_ctbm, cmd_ctbr, cmd_ctatt, cmd_vel, cmd_acc")

        # Take the commanded thrust and body moments and convert them to motor speeds
        TM = np.concatenate(([cmd_thrust], cmd_moment))               # Concatenate thrust and moment into an array
        cmd_motor_forces = self.TM_to_f @ TM                                                # Convert to cmd_motor_forces from allocation matrix
        cmd_motor_speeds = cmd_motor_forces / self.k_eta                                    # Convert to motor speeds from thrust coefficient. 
        cmd_motor_speeds = np.sign(cmd_motor_speeds) * np.sqrt(np.abs(cmd_motor_speeds))

        return cmd_motor_speeds

    @classmethod
    def rotate_k(cls, q):
        """
        Rotate the unit vector k by quaternion q. This is the third column of
        the rotation matrix associated with a rotation by q.
        """
        return np.array([  2*(q[0]*q[2]+q[1]*q[3]),
                           2*(q[1]*q[2]-q[0]*q[3]),
                         1-2*(q[0]**2  +q[1]**2)    ])

    @classmethod
    def hat_map(cls, s):
        """
        Given vector s in R^3, return associate skew symmetric matrix S in R^3x3
        In the vectorized implementation, we assume that s is in the shape (N arrays, 3)
        """
        if len(s.shape) > 1:  # Vectorized implementation
            return np.array([[ np.zeros(s.shape[0]), -s[:,2],  s[:,1]],
                             [ s[:,2],     np.zeros(s.shape[0]), -s[:,0]],
                             [-s[:,1],  s[:,0],     np.zeros(s.shape[0])]])
        else:
            return np.array([[    0, -s[2],  s[1]],
                             [ s[2],     0, -s[0]],
                             [-s[1],  s[0],     0]])

    @classmethod
    def _pack_state(cls, state):
        """
        Convert a state dict to Quadrotor's private internal vector representation.
        """
        s = np.zeros((20,))   # FIXME: this shouldn't be hardcoded. Should vary with the number of rotors. 
        s[0:3]   = state['x']       # inertial position
        s[3:6]   = state['v']       # inertial velocity
        s[6:10]  = state['q']       # orientation
        s[10:13] = state['w']       # body rates
        s[13:16] = state['wind']    # wind vector
        s[16:]   = state['rotor_speeds']     # rotor speeds

        return s

    @classmethod
    def _norm(cls, v):
        """
        Given a vector v in R^3, return the 2 norm (length) of the vector
        """
        norm = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        return norm

    @classmethod
    def _unpack_state(cls, s):
        """
        Convert Quadrotor's private internal vector representation to a state dict.
        x = inertial position
        v = inertial velocity
        q = orientation
        w = body rates
        wind = wind vector
        rotor_speeds = rotor speeds
        """
        state = {'x':s[0:3], 'v':s[3:6], 'q':s[6:10], 'w':s[10:13], 'wind':s[13:16], 'rotor_speeds':s[16:]}
        return state

    def update_payload(self, payload_mass, payload_position, payload_inertia=None):
        """
        Updates the payload parameters.
        
        Parameters:
            payload_mass: Mass of the payload in kg
            payload_position: Position of the payload's COM in the body frame
            payload_inertia: Inertia tensor of the payload about its COM (optional)
        """

        # Update payload parameters
        self.payload_mass = payload_mass
        self.payload_position = payload_position
        self.payload_inertia = payload_inertia

    def attach_payload(self):
        """
        Updates the center of mass and inertia when a payload is attached.
        
        Parameters:
            payload_mass: Mass of the payload in kg
            payload_position: Position of the payload's COM in the body frame
            payload_inertia: Inertia tensor of the payload about its COM (optional)
        """
        if self.payload_mass == 0:
            return np.zeros(3)
        
        # Store original values
        original_mass = self.mass
        original_com = self.com.copy()
        original_inertia = self.inertia.copy()
        
        # Calculate new total mass
        total_mass = original_mass + self.payload_mass
        
        # Calculate new COM position using weighted average
        new_com = (original_mass * original_com + self.payload_mass * self.payload_position) / total_mass
        
        # Update mass and COM
        self.mass = total_mass
        self.com = new_com
        # Update inertia tensor
        
        # 1. Shift the original inertia tensor to the new COM
        r_original = original_com - new_com  # Vector from new COM to original COM
        r_squared = np.sum(r_original**2)
        r_outer = np.outer(r_original, r_original)
        
        # Apply parallel axis theorem to shift original inertia to new COM
        shifted_original_inertia = original_inertia - original_mass * (r_squared * np.eye(3) - r_outer)
        
        # 2. Add the payload's contribution to the inertia
        if self.payload_inertia is not None:
            # If payload inertia is provided, shift it to the new COM
            r_payload = self.payload_position - new_com  # Vector from new COM to payload COM
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Apply parallel axis theorem to shift payload inertia to new COM
            shifted_payload_inertia = self.payload_inertia + self.payload_mass * (r_squared * np.eye(3) - r_outer)
            
            # Add to get total inertia
            self.inertia = shifted_original_inertia + shifted_payload_inertia
        else:
            # If no payload inertia provided, assume point mass
            r_payload = self.payload_position - new_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Point mass inertia contribution
            payload_inertia_contribution = self.payload_mass * (r_squared * np.eye(3) - r_outer)
            
            # Add to get total inertia
            self.inertia = shifted_original_inertia + payload_inertia_contribution
        
        # Update inverse inertia
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # Re-extract geometry with updated COM
        self.extract_geometry()
        
        # Update weight vector
        self.weight = np.array([0, 0, -self.mass*self.g])
        
        # Update control allocation matrix if needed
        # This is necessary because rotor positions relative to COM have changed
        k = self.k_m/self.k_eta
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                 np.hstack([np.cross(self.rotor_pos[key] - self.com,np.array([0,0,1])).reshape(-1,1)[0:2] 
                                 for key in self.rotor_pos]), 
                                 (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
        
        return new_com - original_com  # Return the COM shift
    
    def detach_payload(self):
        """
        Detaches the payload from the vehicle and restores original properties.
        
        This method:
        1. Restores the original mass, COM, and inertia
        2. Updates all dependent properties (weight, control allocation, etc.)
        3. Resets payload parameters
        
        Returns:
            The COM shift vector (original_com - previous_com)
        """
        if self.payload_mass == 0:
            return np.zeros(3)  # No payload to detach
        
        # Store current COM for calculating shift
        previous_com = self.com.copy()
        
        # Store payload info for return value
        detached_payload_mass = self.payload_mass
        detached_payload_position = self.payload_position.copy()
        
        # Restore original mass
        original_mass = self.mass - self.payload_mass
        self.mass = original_mass
        
        # Recalculate COM without payload
        if original_mass > 0:
            # Calculate original COM
            self.com = (self.mass * self.com - self.payload_mass * self.payload_position) / original_mass
        
        # Recalculate inertia tensor
        # 1. Remove payload contribution from inertia
        if self.payload_inertia is not None:
            # If payload inertia was provided, remove its shifted contribution
            r_payload = self.payload_position - previous_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Remove shifted payload inertia
            shifted_payload_inertia = self.payload_inertia + self.payload_mass * (r_squared * np.eye(3) - r_outer)
            self.inertia = self.inertia - shifted_payload_inertia
        else:
            # If payload was a point mass, remove its contribution
            r_payload = self.payload_position - previous_com
            r_squared = np.sum(r_payload**2)
            r_outer = np.outer(r_payload, r_payload)
            
            # Remove point mass inertia contribution
            payload_inertia_contribution = self.payload_mass * (r_squared * np.eye(3) - r_outer)
            self.inertia = self.inertia - payload_inertia_contribution
        
        # 2. Shift the inertia tensor to the new COM
        r_shift = previous_com - self.com  # Vector from new COM to previous COM
        r_squared = np.sum(r_shift**2)
        r_outer = np.outer(r_shift, r_shift)
        
        # Apply parallel axis theorem to shift inertia to new COM
        self.inertia = self.inertia - self.mass * (r_squared * np.eye(3) - r_outer)
        
        # Update inverse inertia
        self.inv_inertia = np.linalg.inv(self.inertia)
        
        # Re-extract geometry with updated COM
        self.extract_geometry()
        
        # Update weight vector
        self.weight = np.array([0, 0, -self.mass*self.g])
        
        # Update control allocation matrix
        k = self.k_m/self.k_eta
        self.f_to_TM = np.vstack((np.ones((1,self.num_rotors)),
                                np.hstack([np.cross(self.rotor_pos[key] - self.com,np.array([0,0,1])).reshape(-1,1)[0:2] 
                                for key in self.rotor_pos]), 
                                (k * self.rotor_dir).reshape(1,-1)))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)
        
        # Reset payload parameters
        self.payload_mass = 0
        self.payload_position = np.zeros(3)
        self.payload_inertia = None
        
        # Return the COM shift
        return self.com - previous_com