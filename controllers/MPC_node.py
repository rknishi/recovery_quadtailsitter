#!/usr/bin/env python3
import numpy as np
import yaml
import rclpy
import casadi as ca
from custom_msg.msg import FwOdometry, ThrustTorqueSetpoint, CustomReference
from common import quat_rotate, quat_conj, quat_mult, quat_to_rot, rot_to_quat, quat_conj_vectorized, quat_mult_vectorized, decompose_quaternion_vectorized, decompose_quaternion_error_vectorized, rotate_quat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node


class MPC_Flight_controller(Node):
    def __init__(self) -> None:
        super().__init__('mpc_flight_controller')
        self.parameters_init()

        #vehicle states
        self.position_odometry = np.zeros(3)
        self.velocity_odometry = np.zeros(3)
        self.q_odometry = np.array([0.5, -0.5, 0.5, 0.5])
        self.omega_odometry = np.zeros(3)
        
        
        #stores the last five values of the omega to lock position
        self.omega_array = np.empty((20, 2))
        self.inclination_error = np.empty(20)
        self.current_index = 0
        
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        #odometry subscriber
        self.odometry_subscriber = self.create_subscription(
            FwOdometry, '/fw_odometry', self.odometry_callback, qos_profile
        )
        
        #thurst torque publisher
        self.control_publisher = self.create_publisher(
            ThrustTorqueSetpoint, '/controller_output', qos_profile
        )
        
        self.ref_publisher = self.create_publisher(
            CustomReference, '/reference', qos_profile
        )
        
        #position lock flags    
        self.lock_position = False #used to start altitude holding mode
        self.position_locked_once = False  # Initialize a flag to track first lock
        self.transition = False
        

        #timer
        self.controller_timer = self.create_timer(0.2, self.controller_timer_callback)
        
        self.opti = ca.Opti()
        self.setup_controller()

        
    def parameters_init(self):
        with open('parameters.yaml', 'r') as file:
            params = yaml.safe_load(file)

        # Vehicle parameters
        vehicle_inertia = np.array(params['vehicle']['inertia'])
        self.Jxx = vehicle_inertia[0]
        self.Jyy = vehicle_inertia[1]
        self.Jzz = vehicle_inertia[2]
        self.vehicle_mass = params['vehicle']['mass']
        
        #constants for dynamics
        self.area = 0.15
        self.AR = 6.5
        self.span = np.sqrt(self.area * self.AR)  # Wing span
        # Aerodynamic coefficients
        self.CL0=0.15188
        self.eff=0.97
        self.CLa=5.015
        self.CD0=0.029
        self.Cem0=0.075
        self.Cema=-0.463966
        self.CYb=-0.258244
        self.Cellb=-0.039250
        self.Cenb=0.100826
        self.CDp=0.0
        self.CYp=0.065861
        self.CLp=0.0
        self.Cellp=-0.487407
        self.Cemp=0.0
        self.Cenp=-0.040416
        self.CDq=0.055166
        self.CYq=0.0
        self.CLq=7.971792
        self.Cellq=0.0
        self.Cemq=-12.140140
        self.Cenq=0.0
        self.CDr=0.0
        self.CYr=0.230299
        self.CLr=0.0
        self.Cellr=0.078165
        self.Cemr=0.0
        self.Cenr=-0.089947
        self.alpha_stall=0.3391428111
        self.CD_fp_k1 = -0.224
        self.CD_fp_k2 = -0.115
        self.mac=0.22
        self.rho=1.2041
        self.M = 15

        # Limits of thrust and torque
        self.motor_constant = params['vehicle']['motor_constant']
        self.max_rotor_vel = params['vehicle']['max_rotor_vel']
        self.max_thrust_one_motor = self.max_rotor_vel**2 * self.motor_constant        
        self.max_thrust = self.max_thrust_one_motor * 4 
        self.max_torque_x = params['limits']['max_x_torque']  
        self.max_torque_y = params['limits']['max_y_torque'] 
        self.max_torque_z = params['limits']['max_z_torque'] 

        #simulation parameters
        self.wind_velocity = np.array(params['simulation']['wind_velocity'])
        self.g = params['simulation']['gravity']
        
        self.counter_lock = 0
        # Controller parameters
        self.T = params['nmpc']['T']
        self.N= params['nmpc']['N']
        #weight matrices
        self.Q_init = np.array(params['nmpc']['Q_quat_init'])
        self.Q_angle_init = np.array(params['nmpc']['Q_angle_init'])
        self.R_init = np.array(params['nmpc']['R_init'])
        self.R_var_init = np.array(params['nmpc']['R_var_init'])
        
        self.Q_second = np.array(params['nmpc']['Q_quat_second'])
        self.Q_angle_second = np.array(params['nmpc']['Q_angle_second'])
        self.R_second = np.array(params['nmpc']['R_second'])
        self.R_var_second = np.array(params['nmpc']['R_var_second'])
            
        
        #number of states
        self.nstates = 13
        #number of controls
        self.ncontrols = 4
        
        #predifined output
        self.predifined_output = np.array([self.g*self.vehicle_mass, 0, 0, 0])
        
        #reference state corresponding to hover
        self.ref = np.array([0, 0, -35,  #position
                             0, 0, 0, #linear velocity
                             0.5, -0.5, 0.5, 0.5, #quaternion
                             0, 0, 0]) #angular velocity
        #counter to see if a first odometry reading has been received
        self.counter = 0
        
        self.next_states = None
        self.u0 = np.ones((self.N, self.ncontrols))
        self.last_cntr = np.array([8,0,0,0])
        

    def odometry_callback(self, msg):
        self.position_odometry = np.array([msg.x, msg.y, msg.z])
        self.velocity_odometry = np.array([msg.vx, msg.vy, msg.vz])
        self.q_odometry = np.array([msg.q0, msg.q1, msg.q2, msg.q3])
        self.omega_odometry = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
        if self.counter == 0:
            # Initialize next_states with the current state values
            self.next_states = np.tile(
                np.concatenate((self.position_odometry, self.velocity_odometry, self.q_odometry, self.omega_odometry)),
                (self.N + 1, 1)
            )
            self.u0 = np.tile(self.predifined_output, (self.N, 1))
        self.counter += 1
    
    
    def controller_timer_callback(self) -> None:
        if self.counter == 0:
            self.publish_commands(np.array([0.1,0,0,0]))
        # Shift the array to make space for the new message and store the new angular velocities
        self.omega_array = np.roll(self.omega_array, -1, axis=0)
        self.omega_array[-1] = self.omega_odometry[1:]

        
        # Calculate the current body x-axis (thrust direction) from the quaternion
        eBx = np.array([
                self.q_odometry[0]**2+self.q_odometry[1]**2-self.q_odometry[2]**2-self.q_odometry[3]**2,
                2*(self.q_odometry[1]*self.q_odometry[2]+self.q_odometry[0]*self.q_odometry[3]),
                2*(self.q_odometry[1]*self.q_odometry[3]-self.q_odometry[0]*self.q_odometry[2])
            ])  
        # Calculate the angle (alpha) between current and desired thrust direction
        eBcmd_x = np.array([0, 0, -1])
        alpha = np.arccos(np.clip(np.dot(eBx, eBcmd_x), -1.0, 1.0))
       # print(alpha)
        # Shift the array to make space for the new message and store the new angular velocities
        self.inclination_error = np.roll(self.inclination_error, -1, axis=0)
        self.inclination_error[-1] = alpha
        
        self.transition = False
        
            # #if np.all(self.inclination_error < np.deg2rad(10)) and np.all(self.omega_array < 6):
        if np.mean(np.abs(self.inclination_error)) < np.deg2rad(20) and np.mean(np.abs(self.omega_array < 8)) and self.lock_position == False :
            self.lock_position = True
            if self.counter_lock > 0:
               self.ref[:3] = self.position_odometry
               if np.abs(alpha) < 1e-8:
                   q_erp = np.array([1,0,0,0])
               else:
                   if np.linalg.norm(np.cross(eBx, eBcmd_x)) > 1e-8:
                       n = np.cross(eBx, eBcmd_x)/np.linalg.norm(np.cross(eBx, eBcmd_x))
                   else:
                       n = np.array([0,0,0])
                   n_b = rotate_quat(q = quat_conj(self.q_odometry), v = n)
                   q_erp = np.array([np.cos(alpha/2), n_b[0]*np.sin(alpha/2), n_b[1]*np.sin(alpha/2), n_b[2]*np.sin(alpha/2)])

               qex = quat_mult(quat_conj(quat_mult(self.q_odometry, q_erp)),self.ref[6:10])
               print('old qd: ', self.ref[6:10]) 
            #   self.ref[6:10] = quat_mult(qex, np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]))
               print('new q_d: ', self.ref[6:10])
               
               self.position_locked_once = True
               print('position locked')
               print('pos_ref ', self.ref[:3])
               self.transition = True

        elif np.mean(np.abs(self.inclination_error)) > np.deg2rad(20)  and self.lock_position == True: #and self.counter_lock < 1:
            self.lock_position = False
            #if self.counter_lock > 0:
                # self.ref[:3] = self.position_odometry
            self.counter_lock +=1
            self.position_locked_once = False 
            print('position unlocked ', self.counter_lock)
            print('pos_ref ', self.ref[:3])
            print()
            self.transition = True
        
        cntr = self.solve(x_current = np.concatenate((self.position_odometry, self.velocity_odometry, self.q_odometry, self.omega_odometry)), x_reference = self.ref, next_controls = self.predifined_output, u_cur = self.last_cntr)
        self.last_cntr = cntr
        
        self.publish_commands(cntr)    
        self.publish_ref()    
        # self.get_logger().info(f"Controller output: {cntr}")
                
    def publish_commands(self, controller_output):
        msg = ThrustTorqueSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        msg.thrust = controller_output[0]
        msg.tx = controller_output[1]
        msg.ty = controller_output[2]
        msg.tz = controller_output[3]
        
        msg.positionlock = self.lock_position
        
        #controllers parameters
        msg.qquat = self.Q_init.astype(np.float32)
        msg.qangle = self.Q_angle_init.astype(np.float32)
        msg.r = self.R_init.astype(np.float32)
        msg.rvar = self.R_var_init.astype(np.float32)

        
        self.control_publisher.publish(msg)
        
    
    def setup_controller(self):
        #states
        pos = ca.MX.sym('pos', 3)
        vel = ca.MX.sym('vel', 3)
        q = ca.MX.sym('q', 4)
        omega = ca.MX.sym('omega', 3)
        x = ca.vertcat(pos, vel, q, omega)
        #controls
        thrust = ca.MX.sym('thrust', 1)
        torque_x = ca.MX.sym('torque_x', 1)
        torque_y = ca.MX.sym('torque_y', 1)
        torque_z = ca.MX.sym('torque_z', 1)
        u = ca.vertcat(thrust, torque_x, torque_y, torque_z)
 
        #dynamics
        # intg_options = {}
        # intg_options["number_of_finite_elements"] = 4
        # intg_options["simplify"] = True
        # dt = self.T/self.N
        
        # dae = {}
        # dae['x'] = x
        # dae['p'] = u
        # dae['ode'] = ode
        
        # intg = ca.integrator('intg', 'rk', dae, 0, dt, intg_options)
        # print(intg)
        
        # F = ca.Function('F', [x, u], [intg(x0=x, p=u)['xf']], ['x', 'u'], ['xnext'])
        
                #integrate using rk4 method explicitly
        dt = self.T/self.N/4
        for i in range(self.N):  
            #         #rk4 method
            k1 = self.dynamics_quat(pos, vel, q, omega, thrust, torque_x, torque_y, torque_z)
            x2 = x + dt/2*k1
            k2 = self.dynamics_quat(x2[:3], x2[3:6], x2[6:10], x2[10:], thrust, torque_x, torque_y, torque_z)
            x3 = x + dt/2*k2
            k3 = self.dynamics_quat(x3[:3], x3[3:6], x3[6:10], x3[10:], thrust, torque_x, torque_y, torque_z)   
            x4 = x + dt*k3
            k4 = self.dynamics_quat(x4[:3], x4[3:6], x4[6:10], x4[10:], thrust, torque_x, torque_y, torque_z)
            x_next = x + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        F = ca.Function('F', [x, u], [x_next], ['x', 'u'], ['xnext'])
       # F = F.expand()
        F_mapped = F.map(self.N,"thread",self.N)
                
        self.opti_X = self.opti.variable(self.N+1, self.nstates)
        self.opti_U = self.opti.variable(self.N, self.ncontrols)
        #parameters (dont optimize over)
        self.opti_x0 = self.opti.parameter(self.nstates) #stores the initial state
        self.opti_x_ref = self.opti.parameter(self.nstates) #stores the reference state
        self.opti_cur_u = self.opti.parameter(self.ncontrols) #stores the initial control output
        self.opti_u_ref = self.opti.parameter(self.ncontrols) #stores the reference control output
        
        
        # for k in range(self.N):
        #     self.opti.subject_to(self.opti_X[k+1, :] == F(self.opti_X[k, :], self.opti_U[k, :]).T)

        #     if k ==0:
        #         self.opti.subject_to(self.opti_X[k, :] == self.opti_x0.T)
        
        
        # Apply the mapped function for state transitions
        X = self.opti_X[0:self.N, :]  # Extract the first N states
        U = self.opti_U           # Controls over horizon

        # Predict next states using the mapped function
        X_next = F_mapped(X.T, U.T).T

        # Apply the constraints for state transitions
        self.opti.subject_to(self.opti_X[1:self.N+1, :] == X_next)

        # Initial condition constraints
        self.opti.subject_to(self.opti_X[0, :] == self.opti_x0.T)
        self.opti.subject_to(self.opti.bounded( 0, self.opti_U[:, 0], self.max_thrust))
        self.opti.subject_to(self.opti.bounded(-self.max_torque_x, self.opti_U[:, 1], self.max_torque_x))
        self.opti.subject_to(self.opti.bounded(-self.max_torque_y, self.opti_U[:, 2], self.max_torque_y))
        self.opti.subject_to(self.opti.bounded(-self.max_torque_z, self.opti_U[:, 3], self.max_torque_z))
       # self.opti.subject_to(self.opti.bounded([0, -self.max_torque_x, -self.max_torque_y, -self.max_torque_z],
       #                            	self.opti_U[:, 1:],
       #                            	[self.max_thrust, self.max_torque_x, self.max_torque_y, self.max_torque_z]))
#
        #weight matrices
        self.Q = self.opti.parameter(self.nstates - 4)
        self.Q_angle = self.opti.parameter(2)
        self.R = self.opti.parameter(self.ncontrols)
        self.R_var = self.opti.parameter(self.ncontrols)

        # #objective function
        #vectorize the state error
        # Repeat the reference vector for all time steps (N+1, since it includes the last step)
        X_ref_repeated = ca.repmat(self.opti_x_ref.T, self.N + 1, 1)
        X_error = ca.horzcat(self.opti_X[:, 0:6] - X_ref_repeated[:,0:6],  # Shape: (N+1, 6)
                         self.opti_X[:, 10:] - X_ref_repeated[:,10:])  # Shape: (N+1, 3)
        # Compute quaternion error        
        Q_error = quat_mult_vectorized(X_ref_repeated[:, 6:10], quat_conj_vectorized(self.opti_X[:,6:10])) + 1e-8
        # Decompose all quaternions
        angles = decompose_quaternion_vectorized(Q_error)
        #angles = decompose_quaternion_error_vectorized(quat_conj_vectorized(self.opti_X[:,6:10]), self.opti_x_ref[6:10])
        # Control variable error (first step needs special handling)
        U_var = ca.vertcat(self.opti_U[0, :] - self.opti_cur_u[:].T,  # First control step
                            self.opti_U[1:, :] - self.opti_U[:-1, :])  # All other steps
        
        U_ref_repeated = ca.repmat(self.opti_u_ref.T, self.N, 1)
        U_error = self.opti_U - U_ref_repeated
        # Objective function terms
        state_term = ca.sum1(ca.power(X_error,2) @ self.Q)
        angle_term = ca.sum1(ca.power(angles,2) @ self.Q_angle)
        control_var_term = ca.sum1(ca.power(U_var,2) @ self.R_var)
        control_term = ca.sum1(ca.power(U_error,2) @ self.R)


        # Combine terms for total objective
        obj = state_term + angle_term + control_var_term + control_term


        self.opti.minimize(obj)


        
        opts_settings = {'ipopt.max_iter':250, # 750,
                'ipopt.print_level':0,
                'print_time':0,
                'expand': True,
                'ipopt.acceptable_tol':1e-5,
                'ipopt.acceptable_obj_change_tol':1e-4}
        
        
        self.opti.solver('ipopt', opts_settings)
        # self.opti.solver("fatrop")
 
    
    def dynamics_quat(self, pos, vel, q, omega, thrust, torque_x, torque_y, torque_z):
        #f_aero should be in inertial frame and m_aero in body frame
        f_aero, m_aero = self.estimate_aero_quat_poly(
           vehicle_velocity=ca.vertcat(vel[1], vel[0], -vel[2]), #pass the velocity in ENU frame, as the plugin calculates
           vehicle_quat=q,
           vehicle_omega = omega
        )
    
        # f_aero = ca.vertcat(0, 0, 0)
        
        # m_aero = ca.vertcat(0, 0, 0)

        
        # Dynamics calculations http://www.iri.upc.edu/people/jsola/JoanSola/objectes/notes/kinematics.pdf
        dynamics = ca.vertcat(
            vel[0],  # dx (velocity in x direction)
            vel[1],  # dy (velocity in y direction)
            vel[2],  # dz (velocity in z direction)
            ((2*(q[0]*q[0]+q[1]*q[1])-1)*thrust + f_aero[0])/self.vehicle_mass,  # dvx (acceleration in x direction)
            (2*(q[1]*q[2]+ q[0]*q[3])*thrust + f_aero[1])/self.vehicle_mass,  # dvy (acceleration in y direction)
            self.g + (2*(q[1]*q[3]-q[0]*q[2])*thrust + f_aero[2])/self.vehicle_mass,  # dvz (acceleration in z direction)
            1/2*(-omega[0]*q[1] - omega[1]*q[2] - omega[2]*q[3]),  #dq0 = -wx*q1-wy*q2-wz*q3,
            1/2*(omega[0]*q[0] + omega[2]*q[2] - omega[1]*q[3]),   #dq1 = wx*q0+wz*q2-wy*q3
            1/2*(omega[1]*q[0] - omega[2]*q[1] + omega[0]*q[3]),   #dq2 = wy*q0-wz*q1+wx*q3,
            1/2*(omega[2]*q[0] + omega[1]*q[1] - omega[0]*q[2]),    #dq3 = wz*q0+wy*q1-wx*q2,
            (omega[1]*omega[2]*(-self.Jzz+self.Jyy)+torque_x+m_aero[0])/self.Jxx,  # dp (angular velocity derivative around x-axis)
            (omega[0]*omega[2]*(-self.Jxx+self.Jzz)+torque_y+m_aero[1])/self.Jyy, # dq (angular velocity derivative around y-axis)
            (omega[0]*omega[1]*(-self.Jyy+self.Jxx)+torque_z+m_aero[2])/self.Jzz # dr (angular velocity derivative around z-axis)
        )

        return dynamics
    
    def estimate_aero_quat_poly(self, vehicle_velocity, vehicle_quat, vehicle_omega):
        epsilon = 1e-8  # Small value to avoid division by zero

        u = vehicle_velocity + epsilon + +  np.array([-4.1e-5, -1e-05, -2e-5])
        # Quaternions for transformation
        q_ned_gz = ca.vertcat(0, np.sqrt(2+ epsilon)/2,  np.sqrt(2+ epsilon)/2, 0)
        q_attitude_fw_gz = quat_mult(q_ned_gz, vehicle_quat)

        # Rotation matrix from quaternion
        R_attitude_fw_gz = ca.vertcat(
            1 - 2 * (q_attitude_fw_gz[2] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] - q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[2]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] + q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[1]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[2]), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[1]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[2] ** 2)
        )

        # Velocity in the LD plane
        R_attitude_fw_gz_1 = ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7])
        velInLDPlane = u - ca.dot(u, R_attitude_fw_gz_1) * R_attitude_fw_gz_1
        speedInLDPlane = ca.sqrt(ca.dot(velInLDPlane, velInLDPlane)) + epsilon

        # Stability axes
        stability_x_axis = velInLDPlane / (speedInLDPlane + epsilon)
        stability_y_axis = R_attitude_fw_gz_1 + epsilon
        stability_z_axis = ca.cross(stability_x_axis, stability_y_axis) + epsilon
        
        stabx_proj_bodyx = ca.dot(stability_x_axis, ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]))
        stabx_proj_bodyz = ca.dot(stability_x_axis, ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8]))

        # # Angle of attack and sideslip angle
        # alpha = ca.arctan2(u[2], ca.fmax(u[0], epsilon))  # Prevent division by zero
        # beta = ca.arcsin(u[1] / ca.fmax(ca.sqrt(ca.dot(u, u)), epsilon))  # Prevent division by zero
        alpha = ca.arctan2(stabx_proj_bodyz, stabx_proj_bodyx) 
        velSW = ca.dot(u, ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7]))
        velFW = ca.dot(u, ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]))
        beta = ca.arctan2(velSW, velFW)
        
        dyn_pres = 0.5 * self.rho* speedInLDPlane*speedInLDPlane + epsilon
        half_rho_vel = 0.5 * self.rho* speedInLDPlane+ epsilon
        #compute the coefficients
        #polynomial approximation
        #CL =   4.22819171e-05*alpha**12 -2.05792345e-05*alpha**11 -1.31837307e-03*alpha**10  +8.68188241e-04*alpha**9  +1.58185642e-02*alpha**8 -2.04789751e-02*alpha**7 -9.13327827e-02*alpha**6  +2.42366992e-01*alpha**5  +2.58218274e-01*alpha**4 -1.21923589e+00*alpha**3 -3.19311542e-01*alpha**2  +1.78450353e+00*alpha+1.20632268e-01
     #   CD = -8.48010742e-04*alpha**6 + 3.84369107e-02*alpha**5 + 1.38581751e-02*alpha**4 -4.56721550e-01*alpha**3 -6.41910716e-02*alpha**2 + 1.02166547e+00*alpha + 7.42863362e-02 + CL**2/np.pi/self.AR
      
        #truth
        sigma = (1+ca.exp(-1*self.M*(alpha-self.alpha_stall))+ca.exp(self.M*(alpha+self.alpha_stall)))/((1+ca.exp(-1*self.M*(alpha-self.alpha_stall)))*(1+ca.exp(self.M*(alpha+self.alpha_stall))))
        CL = (1-sigma)*(self.CL0+self.CLa*alpha)+sigma*(2*(alpha/ca.fabs(alpha))*ca.sin(alpha)*ca.sin(alpha)*ca.cos(alpha))
        CD_fp = 2/(1+ca.exp(self.CD_fp_k1+self.CD_fp_k2*(self.AR)))
        CD = (1-sigma)*(self.CD0+(CL*CL)/(np.pi*self.AR*self.eff))+sigma*ca.fabs(CD_fp*(0.5-0.5*np.cos(2*alpha)))

        #compute forces considering only the effects of the angle of attack
        #lift = (CL * dyn_pres) * self.area * stability_z_axis + epsilon
        #drag = (CD * dyn_pres) * self.area * (-stability_x_axis) + epsilon
       
        #compute forces considering every part of the dynamics
        lift = (CL* dyn_pres)*(self.area*(-1*stability_z_axis))
        drag = (CD*dyn_pres)*(self.area*(-1*stability_x_axis))

        Cy = self.CYb*beta
        #side_force = (Cy*dyn_pres)*(self.area*stability_y_axis) + epsilon
        side_force = (Cy*dyn_pres)*(self.area*stability_y_axis)

        # Moments
        pm = ((self.Cemp*vehicle_omega[0]*self.span/2*half_rho_vel)+(self.Cemq*(vehicle_omega[1]*self.mac/2)*half_rho_vel)+(self.Cemr*(vehicle_omega[2]*self.span/2)*half_rho_vel))*(self.area*self.mac*ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7]))
        #pm = (self.Cemq * vehicle_omega[1] * self.mac / 2 * half_rho_vel) * self.area * self.mac * R_attitude_fw_gz_1 + epsilon
        
        Cell = self.Cellb * beta + epsilon
       # rm = ((Cell*dyn_pres))*(self.area*self.span*ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]))
        rm = (self.Cellp * vehicle_omega[0] * self.span / 2 * half_rho_vel) * self.area * self.span * ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]) + epsilon

        Cen = self.Cenb * beta + epsilon
        #ym = ((Cen*dyn_pres))*(self.area*self.span*ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8]))
        ym = (self.Cenr * vehicle_omega[2] * self.span / 2 * half_rho_vel) * self.area * self.span * ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8]) + epsilon


        aero_force = lift + drag + side_force
        aero_moment = pm + rm + ym

        #force expressed in ENU - rotate to convert to NED world frame
        #aero_force = ca.vertcat(aero_force[1], aero_force[0], -aero_force[2])
        #moment expressed in gazebo body frame - rotate to FW body frame        
       # aero_moment = ca.vertcat(aero_moment[1], aero_moment[0], -aero_moment[2])
        #aero_moment = self.quat_rotate(q = quat_conj(q_ned_gz), v = aero_moment)
        return aero_force, aero_moment
    
    def estimate_aero_quat_casadi(self,vehicle_velocity, vehicle_quat, vehicle_omega,
        CL0=0.15188, AR=6.5, eff=0.97, CLa=5.015, CD0=0.029, Cem0=0.075, Cema=-0.463966, CYb=-0.258244, Cellb=-0.039250, Cenb=0.100826, CDp=0.0, CYp=0.065861, CLp=0.0, Cellp=-0.487407, Cemp=0.0, Cenp=-0.040416, CDq=0.055166, CYq=0.0, CLq=7.971792, Cellq=0.0, Cemq=-12.140140, Cenq=0.0, CDr=0.0, CYr=0.230299, CLr=0.0, Cellr=0.078165, Cemr=0.0, Cenr=-0.089947, alpha_stall=0.3391428111, CD_fp_k1 = -0.224, CD_fp_k2 = -0.115, area=0.15, mac=0.22, rho=1.2041, M = 15):
        # Small value to avoid division by zero
        epsilon = 1e-8
        span = np.sqrt(AR* area)
    #  vehicle_velocity = ca.vertcat(vehicle_velocity[1], vehicle_velocity[0], -vehicle_velocity[2])

        u = vehicle_velocity + epsilon
        # Quaternions for transformation
        q_ned_gz = ca.vertcat(0, np.sqrt(2)/2 + epsilon,  np.sqrt(2)/2 + epsilon, 0)
        q_attitude_fw_gz = quat_mult(q_ned_gz, vehicle_quat)
        #q_attitude_fw_gz = vehicle_quat
        # Rotation matrix from quaternion
        R_attitude_fw_gz = ca.vertcat(
            1 - 2 * (q_attitude_fw_gz[2] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] - q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[2]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] + q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[1]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[2]), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[1]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[2] ** 2)
        )

        # Velocity in the LD plane
        R_attitude_fw_gz_1 = ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7])
        velInLDPlane = u - ca.dot(u, R_attitude_fw_gz_1) * R_attitude_fw_gz_1
        speedInLDPlane = ca.sqrt(ca.dot(velInLDPlane, velInLDPlane)) + epsilon

        # Stability axes
        stability_x_axis = velInLDPlane / (speedInLDPlane + epsilon )
        stability_y_axis = R_attitude_fw_gz_1  + epsilon
        stability_z_axis = ca.cross(stability_x_axis, stability_y_axis)  + epsilon
        
        stabx_proj_bodyx = ca.dot(stability_x_axis, ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6])) + epsilon
        stabx_proj_bodyz = ca.dot(stability_x_axis, ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8])) + epsilon

        # # Angle of attack and sideslip angle
        # alpha = ca.arctan2(u[2], ca.fmax(u[0], epsilon))  # Prevent division by zero
        # beta = ca.arcsin(u[1] / ca.fmax(ca.sqrt(ca.dot(u, u)), epsilon))  # Prevent division by zero
        alpha = ca.arctan2(stabx_proj_bodyz  + epsilon , stabx_proj_bodyx  + epsilon) 
        velSW = ca.dot(u, ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7]))  + epsilon
        velFW = ca.dot(u, ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6])) + epsilon
        beta = ca.arctan2(velSW  + epsilon, velFW  + epsilon)
        
        dyn_pres = 0.5 *  rho* speedInLDPlane*speedInLDPlane
        half_rho_vel = 0.5 *  rho* speedInLDPlane
        #compute the coefficients
        
        #truth
        sigma = (1+ca.exp(-1* M*(alpha- alpha_stall))+ca.exp( M*(alpha+ alpha_stall)))/((1+ca.exp(-1* M*(alpha- alpha_stall)))*(1+ca.exp( M*(alpha+ alpha_stall))))
        CL = (1-sigma)*( CL0+ CLa*alpha)+sigma*(2*(alpha/ca.fabs(alpha))*ca.sin(alpha)*ca.sin(alpha)*ca.cos(alpha))
        CD_fp = 2/(1+ca.exp( CD_fp_k1+ CD_fp_k2*( AR)))
        CD = (1-sigma)*( CD0+(CL*CL)/(np.pi* AR* eff))+sigma*ca.fabs(CD_fp*(0.5-0.5*np.cos(2*alpha)))

        #polynomial approximation
      #  CL =   4.22819171e-05*alpha**12 -2.05792345e-05*alpha**11 -1.31837307e-03*alpha**10  +8.68188241e-04*alpha**9  +1.58185642e-02*alpha**8 -2.04789751e-02*alpha**7 -9.13327827e-02*alpha**6  +2.42366992e-01*alpha**5  +2.58218274e-01*alpha**4 -1.21923589e+00*alpha**3 -3.19311542e-01*alpha**2  +1.78450353e+00*alpha+1.20632268e-01
        #CD = -8.48010742e-04*alpha**6 + 3.84369107e-02*alpha**5 + 1.38581751e-02*alpha**4 -4.56721550e-01*alpha**3 -6.41910716e-02*alpha**2 + 1.02166547e+00*alpha + 7.42863362e-02 + CL**2/np.pi/ AR
        #compute forces considering only the effects of the angle of attack
        #lift = (CL * dyn_pres) *  area * stability_z_axis + epsilon
        #drag = (CD * dyn_pres) *  area * (-stability_x_axis) + epsilon
        
        #compute forces considering every part of the dynamics
        lift = (CL* dyn_pres+( CLp*vehicle_omega[0]* span/2*half_rho_vel)+( CLq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( CLr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area*(-1*stability_z_axis))
        drag = (CD*dyn_pres+( CDp*vehicle_omega[0]* span/2*half_rho_vel)+( CDq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( CDr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area*(-1*stability_x_axis))

        Cy =  CYb*beta
        #side_force = (Cy*dyn_pres)*( area*stability_y_axis) + epsilon
        side_force = (Cy*dyn_pres+( CYp*vehicle_omega[0]* span/2*half_rho_vel)+( CYq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( CYr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area*stability_y_axis)

        # Moments
        pm = (( Cemp*vehicle_omega[0]* span/2*half_rho_vel)+( Cemq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( Cemr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area* mac*ca.vertcat(R_attitude_fw_gz[1], R_attitude_fw_gz[4], R_attitude_fw_gz[7]))
        #pm = ( Cemq * vehicle_omega[1] *  mac / 2 * half_rho_vel) *  area *  mac * R_attitude_fw_gz_1 
        
        Cell =  Cellb * beta 
        rm = ((Cell*dyn_pres)+( Cellp*(vehicle_omega[0]* span/2*half_rho_vel))+( Cellq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( Cellr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area* span*ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]))
        #rm = ( Cellp * vehicle_omega[0] *  span / 2 * half_rho_vel) *  area *  span * ca.vertcat(R_attitude_fw_gz[0], R_attitude_fw_gz[3], R_attitude_fw_gz[6]) 

        Cen =  Cenb * beta
        ym = ((Cen*dyn_pres)+( Cenp*(vehicle_omega[0]* span/2*half_rho_vel))+( Cenq*(vehicle_omega[1]* mac/2)*half_rho_vel)+( Cenr*(vehicle_omega[2]* span/2)*half_rho_vel))*( area* span*ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8]))
        #ym = ( Cenr * vehicle_omega[2] *  span / 2 * half_rho_vel) *  area *  span * ca.vertcat(R_attitude_fw_gz[2], R_attitude_fw_gz[5], R_attitude_fw_gz[8]) 


        aero_force = lift + drag + side_force
        aero_moment = pm + rm + ym

        #force expressed in ENU - rotate to convert to NED world frame
        aero_force = ca.vertcat(aero_force[1], aero_force[0], -aero_force[2])
        # #moment expressed in gazebo world frame - rotate to NED frame
        aero_moment = ca.vertcat(aero_moment[1], aero_moment[0], -aero_moment[2])
        #aero_moment rotate to FW body frame
        aero_moment = quat_rotate(q = quat_conj(vehicle_quat), v = aero_moment)        
        # aero_moment = ca.vertcat(aero_moment[1], aero_moment[0], -aero_moment[2])
        #aero_moment =  quat_rotate(q = quat_conj(q_ned_gz), v = aero_moment)
        return aero_force, aero_moment

        
    def solve(self, x_current, x_reference, next_controls, u_cur):
       
        self.opti.set_value(self.opti_x0, ca.vertcat(x_current))        
        self.opti.set_value(self.opti_x_ref, ca.vertcat(x_reference))
        self.opti.set_value(self.opti_cur_u, ca.vertcat(u_cur))
        self.opti.set_value(self.opti_u_ref, ca.vertcat(next_controls))
        
        if self.lock_position: 
            self.opti.set_value(self.Q, self.Q_init)        
            self.opti.set_value(self.Q_angle, self.Q_angle_init)

        else:
            Q_no_position = self.Q_init
            Q_no_position[0:3] = 0
            self.opti.set_value(self.Q, Q_no_position)
            Q_angle_no_position = self.Q_angle_init
            Q_angle_no_position[0] = 0
            self.opti.set_value(self.Q_angle, Q_angle_no_position)
        #self.opti.set_value(self.Q, self.Q_init)
        #self.opti.set_value(self.Q_angle, self.Q_angle_init)
        self.opti.set_value(self.R_var, self.R_var_init)
        self.opti.set_value(self.R, self.R_init)
        if self.next_states is not None:
                self.opti.set_initial(self.opti_X, self.next_states)
        self.opti.set_initial(self.opti_U, self.u0)
        ## provide the initial guess of the optimization targets
        if self.transition ==  True:
            print('transicao')
            self.opti.set_value(self.R_var, np.array([5000, 500, 500, 500]))
        #    if self.next_states is not None:
        #        self.opti.set_initial(self.opti_X, self.next_states)
        #    self.opti.set_initial(self.opti_U, self.u0)
        #else:
        #    self.opti.set_initial(self.opti_U, np.tile(self.predifined_output, (self.N, 1)))

        # self.opti.subject_to(self.opt_states[0, :] == current_state)

        try:
            sol = self.opti.solve()
            return sol.value(self.opti_U)[0, :]

        except Exception as e:
            print(f"Optimization failed: {str(e)}")

            # Check if the error is due to max iterations and return a predefined output
            if "Maximum_Iterations_Exceeded" in str(e):
                print("Max iterations reached. Returning predefined output.")
                #return self.predifined_output
                return self.last_cntr

            # Debugging information
            for var in self.opti.debug.x:
                print(f"{var}: {self.opti.debug.value(var)}")
            
            #return self.predifined_output
            return self.last_cntr
            # Re-raise the exception if it's not a max iteration issue
           # raise e

    def publish_ref(self):
        msg = CustomReference()
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        msg.altitude_ref = self.ref[2]
        msg.x_ref = self.ref[0]
        msg.y_ref = self.ref[1]
        msg.q0_ref = self.ref[6]
        msg.q1_ref = self.ref[7]
        msg.q2_ref = self.ref[8]
        msg.q3_ref = self.ref[9]
        self.ref_publisher.publish(msg)
            
def main(args=None) -> None:
    print('Starting MPC node...')
    rclpy.init(args=args)
    node = MPC_Flight_controller() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
