#!/usr/bin/env python3
import numpy as np
import yaml
import time
import rclpy
from scipy.linalg import logm
from custom_msg.msg import FwOdometry, ThrustTorqueSetpoint, CustomControllerOutput, CustomReference
from common import quat_to_rot, quat_mult, quat_conj, rotate_quat, rot_to_quat
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.node import Node




class PID_Flight_controller(Node):
    def __init__(self) -> None:
        super().__init__('pid_flight_controller')
        self.parameters_init()
        
        self.counter = 0

        #vehicle states
        self.position = np.array([0.0001, 0.0001, 0.0001])
        self.velocity = np.array([0.0001, 0.0001, 0.0001])
        self.q_odometry = np.array([0.5,-0.5,0.5,0.5])
        self.angular_velocity =  np.array([0.0001, 0.0001, 0.0001])
        
        #store the last five values of the omega
        self.omega_array = np.empty((20, 2))
        self.inclination_error = np.empty(20)
        self.current_index = 0
        
        self.e_height_int = 0
        self.h_error_last = 0
        self.last_h = 0
        self.last_attitude_rates = np.zeros(3)
        self.e_attitude_rates_int = np.zeros(3)
        self.e_attitude_rates_last = np.zeros(3)

        self.clean_height_error()
        
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
        
        self.lock_position = False #used to start altitude holding mode
        self.position_locked_once = False  # Initialize a flag to track first lock
        self.counter_lock = 0
        #timer
        self.controller_timer = self.create_timer(0.2, self.controller_timer_callback)
        # self.controller_timer = self.create_timer(0.01, self.controller_timer_callback)

        
    def parameters_init(self):
        with open('parameters.yaml', 'r') as file:
            params = yaml.safe_load(file)

        # Vehicle parameters
        vehicle_inertia = np.array(params['vehicle']['inertia'])
        self.Ixx = vehicle_inertia[0]
        self.Iyy = vehicle_inertia[1]
        self.Izz = vehicle_inertia[2]
        self.vehicle_mass = params['vehicle']['mass']

        # Limits of thrust and torque
        self.max_thrust = params['limits']['max_thrust']
        self.max_torque_x = params['limits']['max_x_torque']  
        self.max_torque_y = params['limits']['max_y_torque'] 
        self.max_torque_z = params['limits']['max_z_torque'] 

        #simulation parameters
        self.wind_velocity = np.array(params['simulation']['wind_velocity'])
        self.g = params['simulation']['gravity']
        
        #controller parameters
        self.height_p_gains = params['pid3']['height_p_gains']
        self.height_i_gains = params['pid3']['height_i_gains']
        self.height_d_gains = params['pid3']['height_d_gains']
        self.attitude_p_gains = np.array(params['pid3']['attitude_p_gains'])
        self.attitude_rates_p_gains = np.array(params['pid3']['attitude_rates_p_gains'])
        self.attitude_rates_i_gains = np.array(params['pid3']['attitude_rates_i_gains'])
        self.attitude_rates_d_gains = np.array(params['pid3']['attitude_rates_d_gains'])
        
        #reference
        self.h_d = -10
        #self.R_d = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        self.q_d = np.array([0.5, -0.5, 0.5, 0.5])
        self.last_time = None
        
        self.last_ftx = None
    
    def clean_height_error(self):
        self.e_height_int = 0
        self.h_error_last = 0

    def clean_attitude_error(self):
        self.e_attitude_rates_int = np.zeros(3)
        self.e_attitude_rates_last = np.zeros(3)  
        
    def get_time(self):
        # Time-step determination
        cur_time = int(self.get_clock().now().nanoseconds)/4 # Current time
        if self.last_time is None:
            self.last_time = cur_time # Initialization of variable

        self.dt = (cur_time - self.last_time)/(10**9) # Time step
        self.last_time = cur_time # Keep in memory the previous time step

    def odometry_callback(self, msg):
        self.position = np.array([msg.x, msg.y, msg.z])
        self.velocity = np.array([msg.vx, msg.vy, msg.vz])
        self.q_odometry = np.array([msg.q0, msg.q1, msg.q2, msg.q3])
        self.angular_velocity = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])

    def estimate_aero_quat_np(self,vehicle_velocity, vehicle_quat, vehicle_angular_velocity, wind = np.array([0,0,0]),
                                  CL0=0.15188, AR=6.5, eff=0.97, CLa=5.015, CD0=0.029, Cem0=0.075, Cema=-0.463966, CYb=-0.258244, Cellb=-0.039250, Cenb=0.100826, CDp=0.0, CYp=0.065861, CLp=0.0, Cellp=-0.487407, Cemp=0.0, Cenp=-0.040416, CDq=0.055166, CYq=0.0, CLq=7.971792, Cellq=0.0, Cemq=-12.140140, Cenq=0.0, CDr=0.0, CYr=0.230299, CLr=0.0, Cellr=0.078165, Cemr=0.0, Cenr=-0.089947, alpha_stall=0.3391428111, CD_fp_k1 = -0.224, CD_fp_k2 = -0.115, area=0.15, mac=0.22, rho=1.2041, M = 15):
        epsilon  = 1e-8
        span = np.sqrt(AR* area)
        vehicle_velocity = np.array([vehicle_velocity[1], vehicle_velocity[0], -vehicle_velocity[2]])
        u = vehicle_velocity + epsilon
        
        q_ned_gz = np.array([0, np.sqrt(2)/2 + epsilon,  np.sqrt(2)/2 + epsilon, 0])
        q_attitude_fw_gz = quat_mult(q_ned_gz, vehicle_quat)


        R_attitude_fw_gz_vec = np.array([
            1 - 2 * (q_attitude_fw_gz[2] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] - q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[2]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[2] + q_attitude_fw_gz[0] * q_attitude_fw_gz[3]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[3] ** 2), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[1]),
            2 * (q_attitude_fw_gz[1] * q_attitude_fw_gz[3] - q_attitude_fw_gz[0] * q_attitude_fw_gz[2]), 2 * (q_attitude_fw_gz[2] * q_attitude_fw_gz[3] + q_attitude_fw_gz[0] * q_attitude_fw_gz[1]), 1 - 2 * (q_attitude_fw_gz[1] ** 2 + q_attitude_fw_gz[2] ** 2)
        ])

        # Velocity in the LD plane
        R_attitude_fw_gz_1 = np.array([R_attitude_fw_gz_vec[1], R_attitude_fw_gz_vec[4], R_attitude_fw_gz_vec[7]]) 
        # velInLDPlane = u - np.dot(u, R_attitude_fw_gz_1) * R_attitude_fw_gz_1
        velInLDPlane = u - np.dot(u, R_attitude_fw_gz_1) * R_attitude_fw_gz_1
        speedInLDPlane = np.sqrt(np.dot(velInLDPlane, velInLDPlane))
        # Stability axes
        stability_x_axis = velInLDPlane / (speedInLDPlane)
        # stability_x_axis = stability_x_axis.reshape(3,)
        stability_y_axis = R_attitude_fw_gz_1 
        stability_z_axis = np.cross(stability_x_axis, stability_y_axis)
        

        stabx_proj_bodyx = np.dot(stability_x_axis, np.array([R_attitude_fw_gz_vec[0], R_attitude_fw_gz_vec[3], R_attitude_fw_gz_vec[6]]) )
        stabx_proj_bodyz = np.dot(stability_x_axis, np.array([R_attitude_fw_gz_vec[2], R_attitude_fw_gz_vec[5], R_attitude_fw_gz_vec[8]]) )
        
        # Angle of attack and sideslip angle
        alpha = np.arctan2(stabx_proj_bodyz, stabx_proj_bodyx) # Prevent division by zero
        #sideslip angle
        velSW = np.dot(u, np.array([R_attitude_fw_gz_vec[1], R_attitude_fw_gz_vec[4], R_attitude_fw_gz_vec[7]]))
        velFW = np.dot(u, np.array([R_attitude_fw_gz_vec[0], R_attitude_fw_gz_vec[3], R_attitude_fw_gz_vec[6]]))
        beta = np.arctan2(velSW, velFW)
        sigma = (1+np.exp(-1* M*(alpha- alpha_stall))+np.exp( M*(alpha+ alpha_stall)))/((1+np.exp(-1* M*(alpha- alpha_stall)))*(1+np.exp( M*(alpha+ alpha_stall))))
        dyn_press = 0.5* rho*speedInLDPlane*speedInLDPlane
        half_rho_vel = 0.5* rho*speedInLDPlane

        CL = (1-sigma)*( CL0+ CLa*alpha)+sigma*(2*(alpha/np.abs(alpha))*np.sin(alpha)*np.sin(alpha)*np.cos(alpha))
        # lift = (CL* dyn_press+( CLp*vehicle_angular_velocity[0]* span/2*half_rho_vel)+( CLq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( CLr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area*(-1*stability_z_axis))
        
        CD_fp = 2/(1+np.exp( CD_fp_k1+ CD_fp_k2*(np.max([ AR,1/ AR]))))
        CD = (1-sigma)*( CD0+(CL*CL)/(np.pi* AR* eff))+sigma*np.abs(CD_fp*(0.5-0.5*np.cos(2*alpha)))

        #compute drag at ref_pt
        # drag = (CD*dyn_press+( CDp*vehicle_angular_velocity[0]* span/2*half_rho_vel)+( CDq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( CDr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area*(-1*stability_x_axis))
        #compute forces considering every part of the dynamics
        lift = (CL* dyn_press)*(area*(-1*stability_z_axis))
        drag = (CD*dyn_press)*(area*(-1*stability_x_axis))
   
        # compute side force
        Cy =   CYb*beta
        # side_force = (Cy*dyn_press+( CYp*vehicle_angular_velocity[0]* span/2*half_rho_vel)+( CYq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( CYr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area*stability_y_axis)
        side_force = (Cy*dyn_press)*(area*stability_y_axis)

        Cem = 0

        pm = ((Cem*dyn_press)+( Cemp*vehicle_angular_velocity[0]* span/2*half_rho_vel)+( Cemq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( Cemr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area* mac*np.array([R_attitude_fw_gz_vec[1], R_attitude_fw_gz_vec[4], R_attitude_fw_gz_vec[7]]))
        #compute roll moment coefficient, Cell
        #start with angle of attack, sideslip and control terms
        #Cell =  Cellb*beta
        Cell = 0
        rm = ((Cell*dyn_press)+( Cellp*(vehicle_angular_velocity[0]* span/2*half_rho_vel))+( Cellq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( Cellr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area* span*np.array([R_attitude_fw_gz_vec[0], R_attitude_fw_gz_vec[3], R_attitude_fw_gz_vec[6]]))
        #compute yaw moment coefficient, Cen
       # Cen =  Cenb*beta
        Cen = 0
        ym = ((Cen*dyn_press)+( Cenp*(vehicle_angular_velocity[0]* span/2*half_rho_vel))+( Cenq*(vehicle_angular_velocity[1]* mac/2)*half_rho_vel)+( Cenr*(vehicle_angular_velocity[2]* span/2)*half_rho_vel))*( area* span*np.array([R_attitude_fw_gz_vec[2], R_attitude_fw_gz_vec[5], R_attitude_fw_gz_vec[8]]))
    
        aero_force = lift + drag + side_force
        aero_moment = pm + rm + ym
        #transform aero force to inertial NED frame
        aero_force = np.array([aero_force[1], aero_force[0], -aero_force[2]])
        #transform aero moment to NED
        aero_moment = np.array([aero_moment[1], aero_moment[0], -aero_moment[2]])
        #transform aero moment to body frame
       # print(aero_moment)
        aero_moment = rotate_quat(q = quat_conj(vehicle_quat), v = aero_moment)
    

        return aero_force, aero_moment  
    
    def compute_reduced_attitude_control_fixed_wing(self,current_attitude, r_gain, p_gain):
        """Computes reduced attitude control for a fixed-wing frame (tailsitter) using quaternion error."""
        
        # Normalize the desired acceleration (desired thrust direction)
        eBcmd_x = np.array([0, 0, -1]) 
        
        # Rotate the current attitude by pi/2 over the y-axis to account for the frame difference
        # current_attitude = rotate_by_y_axis(current_attitude, np.pi / 2)
        
        # Calculate the current body x-axis (thrust direction) from the quaternion

        eBx = np.array([
            current_attitude[0]**2+current_attitude[1]**2-current_attitude[2]**2-current_attitude[3]**2,
            2*(current_attitude[1]*current_attitude[2]+current_attitude[0]*current_attitude[3]),
            2*(current_attitude[1]*current_attitude[3]-current_attitude[0]*current_attitude[2])
        ])
        # Calculate the angle (alpha) between current and desired thrust direction
        
        alpha = np.arccos(np.clip(np.dot(eBx, eBcmd_x), -1.0, 1.0))
        
        # Calculate the rotation axis (k) between the two directions
        k = np.cross(eBx, eBcmd_x)
        if np.linalg.norm(k) > 0:
            k = k / np.linalg.norm(k)  # Normalize the axis
        q_error_red = np.array([np.cos(alpha / 2), k[0] * np.sin(alpha / 2), k[1] * np.sin(alpha / 2), k[2] * np.sin(alpha / 2)])
        if np.abs(alpha) < 1e-6:
            q_error_red = np.array([1, 0, 0, 0]) 
        # Compute the reduced error quaternion qe,red
        
        # Calculate the desired attitude by applying the error quaternion to the current attitude
        qcmd_red = quat_mult(current_attitude, q_error_red)
        
        # Convert quaternion error to roll and pitch rates
        z_des = 2 * r_gain * q_error_red[1] if q_error_red[0] >= 0 else -2 * r_gain * q_error_red[1]
        y_des = 2 * p_gain * q_error_red[2] if q_error_red[0] >= 0 else -2 * p_gain * q_error_red[2]

        return z_des, y_des
    
    
    def attitude_quaternion(self, vehicle_quat, vehicle_velocity, vehicle_angular_velocity, wind_velocity, dt):
        if(np.linalg.norm(vehicle_quat) > 1.1):
            print('######################################CqUATERNION ERRADO###################################')
        e_cmd_x = np.array([0,0,-1])
        ex_cur = np.array([
            vehicle_quat[0]**2+vehicle_quat[1]**2-vehicle_quat[2]**2-vehicle_quat[3]**2,
            2*(vehicle_quat[1]*vehicle_quat[2]+vehicle_quat[0]*vehicle_quat[3]),
            2*(vehicle_quat[1]*vehicle_quat[3]-vehicle_quat[0]*vehicle_quat[2])
        ])       
        alpha = np.arccos(np.clip(np.dot(ex_cur, e_cmd_x), -1.0, 1.0))

        if np.abs(alpha) < 1e-8:
            q_erp = np.array([1,0,0,0])
        else:
            if np.linalg.norm(np.cross(ex_cur, e_cmd_x)) > 1e-8:
                n = np.cross(ex_cur, e_cmd_x)/np.linalg.norm(np.cross(ex_cur, e_cmd_x))
            else:
                n = np.array([0,0,0])
            n_b = rotate_quat(q = quat_conj(vehicle_quat), v = n)
            q_erp = np.array([np.cos(alpha/2), n_b[0]*np.sin(alpha/2), n_b[1]*np.sin(alpha/2), n_b[2]*np.sin(alpha/2)])

        pitch_rate = 2*q_erp[2] if q_erp[0] >= 0 else -2*q_erp[2]
        yaw_rate = 2*q_erp[3] if q_erp[0] >= 0 else -2*q_erp[3]
        qex = quat_mult(quat_conj(quat_mult(vehicle_quat, q_erp)),self.q_d)
        x_des = 0
        eyc = np.array([0, np.cos(x_des), np.sin(x_des)])
        ezc = np.array([0, -np.sin(x_des), np.cos(x_des)])
        
        if np.linalg.norm(np.cross(eyc, e_cmd_x)) < 1e-8:
            roll_rate = 0
        else:
            e_cmd_z = np.cross(eyc, e_cmd_x)/np.linalg.norm(np.cross(eyc, e_cmd_x))
            e_cmd_y = np.cross(e_cmd_z, e_cmd_x)/np.linalg.norm(np.cross(e_cmd_z, e_cmd_x))
            
            R_cmd = np.column_stack((e_cmd_x, e_cmd_y, e_cmd_z))
            q_cmd = rot_to_quat(R_cmd)
            
            qex = quat_mult(quat_conj(quat_mult(vehicle_quat, q_erp)), q_cmd)
            roll_rate = 2*qex[1] if qex[0] >= 0 else -2*qex[1]
        
        omega_d = np.array([roll_rate, pitch_rate, yaw_rate])
        omega_d = omega_d * self.attitude_p_gains 
        
        if not(self.lock_position):
             omega_d[0] = 0
    
        
        print('omega_d', omega_d)
        #angular_rate_loop
        _, aero_moment = self.estimate_aero_quat_np(vehicle_velocity = vehicle_velocity, 
                                                    vehicle_quat = vehicle_quat, 
                                                    vehicle_angular_velocity = vehicle_angular_velocity, 
                                                    wind = wind_velocity)
        
     #   omega_d = np.clip(omega_d, -np.array([3.5, 6.0, 5.0]), np.array([3.5, 6.0, 6.0]))
        w_error = -vehicle_angular_velocity + omega_d
        if not(self.lock_position):
             w_error[0] = 0
        #feed-forward
        ff = -aero_moment+np.cross(vehicle_angular_velocity, np.diag([self.Ixx, self.Iyy, self.Izz]))@vehicle_angular_velocity
        pid = self.attitude_rates_p_gains*w_error + self.attitude_rates_i_gains*self.e_attitude_rates_int + self.attitude_rates_d_gains * (vehicle_angular_velocity - self.last_attitude_rates)/dt 
        #print('pid', pid)
        #print('ff', ff)
        
        #if self.lock_position == False:
         #   pid[0] = 0
        Mt = pid + aero_moment - np.cross(vehicle_angular_velocity, np.diag([self.Ixx, self.Iyy, self.Izz]))@vehicle_angular_velocity
        
        
        if abs(Mt[0]) < self.max_torque_x:
            self.e_attitude_rates_int[0] += w_error[0]*dt    
        if abs(Mt[1]) < self.max_torque_y:
            self.e_attitude_rates_int[1] += w_error[1]*dt
        if abs(Mt[2]) < self.max_torque_z:
            self.e_attitude_rates_int[2] += w_error[2]*dt

        #update last error
        self.e_attitude_rates_last = w_error
        self.last_attitude_rates = vehicle_angular_velocity
        return Mt


    def altitude_controller(self, h, h_d,vehicle_velocity, vehicle_angular_velocity, vehicle_q, wind_velocity, dt):

        f_al, _ = self.estimate_aero_quat_np(vehicle_velocity = vehicle_velocity, 
                                             vehicle_quat = vehicle_q, 
                                             vehicle_angular_velocity = vehicle_angular_velocity, 
                                             wind = wind_velocity)
        #f_al = np.array([0,0,0])
        ff =  (-self.vehicle_mass*self.g - f_al[2])
        pid = 0
        h_e = h - h_d 
        #pid with derivative term on the error
        # pid = self.height_p_gains*h_e + self.height_i_gains*self.e_height_int+self.height_d_gains*(h_e - self.h_error_last)/dt
        #pid with derivative term on the ouput
        pid = self.height_p_gains*h_e + self.height_i_gains*self.e_height_int+self.height_d_gains*(h - self.last_h)/dt
        pid = -pid
        
        if self.lock_position == False: #if it is locked, the controller will hold the altitude. Otherwise, it will only compensate the gravity force
            pid = 0
            self.h_d = h
        self.h_error_last = h_e
        self.last_h = h

        ftx = pid+ff

        #ftx = ftx/(vehicle_attitude[2,0])
        ftx = ftx/(2 * (vehicle_q[1] * vehicle_q[3] - vehicle_q[0] * vehicle_q[2])) 
   #     if not(self.last_ftx is None):
    #        if np.abs(ftx - self.last_ftx) > 20:
     #           ftx = self.last_ftx 
      #  self.last_ftx = ftx

        return ftx
    
    def controller_timer_callback(self) -> None:
        self.get_time()
        self.counter+=1
        if self.counter == 99:
            self.clean_attitude_error()
            self.clean_attitude_error()
        if self.dt > 0:
            # Shift the array to make space for the new message and store the new angular velocities
            self.omega_array = np.roll(self.omega_array, -1, axis=0)
            self.omega_array[-1] = self.angular_velocity[1:]

            
            # Calculate the current body x-axis (thrust direction) from the quaternion
            e_cmd_x = np.array([0,0,-1])
            ex_cur = np.array([
                self.q_odometry[0]**2+self.q_odometry[1]**2-self.q_odometry[2]**2-self.q_odometry[3]**2,
                2*(self.q_odometry[1]*self.q_odometry[2]+self.q_odometry[0]*self.q_odometry[3]),
                2*(self.q_odometry[1]*self.q_odometry[3]-self.q_odometry[0]*self.q_odometry[2])
            ])               
            alpha = np.arccos(np.clip(np.dot(ex_cur, e_cmd_x), -1.0, 1.0))
           # print(alpha)

            # Shift the array to make space for the new message and store the new angular velocities
            self.inclination_error = np.roll(self.inclination_error, -1, axis=0)
            self.inclination_error[-1] = alpha
            
            # #if np.all(self.inclination_error < np.deg2rad(10)) and np.all(self.omega_array < 6):
            if np.mean(np.abs(self.inclination_error)) < np.deg2rad(20) and np.mean(np.abs(self.omega_array < 8)) and self.lock_position == False:
                self.lock_position = True
                self.h_d = self.position[2]
                self.position_locked_once = True
                
                print('position locked')
                
                if np.abs(alpha) < 1e-8:
                    q_erp = np.array([1,0,0,0])
                else:
                    if np.linalg.norm(np.cross(ex_cur, e_cmd_x)) > 1e-8:
                        n = np.cross(ex_cur, e_cmd_x)/np.linalg.norm(np.cross(ex_cur, e_cmd_x))
                    else:
                        n = np.array([0,0,0])
                    n_b = rotate_quat(q = quat_conj(self.q_odometry), v = n)
                    q_erp = np.array([np.cos(alpha/2), n_b[0]*np.sin(alpha/2), n_b[1]*np.sin(alpha/2), n_b[2]*np.sin(alpha/2)])

                qex = quat_mult(quat_conj(quat_mult(self.q_odometry, q_erp)),self.q_d)
                print('old qd: ', self.q_d) 
                #self.q_d = quat_mult(qex, np.array([np.sqrt(2)/2, 0, -np.sqrt(2)/2, 0]))
                print('new q_d: ', self.q_d)
                
                
            elif np.mean(np.abs(self.inclination_error)) > np.deg2rad(20)  and self.lock_position == True:# and self.counter_lock < 2:
                self.lock_position = False
                self.counter_lock +=1
                self.clean_height_error()
                self.position_locked_once = False
                print('position unlocked')
                
            
            thrust_d = self.altitude_controller(h = self.position[2], 
                                                h_d = self.h_d, 
                                                vehicle_velocity = self.velocity, 
                                                vehicle_angular_velocity = self.angular_velocity, 
                                                vehicle_q = self.q_odometry, 
                                                wind_velocity = self.wind_velocity, 
                                                dt = self.dt)
            
            torque_d= self.attitude_quaternion(vehicle_quat = self.q_odometry, 
                                               vehicle_velocity = self.velocity, 
                                               vehicle_angular_velocity = self.angular_velocity, 
                                               wind_velocity = self.wind_velocity,
                                               dt = self.dt)
            controller_output = np.array([thrust_d, torque_d[0], torque_d[1], torque_d[2]]) 
           
            self.publish_commands(controller_output)
            self.publish_ref()
    
            
    def publish_commands(self, controller_output):
        msg = ThrustTorqueSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        msg.thrust = float(controller_output[0])
        msg.tx = float(controller_output[1])
        msg.ty = float(controller_output[2])
        msg.tz = float(controller_output[3])
        msg.positionlock = self.lock_position
        self.control_publisher.publish(msg)
    
    def publish_ref(self):
        msg = CustomReference()
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        msg.altitude_ref = self.h_d
        msg.q0_ref = self.q_d[0]
        msg.q1_ref = self.q_d[1]
        msg.q2_ref = self.q_d[2]
        msg.q3_ref = self.q_d[3]
        self.ref_publisher.publish(msg)

    
def main(args=None) -> None:
    print('Starting PID node...')
    rclpy.init(args=args)
    node = PID_Flight_controller() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

        
        
        

        