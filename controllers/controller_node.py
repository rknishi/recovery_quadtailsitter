#!/usr/bin/env python3
import rclpy
import numpy as np
import yaml
import time
from px4_msgs.msg import VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleCommand, VehicleAttitudeSetpoint, OffboardControlMode
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from custom_msg.msg import FwOdometry, ThrustTorqueSetpoint

        
class ControllerNode(Node):
    def __init__(self) -> None:
        super().__init__('controller_node')
        self.init_parameters()
   
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        #Subscribers
        self.control_thrust_subscriber = self.create_subscription(
            ThrustTorqueSetpoint, '/controller_output', self.controller_callback, qos_profile
        )

        #Publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        
        #just for simulation, make a counter to determine starting time of controller
        self.counter = 0

        #Timer
        self.offboard_timer = self.create_timer(0.5, self.offboard_timer_callback)        

    def init_parameters(self):
        with open('parameters.yaml', 'r') as file:
            params = yaml.safe_load(file)
            
        self.motor_constant = params['vehicle']['motor_constant']
        self.max_rotor_vel = params['vehicle']['max_rotor_vel']   
        self.max_thrust_one_motor = self.max_rotor_vel**2 * self.motor_constant        
        self.max_thrust = self.max_thrust_one_motor * 4 
        self.max_torque_x = params['limits']['max_x_torque']  
        self.max_torque_y = params['limits']['max_y_torque'] 
        self.max_torque_z = params['limits']['max_z_torque'] 
         
    def controller_callback(self, msg):
        """Callback function for vehicle_thrust topic subscriber"""
        #self.timestamp =
        thrust = msg.thrust #thrust in fw frame not normalized
        #avoid negative thrust
        if thrust < 0:
            thrust = 0

        #torques in fw frame not normalized
        torque_x = msg.tx  
        torque_y = msg.ty
        torque_z = msg.tz

        #normalize thrust
        thrust_norm = np.clip(np.sqrt(thrust/self.motor_constant/4)/(self.max_rotor_vel), 0.35, 1)
        #normalize torques
        torque_norm = np.clip(np.array([torque_x, torque_y, torque_z])/np.array([self.max_torque_x, self.max_torque_y, self.max_torque_z]), -1, 1)

        thrust_sp_msg = VehicleThrustSetpoint()
        thrust_sp_msg.timestamp = int(self.get_clock().now().nanoseconds)
        thrust_sp_msg.xyz[0] = 0  
        thrust_sp_msg.xyz[1] = 0
        thrust_sp_msg.xyz[2] = -thrust_norm


        torque_sp_msg = VehicleTorqueSetpoint()
        torque_sp_msg.timestamp = int(self.get_clock().now().nanoseconds)
        #change to fw frame
            
        #nmpc
        torque_sp_msg.xyz[0] = torque_norm[2]
        torque_sp_msg.xyz[1] = torque_norm[1]
        torque_sp_msg.xyz[2] = -torque_norm[0] 
        

        
        #limit the amount to decimal cases
        
        
        self.counter+=1
        
       
            
        
        if self.counter == 3:
            self.arm()
            self.engage_offboard_mode()
            time.sleep(1.2)
        
        # elif self.counter >3 and self.counter <= 225:
        elif self.counter == 4:
            for i in range(202):
                thrust_sp_msg.xyz[0] = 0  
                thrust_sp_msg.xyz[1] = 0
                thrust_sp_msg.xyz[2] = -0.1
                torque_sp_msg.xyz[0] = 0
                torque_sp_msg.xyz[1] = 0
                torque_sp_msg.xyz[2] = 0
                self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
                self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
                print('Off')
                # time.sleep(0.01)
                time.sleep(0.2)
            #     self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
            #     self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
            #     time.sleep(0.2)
            # self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
            # self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
            # #time.sleep(4.0)
            # self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
            # self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
            # #time.sleep(4.0)
            # self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
            # self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
            # #time.sleep(5.9)
            # self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
            # self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)
        # elif self.counter > 225:
        else:
            print(f'Cnt {thrust_norm:.3f} {torque_norm[0]:.3f} {torque_norm[1]:.3f} {torque_norm[2]:.3f}')
            # thrust_sp_msg.xyz[0] = 0  
            # thrust_sp_msg.xyz[1] = 0
            # thrust_sp_msg.xyz[2] = 0
            # torque_sp_msg.xyz[0] = 0
            # torque_sp_msg.xyz[1] = 0.001
            # torque_sp_msg.xyz[2] = 0
        self.vehicle_thrust_setpoint_publisher.publish(thrust_sp_msg)
        self.vehicle_torque_setpoint_publisher.publish(torque_sp_msg)

        
            
    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')
    
    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")


    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        self.vehicle_command_publisher.publish(msg)

    def offboard_timer_callback(self) -> None:
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.body_rate = False
        msg.direct_actuator = False
        msg.thrust_and_torque = True
        msg.attitude = False
        msg.timestamp = int(self.get_clock().now().nanoseconds)
        self.offboard_control_mode_publisher.publish(msg)
    


def main(args=None) -> None:
    print('Starting controller node...')
    rclpy.init(args=args)
    node = ControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()



