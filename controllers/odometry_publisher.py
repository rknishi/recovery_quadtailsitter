#!/usr/bin/env python3
import rclpy
import numpy as np
import time
from custom_msg.msg import FwOdometry
from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleOdometry, VehicleAngularVelocity
from common import quat_mult  # Ensure this is available
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class OdometryPublisher(Node):  
    def __init__(self) -> None:
        super().__init__('odometry_node')

        self.ground_truth = True

        self.vehicle_x = 0
        self.vehicle_y = 0
        self.vehicle_z = 0

        self.vehicle_vx = 0
        self.vehicle_vy = 0
        self.vehicle_vz = 0

        self.vehicle_q0 = 0
        self.vehicle_q1 = 0
        self.vehicle_q2 = 0
        self.vehicle_q3 = 0

        self.vehicle_rollspeed = 0
        self.vehicle_pitchspeed = 0
        self.vehicle_yawspeed = 0

        self.position_received = False
        self.attitude_received = False
        self.angular_velocity_received = False
        self.odometry_received = False

        self.time_position = 0
        self.time_attitude = 0
        self.time_angular_velocity = 0
        self.time_odometry = 0

        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribers
        # Use ground_truth data
        if self.ground_truth:
            self.vehicle_position_subscriber = self.create_subscription(
                VehicleLocalPosition, '/fmu/out/vehicle_local_position_groundtruth', self.vehicle_position_callback, qos_profile)
            self.vehicle_attitude_subscriber = self.create_subscription(
                VehicleAttitude, '/fmu/out/vehicle_attitude_groundtruth', self.vehicle_attitude_callback, qos_profile)
            self.vehicle_angular_velocity_subscriber = self.create_subscription(
                VehicleAngularVelocity, '/fmu/out/vehicle_angular_velocity_groundtruth', self.vehicle_angular_velocity_callback, qos_profile)
        else:
            # Use odometry/estimator data
            self.vehicle_odometry_subscriber = self.create_subscription(
                VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_callback, qos_profile)

        # Publishers
        self.fw_odometry_publisher = self.create_publisher(FwOdometry, '/fw_odometry', qos_profile)

    def vehicle_odometry_callback(self, vehicle_odometry):  
        """Callback function for vehicle odometry subscriber"""
        vehicle_position = vehicle_odometry.position
        self.vehicle_x = vehicle_position[0]
        self.vehicle_y = vehicle_position[1]
        self.vehicle_z = vehicle_position[2]

        vehicle_velocity = vehicle_odometry.velocity
        self.vehicle_vx = vehicle_velocity[0]
        self.vehicle_vy = vehicle_velocity[1]
        self.vehicle_vz = vehicle_velocity[2]


        # Vehicle attitude always in MC frame -> change to conventional FW frame
        q_fw_to_drone = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
        vehicle_attitude = quat_mult(vehicle_odometry.q, q_fw_to_drone)
        self.vehicle_q0 = vehicle_attitude[0]
        self.vehicle_q1 = vehicle_attitude[1]
        self.vehicle_q2 = vehicle_attitude[2]
        self.vehicle_q3 = vehicle_attitude[3]

        # Angular velocity always in drone frame -> change to conventional FW frame
        mc_angular_velocity = vehicle_odometry.angular_velocity
        self.vehicle_rollspeed = -mc_angular_velocity[2]  # Roll
        self.vehicle_pitchspeed = mc_angular_velocity[1]   # Pitch
        self.vehicle_yawspeed = mc_angular_velocity[0]   # Yaw

        self.odometry_received = True
        self.time_odometry = vehicle_odometry.timestamp
        self.check_and_publish_odometry()

    def vehicle_position_callback(self, vehicle_position_msg):
        self.vehicle_x = vehicle_position_msg.x
        self.vehicle_y = vehicle_position_msg.y
        self.vehicle_z = vehicle_position_msg.z


        self.vehicle_vx = vehicle_position_msg.vx
        self.vehicle_vy = vehicle_position_msg.vy
        self.vehicle_vz = vehicle_position_msg.vz

        self.position_received = True
        self.time_position = vehicle_position_msg.timestamp
        self.check_and_publish_odometry()

    def vehicle_attitude_callback(self, vehicle_attitude_msg):
        # Vehicle attitude always in MC frame -> change to conventional FW frame
        q_fw_to_drone = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2, 0])
        vehicle_attitude = quat_mult(vehicle_attitude_msg.q, q_fw_to_drone)
        self.vehicle_q0 = vehicle_attitude[0]
        self.vehicle_q1 = vehicle_attitude[1]
        self.vehicle_q2 = vehicle_attitude[2]
        self.vehicle_q3 = vehicle_attitude[3]

        self.attitude_received = True
        self.time_attitude = vehicle_attitude_msg.timestamp
        self.check_and_publish_odometry()

    def vehicle_angular_velocity_callback(self, vehicle_angular_msg):
        # Angular velocity always in drone frame -> change to conventional FW frame
        mc_angular_velocity = vehicle_angular_msg.xyz
        self.vehicle_rollspeed = -mc_angular_velocity[2]  # Roll
        self.vehicle_pitchspeed = mc_angular_velocity[1]   # Pitch
        self.vehicle_yawspeed = mc_angular_velocity[0]   # Yaw  
        self.angular_velocity_received = True
        self.time_angular_velocity = vehicle_angular_msg.timestamp
        self.check_and_publish_odometry()

    def check_and_publish_odometry(self):
        if self.position_received and self.attitude_received and self.angular_velocity_received:
            self.time_odometry = max(self.time_position, self.time_attitude, self.time_angular_velocity)
            self.publish_odometry()
            self.position_received = False
            self.attitude_received = False
            self.angular_velocity_received = False

    def publish_odometry(self):
        fw_odometry = FwOdometry()
        print('Publishing odometry')
        fw_odometry.timestamp = int(self.get_clock().now().nanoseconds)
        fw_odometry.x = float(self.vehicle_x)
        fw_odometry.y = float(self.vehicle_y)
        fw_odometry.z = float(self.vehicle_z)
        fw_odometry.vx = float(self.vehicle_vx)
        fw_odometry.vy = float(self.vehicle_vy)
        fw_odometry.vz = float(self.vehicle_vz)
        fw_odometry.q0 = float(self.vehicle_q0)
        fw_odometry.q1 = float(self.vehicle_q1)
        fw_odometry.q2 = float(self.vehicle_q2)
        fw_odometry.q3 = float(self.vehicle_q3)
        fw_odometry.rollspeed = float(self.vehicle_rollspeed)
        fw_odometry.pitchspeed = float(self.vehicle_pitchspeed)
        fw_odometry.yawspeed = float(self.vehicle_yawspeed)
        self.fw_odometry_publisher.publish(fw_odometry)


def main(args=None) -> None:
    print('Starting odometry node...')
    rclpy.init(args=args)
    node = OdometryPublisher() 
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
