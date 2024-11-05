# Recovery Quadtailsitter

This is a project to simulate recovery controllers for a quadrotor tailsitter without control surfaces. Two different controllers are designed: one based on PID and another on NMPC.

To use this repository, you must have PX4 SITL simulator, ROS2, and Gazebo Classic installed in your computer. Follow the instructions on:
 - [Gazebo Classic PX4](https://docs.px4.io/main/en/sim_gazebo_classic/)

- [ROS2 PX4](https://docs.px4.io/main/en/ros2/user_guide.html#ros2-launch)

After installing the simulator, you must clone/download the repository into your local folder and add/merge the folders to the following packages within the PX4 SITL catkin directory:

- **quadtailsitter_link_velocity:** merge with /Tools/simulation/gazebo-classic/sitl_gazebo-classic/models
- **airframes:** merge with /ROMFS/px4fmu_common/init.d-posix/airframes
- **custom_msg:** merge with /msg or add to your ROS2 workspace
- **plugin:** in /Tools/simulation/gazebo-classic/sitl_gazebo-classic/
     - add package.xml and CMakeLists.txt
     - /src/ add set_vel_plugin.cpp
     - /include/ add set_vel_plugin.h



