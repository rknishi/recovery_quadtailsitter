#ifndef _SET_VEL_PLUGIN_HH_
#define _SET_VEL_PLUGIN_HH_

#include <gazebo/common/PID.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/util/system.hh>
#include <gazebo/sensors/sensors.hh>
#include <ignition/math.hh>
#include <development/mavlink.h>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/gazebo.hh>
#include "CommandMotorSpeed.pb.h"

namespace gazebo
{     

    typedef const boost::shared_ptr<const mav_msgs::msgs::CommandMotorSpeed> CommandMotorSpeedPtr;

    //default PID gains
    static double kPIDXP = 50.0;
    static double kPIDXI = 0.0;
    static double kPIDXD = 0.0;
    static double kPIDXIMax = 0.0;
    static double kPIDXIMin = 0.0;
    static double kPIDXCmdMax = 65;
    static double kPIDXCmdMin = -65;

    static double kPIDYP = 50.0;
    static double kPIDYI = 0.0;
    static double kPIDYD = 0.0;
    static double kPIDYIMax = 0.0;
    static double kPIDYIMin = 0.0;
    static double kPIDYCmdMax = 50;
    static double kPIDYCmdMin = -50;

    static double kPIDZP = 50.0;
    static double kPIDZI = 0.0;
    static double kPIDZD = 0.0;
    static double kPIDZIMax = 0.0;
    static double kPIDZIMin = 0.0;
    static double kPIDZCmdMax = 1000;
    static double kPIDZCmdMin = -1000;

    static double kPIDRollP = 3.0; //1.2
    static double kPIDRollI = 1.623;
    static double kPIDRollD = 0.5;
    static double kPIDRollIMax = 3.0;
    static double kPIDRollIMin = -3.0;
    static double kPIDRollCmdMax = 3;
    static double kPIDRollCmdMin = -3;

    static double kPIDPitchP = 4.1;
    static double kPIDPitchI = 01.02;
    static double kPIDPitchD = 0.44;
    static double kPIDPitchIMax = 3.0;
    static double kPIDPitchIMin = -3.0;
    static double kPIDPitchCmdMax = 4.775;
    static double kPIDPitchCmdMin = -4.775;

    static double kPIDYawP = 2.25; //1.2
    static double kPIDYawI = 01.41;
    static double kPIDYawD = 0.5;
    static double kPIDYawIMax = 3.0;
    static double kPIDYawIMin = -3.0;
    static double kPIDYawCmdMax = 3;
    static double kPIDYawCmdMin = -3;


    enum LaunchStatus {
        VEHICLE_STANDBY,
        VEHICLE_INLAUNCH,
        VEHICLE_LAUNCHED
    };



    class GAZEBO_VISIBLE SetVelPlugin : public ModelPlugin
    {
        public: 
            SetVelPlugin();
            ~SetVelPlugin();
            virtual void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf);
            // virtual void Init();

        private:
            void OnUpdate(const common::UpdateInfo&);
            void VelocityCallback(CommandMotorSpeedPtr &rot_velocities);

            sdf::ElementPtr sdf;

            std::string namespace_;
            physics::ModelPtr model_;
            physics::WorldPtr world_;
            physics::LinkPtr link_;

            event::ConnectionPtr _updateConnection;

            LaunchStatus launch_status_ = VEHICLE_STANDBY;

            double xSetpoint_ = 2.0;
            double ySetpoint_ = 3.0;
            double zSetpoint_ = 4.0;
            double zSetpoint2_ = 4.0;

            double rollSetpoint_ = 0.0;
            double pitchSetpoint_ = 0.0;
            double yawSetpoint_ = 0.0;
            common::PID xPid;
            common::PID yPid;
            common::PID zPid;
            common::PID pitchPid;
            common::PID rollPid;
            common::PID yawPid;
            common::Time lastUpdateTime;

            common::Time trigger_time_;

            double max_rot_velocity_ = 3500;
            double ref_motor_rot_vel_ = 0.0;
            double arm_rot_vel_ = 100;
            double launch_duration_ = 0.01;
            double z_force_duration_ = 0.01;

            int motor_number_;

            std::string trigger_sub_topic_ = "/gazebo/command/motor_speed";

            transport::NodePtr node_;
            transport::SubscriberPtr trigger_sub_;
        
    };

}
#endif