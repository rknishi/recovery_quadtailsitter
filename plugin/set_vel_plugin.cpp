#include "set_vel_plugin.h"
#include "common.h"
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/transport.hh>

using namespace std;

namespace gazebo{
GZ_REGISTER_MODEL_PLUGIN(SetVelPlugin)

SetVelPlugin::SetVelPlugin()
// :status("closed")
{
  /// defaults if sdf xml doesn't contain any pid gains
  this->xPid.Init(kPIDXP, kPIDXI, kPIDXD, kPIDXIMax, kPIDXIMin, kPIDXCmdMax, kPIDXCmdMin);
  this->yPid.Init(kPIDYP, kPIDYI, kPIDYD, kPIDYIMax, kPIDYIMin, kPIDYCmdMax, kPIDYCmdMin);
  this->zPid.Init(kPIDZP, kPIDZI, kPIDZD, kPIDZIMax, kPIDZIMin, kPIDZCmdMax, kPIDZCmdMin);
  this->pitchPid.Init(kPIDPitchP, kPIDPitchI, kPIDPitchD, kPIDPitchIMax, kPIDPitchIMin, kPIDPitchCmdMax, kPIDPitchCmdMin);
  this->rollPid.Init(kPIDRollP, kPIDRollI, kPIDRollD, kPIDRollIMax, kPIDRollIMin, kPIDRollCmdMax, kPIDRollCmdMin);
  this->yawPid.Init(kPIDYawP, kPIDYawI, kPIDYawD, kPIDYawIMax, kPIDYawIMin, kPIDYawCmdMax, kPIDYawCmdMin);
}

SetVelPlugin::~SetVelPlugin()
{
  _updateConnection->~Connection();
}

void SetVelPlugin::Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
{
  model_ = _model;
  world_ = model_ -> GetWorld();
  this->sdf = _sdf;

  namespace_.clear();

  //check robot namespace
  if (_sdf->HasElement("robotNamespace"))
    namespace_ = _sdf->GetElement("robotNamespace")->Get<std::string>();
  else
    gzerr << "[gazebo_catapult_plugin] Please specify a robotNamespace.\n";
  
  //check link name
  if(_sdf->HasElement("link_name")){
    sdf::ElementPtr elem = _sdf->GetElement("link_name");
    std::string linkName = elem->Get<std::string>();
    this->link_ = this->model_->GetLink(linkName);

    if(!this->link_) {
      gzerr << "Link with name[" << linkName << "] not found. "
            << "The Catapult plugin will not be able to launch the vehicle\n";
      std::cout << "Link with name[" << linkName << "] not found. "
                << "The Catapult plugin will not be able to launch the vehicle" << std::endl;
    }
    else
      std::cout<<"[gazebo_catapult_plugin] link_name: "<< linkName << std::endl; 
  }

  else {
    gzerr << "[gazebo_catapult_plugin] Default link name is used";
  }

  //check motor number
  if(_sdf->HasElement("motorNumber")){
    motor_number_ = _sdf->GetElement("motorNumber")->Get<int>();
  }
  else
  {
    gzerr << "[gazebo_catapult_plugin] Please specify a motorNumber.\n";
    std::cout << "[gazebo_catapult_plugin] Motor number not specified}\n"; 
  }

  getSdfParam<std::string>(_sdf, "commandSubTopic", trigger_sub_topic_, trigger_sub_topic_); 
  getSdfParam<double>(_sdf, "xSetpoint", xSetpoint_, xSetpoint_);
  getSdfParam<double>(_sdf, "ySetpoint", ySetpoint_, ySetpoint_);
  getSdfParam<double>(_sdf, "zSetpoint", zSetpoint_, zSetpoint_);
  getSdfParam<double>(_sdf, "zSetpoint2", zSetpoint2_, zSetpoint2_);
  getSdfParam<double>(_sdf, "rollSetpoint", rollSetpoint_, rollSetpoint_);
  getSdfParam<double>(_sdf, "pitchSetpoint", pitchSetpoint_, pitchSetpoint_);
  getSdfParam<double>(_sdf, "yawSetpoint", yawSetpoint_, yawSetpoint_);
  getSdfParam<double>(_sdf, "duration", launch_duration_, launch_duration_);
  getSdfParam<double>(_sdf, "z_duration", z_force_duration_ , z_force_duration_);


  // Listen to the update event. This event is broadcast every simulation iteration.
  _updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&SetVelPlugin::OnUpdate, this, _1));

  node_ = transport::NodePtr(new transport::Node());
  node_->Init(namespace_);

  trigger_sub_ = node_->Subscribe("~/" + model_->GetName() + trigger_sub_topic_, &SetVelPlugin::VelocityCallback, this);

}



void SetVelPlugin::OnUpdate(const common::UpdateInfo&)
{
  #if GAZEBO_MAJOR_VERSION >= 9
    common::Time time = world_->SimTime();
  #else
    common::Time time = world_->GetSimTime();
  #endif

  if (time < this->lastUpdateTime) {
    gzerr << "time reset event\n";
    this->lastUpdateTime = time;
    return;
  } else if (time > this->lastUpdateTime) {
    if (ref_motor_rot_vel_ > arm_rot_vel_ && launch_status_ != VEHICLE_LAUNCHED) {
      if (launch_status_ == VEHICLE_STANDBY) {
        trigger_time_ = world_->SimTime();
        launch_status_ = VEHICLE_INLAUNCH;
        std::cout << "[set_velocity_plugin] Launch armed " << std::endl;
      }

      double dt = (time - this->lastUpdateTime).Double();

      // Get current linear velocity
      ignition::math::Vector3 curLinearVel = link_->WorldLinearVel();
      ignition::math::Vector3 curAngularVel = link_->WorldAngularVel();

      // Stage 1: Apply Z force first
      ignition::math::Vector3d worldForce;
      ignition::math::Vector3d worldTorque;

      if (time - trigger_time_ < z_force_duration_) { // Apply Z force for a duration
        double zError = curLinearVel.Z() - this->zSetpoint_;
        double xError = curLinearVel.X() - this->xSetpoint_;

        worldForce.X() = this->xPid.Update(xError, dt);

        worldForce.Z() = this->zPid.Update(zError, dt);

        // Apply Z force only
        this->link_->AddForce(worldForce);

        std::cout << "[set_velocity_plugin] Applying Z force\n";
      } 
      else { 
        // Stage 2: After Z force duration, correct attitude and X/Y positions        
        // Get current orientation (attitude) in quaternions
        ignition::math::Quaterniond curOrientation = link_->WorldPose().Rot();
        ignition::math::Quaterniond desiredOrientation(this->rollSetpoint_, this->pitchSetpoint_, this->yawSetpoint_);

        // Compute orientation error (quaternion difference)
        ignition::math::Quaterniond orientationError = desiredOrientation.Inverse() * curOrientation;

        // Convert quaternion error to axis-angle representation
        ignition::math::Vector3d axis;
        double angle;

        // Normalize the quaternion
        orientationError.Normalize();

        // Calculate the angle
        angle = 2 * acos(orientationError.W());

        // Calculate the axis (safeguard against zero division)
        double s = sqrt(1 - orientationError.W() * orientationError.W());
        if (s < 0.001) { // If s is close to zero, direction of axis is not important
            axis.Set(orientationError.X(), orientationError.Y(), orientationError.Z());
        } else {
            axis.Set(orientationError.X() / s, orientationError.Y() / s, orientationError.Z() / s);
        }
        double rollError = axis.X() * angle;
        double pitchError = axis.Y() * angle;
        double yawError = axis.Z() * angle;


        // Compute X and Y AND z position errors
        double xError = curLinearVel.X() - this->xSetpoint_;
        double yError = curLinearVel.Y() - this->ySetpoint_;
        double zError = curLinearVel.Z() - this->zSetpoint2_;

        // Update PIDs for X, Y, and angular positions (attitude)
        worldForce.X() = this->xPid.Update(xError, dt);
        worldForce.Y() = this->yPid.Update(yError, dt);
        worldForce.Z() = this->zPid.Update(zError, dt); // Continue maintaining Z position

        worldTorque.X() = this->rollPid.Update(rollError, dt);
        worldTorque.Y() = this->pitchPid.Update(pitchError, dt);
        worldTorque.Z() = this->yawPid.Update(yawError, dt);

        // Apply forces and torques
        this->link_->AddForce(worldForce);
        this->link_->AddTorque(worldTorque);

        std::cout << "[set_velocity_plugin] Attitude and X/Y correction applied\n";

        std::cout << "[set_velocity_plugin] ATorque: " << worldTorque <<"\n";

        std::cout << "[set_velocity_plugin] Err Euler X: " << rollError <<"\n";
        std::cout << "[set_velocity_plugin] Err Euler Y: " << pitchError <<"\n";
        std::cout << "[set_velocity_plugin] Err Euler Z: " << yawError <<"\n";
        std::cout << "[set_velocity_plugin] AFroce: " << worldForce <<"\n";

        std::cout << "[set_velocity_plugin] Err X: " << xError <<"\n";
        std::cout << "[set_velocity_plugin] Err Y: " << yError <<"\n";
        std::cout << "[set_velocity_plugin] Err Z: " << zError <<"\n";

      }

      // Update launch status after duration
      if ((time - trigger_time_) > launch_duration_) {
        launch_status_ = VEHICLE_LAUNCHED;
        std::cout << "[set_velocity_plugin] Done\n";
      }

    }

    this->lastUpdateTime = time;
  }
}




void SetVelPlugin::VelocityCallback(CommandMotorSpeedPtr &rot_velocities) {
  if(rot_velocities->motor_speed_size() < motor_number_) {
    std::cout  << "You tried to access index " << motor_number_
      << " of the MotorSpeed message array which is of size " << rot_velocities->motor_speed_size() << "." << std::endl;
  } else ref_motor_rot_vel_ = std::min(static_cast<double>(rot_velocities->motor_speed(motor_number_)), static_cast<double>(max_rot_velocity_));
}
}
