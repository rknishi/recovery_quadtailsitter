#
# Fixed wing odometry

uint64 timestamp 

uint64 timestamp_sample

#position in meters in NED fixed frame
float32 x
float32 y
float32 z

#orientation quaternion
float32 q0 #quaternion rotation from Fw body frame to reference frame
float32 q1 #quaternion rotation from Fw body frame to reference frame
float32 q2 #quaternion rotation from Fw body frame to reference frame
float32 q3 #quaternion rotation from Fw body frame to reference frame


#velocity in m/s in NED fixed frame
float32 vx
float32 vy
float32 vz 

#angular rate in FW frame
float32 rollspeed
float32 pitchspeed
float32 yawspeed