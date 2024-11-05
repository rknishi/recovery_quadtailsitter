#
# Thrust and Torque Setpoint provided by controllers

uint64 timestamp 

#Thrust (not normalized)
float32 thrust


#torques not normalized and in fw frame
float32 tx
float32 ty
float32 tz

#position locked
bool positionlock

# Controllers parameters
float32[9] qquat
float32[2] qangle
float32[4] r
float32[4] rvar