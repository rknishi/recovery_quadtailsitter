import numpy as np
import casadi as ca

def quat_mult(a, b):
    w1, x1, y1, z1 = a[0], a[1], a[2], a[3]
    w2, x2, y2, z2 = b[0], b[1], b[2], b[3]
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return np.array([w, x, y, z])
    else:
        return ca.vertcat(w, x, y, z)

def rotate_quat(v, q):
    V = np.array([0, v[0], v[1], v[2]])
    conj_q = np.array([q[0], -q[1], -q[2], -q[3]])
    V_rotated = quat_mult(quat_mult(q, V), conj_q)
    v_rotated = V_rotated[1:]
    return v_rotated

def quat_error(qd, q):
        q0e = qd[0]*q[0] + qd[1]*q[1] + qd[2]*q[2] + qd[3]*q[3]
        q1e = -qd[1]*q[0] + qd[0]*q[1] + qd[3]*q[2] - qd[2]*q[3]
        q2e = -qd[2]*q[0] - qd[3]*q[1] + qd[0]*q[2]+qd[1]*q[3]
        q3e = -qd[3]*q[0] + qd[2]*q[1] - qd[1]*q[2] + qd[0]*q[3]
        return np.array([q0e, q1e, q2e, q3e])
    

def quat_to_rot(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = q0**2 + q1**2 - q2**2 - q3**2
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q0 * q2 + q1 * q3)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = q0**2-q1**2+q2**2-q3**2
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = q0**2-q1**2-q2**2+q3**2
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix

def rot_to_quat(rot):
    # From https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
    # Assuming rot is a 3x3 rotation matrix
    trace = np.trace(rot)
    

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        S = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S

    return np.array([qw, qx, qy, qz])

def quat_rotate(q, v):
    qv = ca.vertcat(0, v)
    q_conj = quat_conj(q)
    q_rotated = quat_mult(quat_mult(q, qv), q_conj)
    if isinstance(q, np.ndarray): 
        return np.array([q_rotated[1], q_rotated[2], q_rotated[3]])
    else:
        return ca.vertcat(q_rotated[1], q_rotated[2], q_rotated[3])


def quat_conj(q):
    if isinstance(q, np.ndarray):
        return np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        return ca.vertcat(q[0], -q[1], -q[2], -q[3])
    
def quat_conj_vectorized(Q):
    Q_conj = ca.horzcat(Q[:, 0], -Q[:, 1], -Q[:, 2], -Q[:, 3])
    return Q_conj

def decompose_quaternion_vectorized(Q_error):
	# Extract yaw (qz)
	qz_w = ca.sqrt(Q_error[:, 0]**2 + Q_error[:, 3]**2) + 1e-8
	qz = ca.horzcat(qz_w, ca.DM.zeros((Q_error.shape[0], 1)), ca.DM.zeros((Q_error.shape[0], 1)), Q_error[:, 3])
	qz = qz / ca.sqrt(qz[:, 0]**2 + qz[:, 3]**2 + 1e-8)  # Normalize qz
    
	# Compute qxy (pitch and roll)
	qxy = quat_mult_vectorized(Q_error, quat_conj_vectorized(qz))  # Vectorized multiplication
	qxy = ca.horzcat(qxy[:, 0], qxy[:, 1], qxy[:, 2], ca.DM.zeros((Q_error.shape[0], 1)))  # Remove z-component
	qxy = qxy / ca.sqrt(qxy[:, 0]**2 + qxy[:, 1]**2 + qxy[:, 2]**2 + 1e-8)  # Normalize qxy


	return ca.horzcat(qz[:, 3], qxy[:, 1]**2 + qxy[:, 2]**2)  # Return yaw and combined pitch/roll errors

# Vectorized version of decompose_quaternion_error2
def decompose_quaternion_error_vectorized(Q, q_ref):
    # Assume Q is of shape (N, 4), and q_ref is a single quaternion (4,)
    
    e_cmd_x = ca.DM([0, 0, -1]) + 1e-8  # Commanded direction (3D vector)
    
    # Compute ex_cur (current x-direction in the body frame) for all quaternions
    q0, q1, q2, q3 = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    ex_cur = ca.horzcat(
        q0**2 + q1**2 - q2**2 - q3**2,
        2 * (q1 * q2 + q0 * q3),
        2 * (q1 * q3 - q0 * q2)
    )
    
    # Calculate dot product and alpha for all quaternions
    dot_product = ca.mtimes(ex_cur, e_cmd_x) + 1e-8
    alpha = ca.acos(ca.fmax(ca.fmin(dot_product, 1 - 1e-8), -1 + 1e-8)) + 1e-8
    
    # Calculate rotation axis (n) and normalize it
    n = ca.cross(ex_cur, ca.repmat(e_cmd_x.T, Q.shape[0], 1)) + 1e-8
    n_norm = ca.sqrt(n[:, 0]**2 + n[:, 1]**2 + n[:, 2]**2) + 1e-8
    n = ca.if_else(n_norm > 1e-8, n / ca.repmat(n_norm, 1, 3), ca.DM.zeros(Q.shape[0], 3)) + 1e-8
    
    # Rotate n using quaternion conjugation
    n_b = rotate_quat_vectorized(quat_conj_vectorized(Q), n) + 1e-8
    
    # Compute q_erp for all quaternions
    # Precompute sin and cos of half the angle
    half_alpha = alpha / 2
    sin_half_alpha = ca.sin(half_alpha)
    cos_half_alpha = ca.cos(half_alpha)

    # Compute q_erp for all quaternions
    q_erp = ca.if_else(
        alpha < 1e-6,
        ca.horzcat(ca.DM.ones(Q.shape[0], 1), ca.DM.zeros(Q.shape[0], 3)),
        ca.horzcat(cos_half_alpha, n_b[:, 0] * sin_half_alpha, n_b[:, 1] * sin_half_alpha, n_b[:, 2] * sin_half_alpha)
    )
    # Compute quaternion error between current quaternion and constant reference quaternion
    qex = quat_mult_vectorized2(quat_conj_vectorized(quat_mult_vectorized2(Q, q_erp)), q_ref) + 1e-8
    
    # Return yaw (qz) and combined pitch/roll errors
    return ca.horzcat(qex[:, 1], q_erp[:, 2]**2 + q_erp[:, 3]**2)

def decompose_quaternion_error2(q, q_ref):
    # Extract yaw (qz)
    e_cmd_x = ca.vertcat(0,0,-1) + 1e-8
    ex_cur =  ca.vertcat(
                q[0]**2+q[1]**2-q[2]**2-q[3]**2 + 1e-8,
                2*(q[1]*q[2]+q[0]*q[3] + 1e-8),
                2*(q[1]*q[3]-q[0]*q[2] + 1e-8))
    dot_product = ca.mtimes(ex_cur.T, e_cmd_x) + 1e-8
    alpha = ca.acos(ca.fmax(ca.fmin(dot_product, 1), -1)) + 1e-8 
    # Calculate q_erp
    n = ca.cross(ex_cur, e_cmd_x) 
    n_norm = ca.norm_2(n)
    n = ca.if_else(n_norm > 1e-8, n / n_norm, ca.MX([0, 0, 0]))
    n_b = rotate_quat(q = quat_conj(q), v = n)

    q_erp = ca.if_else(
        ca.fabs(alpha) < 1e-8,
        ca.MX([1, 0, 0, 0]),
        ca.vertcat(ca.cos(alpha / 2), n_b[0] * ca.sin(alpha / 2), n_b[1] * ca.sin(alpha / 2), n_b[2] * ca.sin(alpha / 2)))
    qex = quat_mult(quat_conj(quat_mult(q, q_erp)), q_ref)
    return ca.horzcat(qex[1], q_erp[2]**2 + q_erp[3]**2)  # Return yaw and combined pitch/roll errors

def quat_mult_vectorized(Q1, Q2):
    # Q1 and Q2 should have dimensions (N, 4) where N is the number of quaternions
    w1, x1, y1, z1 = Q1[:, 0], Q1[:, 1], Q1[:, 2], Q1[:, 3]
    w2, x2, y2, z2 = Q2[:, 0], Q2[:, 1], Q2[:, 2], Q2[:, 3]

    # Quaternion product
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Concatenate to form the resulting quaternion matrix (N, 4)
    return ca.horzcat(w, x, y, z)

def quat_mult_vectorized2(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]  # q2 is constant (1 quaternion)
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return ca.horzcat(w, x, y, z)

# Rotate vector by quaternion for vectorized version
def rotate_quat_vectorized(q, v):
    q_v = ca.horzcat(ca.DM.zeros((q.shape[0], 1)), v)
    rotated_v = quat_mult_vectorized2(q, quat_mult_vectorized2(quat_conj_vectorized(q), q_v))
    return rotated_v[:, 1:]
