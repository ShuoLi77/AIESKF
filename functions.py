import math
import time
import matplotlib.pyplot as plt
import numpy as np
import torch

# define some constrans parameters
constant_omega_ie = 7.2921151467e-05  # Earth rotation rate in rad/s
constant_e = 0.0818191908425  # WGS84 eccentricity
constant_a = 6378137.0  # WGS84 Semimajor axis (equatorial radius) in meter
constant_c = 299792458.0  # light speed in m/s
constant_L1 = 1575.42 * 1e6  # GPS L1 Frequency in HZ
constant_pi = math.pi  # pi
constant_rad2deg = 180.0 / constant_pi
contant_meu = 398600500000000.0  # ....... earth's universal gravitational [m^3/s^2]
constant_F = -4.442807633e-10  # ....... Constant, [sec/(meter)^(1/2)]
micro_g_to_meters_per_second_squared = 9.80665e-6
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
# ones = np.ones
# zeros = np.zeros
# arange = np.arange
# meshgrid = np.meshgrid
# sin = np.sin
# sinh = np.sinh
# cos = np.cos
# cosh = np.cosh
# tanh = np.tanh
sin = torch.sin
# sinh = np.sinh
cos = torch.cos
asin = math.asin
acos = math.acos
atan = math.atan
atan2 = math.atan2
# linspace = np.linspace
# exp = np.exp
# log = np.log
# tensor = torch.tensor
# normal = np.random.normal
# randn = np.random.randn
# rand = np.random.rand
# matmul = np.dot
# int32 = np.int32
# float32 = np.float32
# concat = np.concatenate
# stack = np.stack
# abs = np.abs
# eye = np.eye
sqrt = math.sqrt
if torch.cuda.is_available():
    dev = torch.device("cuda")  # cuda:1 cuda:2....etc.
else:
    dev = torch.device("cpu")

def euler_to_CTM(eul):
    # Euler_to_CTM - Converts a set of Euler angles to the corresponding
    # coordinate transformation matrix R_b_n
    # b-frame: b_x, b_y, b_z
    # n-frame: NED
    # Inputs:
    #   eul     Euler angles describing rotation from beta to alpha in the
    #           order [roll, pitch, yaw] (rad)
    # Outputs:
    #   R_b_n       coordinate transformation matrix from b-frame to n-frame
    # Precalculate sines and cosines of the Euler angles
    sin_phi = torch.sin(eul[0])
    cos_phi = torch.cos(eul[0])
    sin_theta = torch.sin(eul[1])
    cos_theta = torch.cos(eul[1])
    sin_psi = torch.sin(eul[2])
    cos_psi = torch.cos(eul[2])
    # Calculate coordinate transformation matrix
    C = torch.zeros(9).reshape(3, 3)
    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta
    # C = [cos(y)*cos(p),  -sin(y)*cos(r)+cos(y)*sin(p)*sin(r), sin(y)*sin(r)+cos(y)*sin(p)*cos(r);
    #      sin(y)*cos(p),   cos(y)*cos(r)+sin(y)*sin(p)*sin(r),-sin(r)*cos(y)+sin(y)*sin(p)*cos(r);
    #            -sin(p),                        sin(r)*cos(p),                      cos(p)*cos(r)];
    # C : n to b
    # C.T b to n
    return C


def CTM_to_euler(C):
    # CTM_to_Euler - Converts a coordinate transformation matrix to the
    # corresponding set of Euler angles#
    #
    # Software for use with "Principles of GNSS, Inertial, and Multisensor
    # Integrated Navigation Systems," Second Edition.
    #
    # This function created 1/4/2012 by Paul Groves
    #
    # Inputs:
    #   C       coordinate transformation matrix describing transformation from
    #           beta to alpha
    #
    # Outputs:
    #   eul     Euler angles describing rotation from beta to alpha in the
    #           order roll, pitch, yaw(rad)
    '''in place'''
    eul = torch.zeros(3)
    eul[0] = torch.atan2(C[1, 2], C[2, 2])
    # roll
    eul[1] = -torch.asin(C[0, 2])
    # pitch
    eul[2] = torch.atan2(C[0, 1], C[0, 0])
    # yaw
    # eul[0] = - math.atan2(C[2,0],C[2,2]);  # roll
    # eul[1] = math.asin(C[2,1]);        # pitch
    # eul[2] = - math.atan2(C[0,1],C[1,1]);  # yaw
    
    # eul = torch.where(torch.isnan(eul), torch.full_like(eul, 0.), eul)
    
    return eul


def skew_symmetric(vec):
    # Skew_symmetric - Calculates skew-symmetric matrix
    #
    # Software for use with "Principles of GNSS, Inertial, and Multisensor
    # Integrated Navigation Systems," Second Edition.
    #
    # This function created 1/4/2012 by Paul Groves
    #
    # Inputs:
    #   a       3-element vector
    # Outputs:
    #   A       3x3matrix
    Matr = torch.zeros(9).reshape(3, 3)
    Matr[0, 0] = 0
    Matr[0, 1] = -vec[2]
    Matr[0, 2] = vec[1]
    Matr[1, 0] = vec[2]
    Matr[1, 1] = 0
    Matr[1, 2] = -vec[0]
    Matr[2, 0] = -vec[1]
    Matr[2, 1] = vec[0]
    Matr[2, 2] = 0
    return Matr


omega_ie_e = torch.tensor([[0], [0], [constant_omega_ie]])
OMEGA_ie_e = skew_symmetric(omega_ie_e).to(dev)


def geo_ned2ecef(*arg):
    # (p_b_llh, v_eb_n, R_b_n):
    # geo_ned2ecef - Converts curvilinear to Cartesian position, velocity
    # resolving axes from NED to ECEF and attitude from NED- to ECEF-referenced
    #
    # Inputs:
    #   Lat           latitude (rad)
    #   Long          longitude (rad)
    #   h             height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   R_b_n         body-to-NED coordinate transformation matrix
    #
    # Outputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   R_b_e         body-to-ECEF-frame coordinate transformation matrix
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity
    # Calculate transverse radius of curvature using (2.105)
    Lat = arg[0][0]
    Long = arg[0][1]
    h = float(arg[0][2])
    RN = R_0 / math.sqrt(1 - (e * math.sin(Lat)) ** 2)
    if len(arg) == 1:
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        r_eb_e = torch.tensor(
            [
                [(RN + h) * cos_lat * cos_lon],
                [(RN + h) * cos_lat * sin_lon],
                [((1 - e**2) * RN + h) * sin_lat],
            ]
        )
        return r_eb_e
    if len(arg) == 2:
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        r_eb_e = torch.tensor(
            [
                [(RN + h) * cos_lat * cos_lon],
                [(RN + h) * cos_lat * sin_lon],
                [((1 - e**2) * RN + h) * sin_lat],
            ]
        )
        # Calculate ECEF to NED coordinate transformation matrix
        R_e_n = torch.tensor(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        R_n_e = R_e_n.T
        # Transform velocity using (2.73)
        v_eb_e = R_n_e @ arg[1]
        return r_eb_e, v_eb_e
    if len(arg) == 3:
        # Convert position using (2.112)
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        r_eb_e = torch.tensor(
            [
                [(RN + h) * cos_lat * cos_lon],
                [(RN + h) * cos_lat * sin_lon],
                [((1 - e**2) * RN + h) * sin_lat],
            ]
        )
        # Calculate ECEF to NED coordinate transformation matrix
        R_e_n = torch.tensor(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        R_n_e = R_e_n.T
        # Transform velocity using (2.73)
        v_eb_e = R_n_e @ arg[1]
        # Transform attitude using (2.15)
        R_b_e = R_n_e @ arg[2]
        return r_eb_e.reshape(3), v_eb_e, R_b_e



def dead_reckoning_ecef(
    Fs, dr_temp, est_C_b_e_new, imu_meas, imu_error, on_cpu: bool = False
):
    # if on_cpu:
    #     result_dev = est_pos_new.device
    #     dev = torch.device("cpu")
    # else:
    #     result_dev = est_pos_new.device
    #     dev = result_dev
    dt = 1/Fs
    dr_result = [dr_temp]
    
    # move initial data to dev
    # est_pos_new = est_pos_new.to(dev)
    # est_vel_new = est_vel_new.to(dev)
    # est_C_b_e_new = est_C_b_e_new.to(dev)
    # imu = imu[:: int(dt * 100), :].to(dev)

    # old_r_eb_e_prev = est_pos_new
    # old_v_eb_e_prev = est_vel_new
    old_C_b_e_prev = est_C_b_e_new
    # old_v_eb_e_new = None
    # old_v_eb_e_new = None
    # old_C_b_e_new = None
    for i in range(imu_meas.shape[0]):
        

        f_ib_b = imu_meas[i, 0:3] - imu_error[0:3]
        omega_ib_b = imu_meas[i, 3:6] - imu_error[3:6] 
        # ATTITUDE UPDATE
        # From (2.145) determine the Earth rotation over the update interval
        # C_Earth = C_e_i' * old_C_e_i
        # with torch.no_grad():

        alpha_ie = torch.tensor(constant_omega_ie * dt, device=dev)
        C_Earth = torch.tensor(
            [
                [cos(alpha_ie), sin(alpha_ie), 0.0],
                [-sin(alpha_ie), cos(alpha_ie), 0.0],
                [0.0, 0.0, 1.0],
            ],
            device=dev,
        )
        # Calculate attitude increment, magnitude, and skew-symmetric matrix
        alpha_ib_b = omega_ib_b * dt
        # this produces a warning, we should fix it
        # mag_alpha = torch.sqrt(alpha_ib_b.detach().T @ alpha_ib_b.detach())
        # print(alpha_ib_b)
        # print(alpha_ib_b.T)
        mag_alpha = torch.norm(alpha_ib_b)
        # print(mag_alpha)

        # quit()
        ''' in-place opreation'''
        Alpha_ib_b = torch.tensor(
            [
                [0, -alpha_ib_b[2], alpha_ib_b[1]],
                [alpha_ib_b[2], 0, -alpha_ib_b[0]],
                [-alpha_ib_b[1], alpha_ib_b[0], 0],
            ],
            device=dev,
        )
        # Obtain coordinate transformation matrix from the new attitude w.r.t. an
        # inertial frame to the old using Rodrigues' formula, (5.73)
        if mag_alpha > 1.0e-8:
            C_new_old = (
                torch.eye(3, device=dev)
                + torch.sin(mag_alpha) / mag_alpha * Alpha_ib_b
                + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b @ Alpha_ib_b
            )
        else:
            C_new_old = torch.eye(3, device=dev) + Alpha_ib_b
        # Update attitude using (5.75)
        old_C_b_e_new = C_Earth @ old_C_b_e_prev @ C_new_old
        # SPECIFIC FORCE FRAME TRANSFORMATION
        # Calculate the average body-to-ECEF-frame coordinate transformation
        # matrix over the update interval using (5.84) and (5.85)
        if mag_alpha > 1.0e-8:
            ave_C_b_e = (
                old_C_b_e_prev
                @ (
                    torch.eye(3, device=dev)
                    + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b
                    + (1 - torch.sin(mag_alpha) / mag_alpha)
                    / mag_alpha ** 2
                    * Alpha_ib_b
                    @ Alpha_ib_b
                )
                - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ old_C_b_e_prev
            )
        else:
            ave_C_b_e = (
                old_C_b_e_prev
                - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ old_C_b_e_prev
            )
                # Transform specific force to ECEF-frame resolving axes using (5.85)
        f_ib_e = ave_C_b_e @ f_ib_b
    
    
    
        # CALCULATE ATTITUDE
        # _,_,est_att_new = ecef2geo_ned(est_pos_new, est_vel_new, est_C_b_e_new)
        est_att_new = torch.squeeze(CTM_to_euler(old_C_b_e_new.T))
        # print(est_att_new)
        old_C_b_e_prev = old_C_b_e_new

        # UPDATE VELOCITY
        est_vel_new = dr_result[i][3:6] + dt * ( f_ib_e + gravity_ECEF(dr_result[i][0:3]) - 2 * skew_symmetric([0, 0, constant_omega_ie]).to(dev) @ dr_result[i][3:6])

        '''should use another way to calculate position? if so there is a gradient question'''
        # UPDATE CARTESIAN POSITION
        est_pos_new = dr_result[i][0:3] + (dr_result[i][3:6] + est_vel_new) / 2 * dt


        ''''Should use detach to gravity?'''
        dr_result.append(torch.cat([est_pos_new,est_vel_new,est_att_new,old_C_b_e_new.reshape(9)]))


        
        
    return  torch.stack(dr_result), old_C_b_e_new


def dead_reckoning_ecef_test(Fs, dr_temp, est_C_b_e, imu_meas, imu_error):
        
    dt = 1/Fs


    f_ib_b = imu_meas[0:3] - imu_error[0:3]
    omega_ib_b = imu_meas[3:6] - imu_error[3:6] 
    # ATTITUDE UPDATE
    # From (2.145) determine the Earth rotation over the update interval
    # C_Earth = C_e_i' * old_C_e_i
    alpha_ie = torch.tensor(constant_omega_ie * dt, device=dev)
    C_Earth = torch.tensor(
        [
            [cos(alpha_ie), sin(alpha_ie), 0.0],
            [-sin(alpha_ie), cos(alpha_ie), 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=dev,
    )
    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * dt
    # this produces a warning, we should fix it
    # mag_alpha = torch.sqrt(alpha_ib_b.detach().T @ alpha_ib_b.detach())
    mag_alpha = torch.norm(alpha_ib_b)
    '''gradient'''
    Alpha_ib_b = torch.tensor(
        [
            [0, -alpha_ib_b[2], alpha_ib_b[1]],
            [alpha_ib_b[2], 0, -alpha_ib_b[0]],
            [-alpha_ib_b[1], alpha_ib_b[0], 0],
        ],
        device=dev,
    )
    # Obtain coordinate transformation matrix from the new attitude w.r.t. an
    # inertial frame to the old using Rodrigues' formula, (5.73)
    if mag_alpha > 1.0e-8:
        C_new_old = (
            torch.eye(3, device=dev)
            + torch.sin(mag_alpha) / mag_alpha * Alpha_ib_b
            + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b @ Alpha_ib_b
        )
    else:
        C_new_old = torch.eye(3, device=dev) + Alpha_ib_b
    # Update attitude using (5.75)
    C_b_e_new = C_Earth @ est_C_b_e @ C_new_old
    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate the average body-to-ECEF-frame coordinate transformation
    # matrix over the update interval using (5.84) and (5.85)
    if mag_alpha > 1.0e-8:
        ave_C_b_e = (
            est_C_b_e
            @ (
                torch.eye(3, device=dev)
                + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b
                + (1 - torch.sin(mag_alpha) / mag_alpha)
                / mag_alpha ** 2
                * Alpha_ib_b
                @ Alpha_ib_b
            )
            - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ est_C_b_e
        )
    else:
        ave_C_b_e = (
            est_C_b_e
            - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ est_C_b_e
        )
    # Transform specific force to ECEF-frame resolving axes using (5.85)
    f_ib_e = ave_C_b_e @ f_ib_b
    
    
    

    # CALCULATE ATTITUDE
    # _,_,est_att_new = ecef2geo_ned(est_pos_new, est_vel_new, est_C_b_e_new)
    est_att_new = torch.squeeze(CTM_to_euler(C_b_e_new.T))
    # print(est_att_new)
    

    # UPDATE VELOCITY
    est_vel_new = dr_temp[3:6] + dt * ( f_ib_e + gravity_ECEF(dr_temp[0:3]) - 2 * skew_symmetric([0, 0, constant_omega_ie]).to(dev) @ dr_temp[3:6])
    '''gradient question'''
    # UPDATE CARTESIAN POSITION
    est_pos_new = dr_temp[0:3] + (dr_temp[3:6] + est_vel_new) / 2 * dt

    ''''Should use detach to gravity?'''
    dr_result = torch.cat([est_pos_new,est_vel_new,est_att_new,C_b_e_new.reshape(9)])
        
    return dr_result, C_b_e_new
        
def Model_LC(meas, dr_temp, est_C_b_e_old, imu, imu_error, P_old, Q, R, Fs_meas):
    
    meas_f_ib_b = imu[:3].reshape(3, 1)
    est_pos_old = dr_temp[0:3]
    
    
    est_L_b_old = ecef2geo_ned(est_pos_old)
    dt = 1/Fs_meas
    """ Set State transition matrix F """
    geocentric_radius = (
        constant_a
        / torch.sqrt(1 - (constant_e * torch.sin(est_L_b_old[0])) ** 2)
        * torch.sqrt(
            torch.cos(est_L_b_old[0]) ** 2
            + (1 - constant_e**2) ** 2 * torch.sin(est_L_b_old[0]) ** 2
        )
    )
    F = torch.eye(15)
    F12 = torch.eye(3) * dt
    F21 = (
        -dt
        * 2
        * gravity_ECEF(est_pos_old).reshape(3,1)
        / geocentric_radius
        @ est_pos_old.reshape(1,3)
        / torch.norm(est_pos_old)
    )
    F22 = torch.eye(3) - 2 * OMEGA_ie_e * dt
    F23 = -dt * skew_symmetric(est_C_b_e_old @ meas_f_ib_b)
    F24 = est_C_b_e_old * dt
    F33 = torch.eye(3) - OMEGA_ie_e * dt
    F35 = est_C_b_e_old * dt

    F[0:3, 3:6] = F12
    F[3:6, 0:3] = F21
    F[3:6, 3:6] = F22
    F[3:6, 6:9] = F23
    F[3:6, 9:12] = F24
    F[6:9, 6:9] = F33
    F[6:9, 12:15] = F35
    
    """ Prediction state vector and state covariance matrix"""
    x_predict = torch.zeros(15)  # State Vector is zero
    P_predict = F @ P_old @ F.T + Q


    """ Set measurement matrix H """
    H = torch.zeros((6, 15))
    H[0:3, 0:3] = -torch.eye(3)
    H[3:6, 3:6] = -torch.eye(3)


    """ Formulate measurement innovations """
    delta_z = meas - dr_temp[0:6]
    # print(delta_z)

    """ Calculate Kalman gain """
    K = P_predict @ H.T @ torch.linalg.inv(H @ P_predict @ H.T + R)
    # print(K)
    """ Update state estimates and state covariance matrix P """
    Inno = x_predict + K @ delta_z
    P_new = (torch.eye(15) - K @ H) @ P_predict

    """ CLOSED-LOOP correct attitude, velocity, and positionm, update IMU bias estimates """

    dr_new = torch.zeros(18)
    dr_new[:6] = dr_temp[:6] - Inno[:6]

    # est_C_b_e_new = functions.euler_to_CTM(self.dr_new[6:]).T
    # imu_error = Inno[9:]

    skew_att = torch.tensor(
        [
            [0, -Inno[8], Inno[7]],
            [Inno[8], 0, -Inno[6]],
            [-Inno[7], Inno[6], 0],
        ]
    )
    # print( K)
    est_C_b_e_est = (torch.eye(3) - skew_att) @ est_C_b_e_old
    # est_IMU_bias_new = est_IMU_bias_old + x_error[9:15]
    '''gradient? There is always a nan returned when the net is not converged '''
    # _,_,dr_new_att= functions.ecef2geo_ned(dr_new_pv[0:3], dr_new_pv[3:6], est_C_b_e_new)
    dr_new_att_est = torch.squeeze(CTM_to_euler(est_C_b_e_est.T))

    imu_error_est = Inno[9:]
    # self.dr_new = dr_temp
    # est_C_b_e_new = est_C_b_e_old
    est_C_b_e_new = est_C_b_e_est
    imu_error_new = imu_error + imu_error_est
    
    dr_new[6:9] = dr_new_att_est
    dr_new[9:18] = est_C_b_e_new.reshape(9)

    return dr_new, est_C_b_e_new, imu_error_new, P_new


def gravity_ECEF(p_eb_e):
    # Gravitation_ECI - Calculates  acceleration due to gravity resolved about
    # ECEF-frame
    #
    # Software for use with "Principles of GNSS, Inertial, and Multisensor
    # Integrated Navigation Systems," Second Edition.
    #
    # This function created 1/4/2012 by Paul Groves
    #
    # Inputs:
    #   r_eb_e  Cartesian position of body frame w.r.t. ECEF frame, resolved
    #           about ECEF-frame axes (m)
    # Outputs:
    #   g       Acceleration due to gravity (m/s^2)
    # Copyright 2012, Paul Groves
    # License: BSD; see license.txt for details
    # Parameters
    dev = p_eb_e.device
    R_0 = 6378137
    # WGS84 Equatorial radius in meters
    mu = 3.986004418e14
    # WGS84 Earth gravitational constant (m^3 s^-2)
    J_2 = 1.082627e-3
    # WGS84 Earth's second gravitational constant
    omega_ie = 7.292115e-5
    # Earth rotation rate (rad/s)
    # Begins
    # Calculate distance from center of the Earth
    mag_r = torch.norm(p_eb_e)
    # If the input position is 0,0,0, produce a dummy output
    if mag_r == 0:
        g = torch.tensor([[0], [0], [0]], device=dev)
    # Calculate gravitational acceleration using (2.142)
    else:
        z_scale = float(5 * (p_eb_e[2] / mag_r) ** 2)
        gamma = (
            -mu
            / mag_r ** 3
            * (
                p_eb_e
                + (
                    1.5
                    * J_2
                    * (R_0 / mag_r) ** 2
                    * torch.tensor(
                        [
                            [(1 - z_scale) * p_eb_e[0]],
                            [(1 - z_scale) * p_eb_e[1]],
                            [(3 - z_scale) * p_eb_e[2]],
                        ],
                        device=dev,
                    )
                ).reshape(3)
            )
        )
        # Add centripetal acceleration using (2.133)
        g = torch.zeros(3, device=dev).reshape(3)
        g[0:2] = gamma[0:2] + omega_ie ** 2 * p_eb_e[0:2]
        g[2] = gamma[2]
    return g


def initialize_NED_attitude(att_euler):
    C_b_n = euler_to_CTM(att_euler).T
    init_error_delta_eul_nb_n = (
        torch.tensor([-0.05, 0.04, 1]).reshape(3, 1) * deg_to_rad
    )
    delta_C_b_n = euler_to_CTM(-init_error_delta_eul_nb_n)
    C_b_n = delta_C_b_n @ C_b_n
    init_att_euler = CTM_to_euler(C_b_n.T)
    return init_att_euler, C_b_n


def lc_propagate_f(est_pos_old, est_C_b_e_old, imu, Fs_meas):
    meas_f_ib_b = imu[:3].reshape(3, 1)
    est_L_b_old = ecef2geo_ned(est_pos_old)
    dt = 1/Fs_meas
    """ Set State transition matrix F """
    geocentric_radius = (
        constant_a
        / torch.sqrt(1 - (constant_e * torch.sin(est_L_b_old[0])) ** 2)
        * torch.sqrt(
            torch.cos(est_L_b_old[0]) ** 2
            + (1 - constant_e**2) ** 2 * torch.sin(est_L_b_old[0]) ** 2
        )
    )
    F = torch.eye(9)
    F12 = torch.eye(3) * dt
    F21 = (
        -dt
        * 2
        * gravity_ECEF(est_pos_old).reshape(3,1)
        / geocentric_radius
        @ est_pos_old.reshape(1,3)
        / torch.norm(est_pos_old)
    )
    F22 = torch.eye(3) - 2 * OMEGA_ie_e * dt
    F23 = -dt * skew_symmetric(est_C_b_e_old @ meas_f_ib_b)
    F[0:3, 3:6] = F12
    F[3:6, 0:3] = F21
    F[3:6, 3:6] = F22
    F[3:6, 6:9] = F23
    return F


def ecef2geo_ned(*arg):
    # (r_eb_e,v_eb_e)
    # ecef2geo_ned - Converts Cartesian to geodetic position and ned velocity, acc
    # resolving axes from ECEF to NED
    #
    # This function created 11/4/2012 by Paul Groves
    #
    # Inputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   a_eb_e        acc of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    # Outputs:
    #   p_b_llh       latitude (rad) longitude (rad) height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   a_eb_n        acc of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    # Parameters
    # Convert position using Borkowski closed-form exact solution
    r_eb_e = arg[0]
    # From (2.113)
    Long = math.atan2(r_eb_e[1], r_eb_e[0])
    # From (C.29) and (C.30)
    k1 = math.sqrt(1 - constant_e**2) * abs(r_eb_e[2])
    k2 = constant_e**2 * constant_a
    beta = math.sqrt(r_eb_e[0] ** 2 + r_eb_e[1] ** 2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta
    # From (C.31)
    P = 4 / 3 * (E * F + 1)
    # From (C.32)
    Q = 2 * (E**2 - F**2)
    # From (C.33)
    D = P**3 + Q**2
    # From (C.34)
    V = (math.sqrt(D) - Q) ** (1 / 3) - (math.sqrt(D) + Q) ** (1 / 3)
    # From (C.35)
    G = 0.5 * (math.sqrt(E**2 + V) + E)
    # From (C.36)
    T = math.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G
    # From (C.37)
    Lat = torch.sign(r_eb_e[2]) * math.atan2(
        (1 - T**2), (2 * T * math.sqrt(1 - constant_e**2))
    )
    # From (C.38)
    h = (beta - constant_a * T) * cos(Lat) + (
        r_eb_e[2] - torch.sign(r_eb_e[2]) * constant_a * math.sqrt(1 - constant_e**2)
    ) * sin(Lat)
    p_b_llh = torch.tensor([[float(Lat)], [float(Long)], [float(h)]])
    if len(arg) == 1:
        return p_b_llh
    if len(arg) == 2:
        v_eb_e = arg[1]
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        R_e_n = torch.tensor(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        # Transform velocity using (2.73)
        v_eb_n = R_e_n @ v_eb_e
        return p_b_llh, v_eb_n
    if len(arg) == 3:
        v_eb_e = arg[1]
        R_b_e = arg[2]
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        R_e_n = torch.tensor(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        # Transform velocity using (2.73)
        v_eb_n = R_e_n @ v_eb_e
        R_b_n = R_e_n @ R_b_e
        euler_b_n = CTM_to_euler(R_b_n.T).reshape(3)
        return p_b_llh, v_eb_n, euler_b_n


def ecef2geo_ned_array(*arg):
    # (r_eb_e,v_eb_e)
    # ecef2geo_ned - Converts Cartesian to geodetic position and ned velocity, acc
    # resolving axes from ECEF to NED
    #
    # This function created 11/4/2012 by Paul Groves
    #
    # Inputs:
    #   r_eb_e        Cartesian position of body frame w.r.t. ECEF frame, resolved
    #                 along ECEF-frame axes (m)
    #   v_eb_e        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    #   a_eb_e        acc of body frame w.r.t. ECEF frame, resolved along
    #                 ECEF-frame axes (m/s)
    # Outputs:
    #   p_b_llh       latitude (rad) longitude (rad) height (m)
    #   v_eb_n        velocity of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    #   a_eb_n        acc of body frame w.r.t. ECEF frame, resolved along
    #                 north, east, and down (m/s)
    # Parameters
    # Convert position using Borkowski closed-form exact solution
    r_eb_e = arg[0]
    # From (2.113)
    Long = math.atan2(r_eb_e[1], r_eb_e[0])
    # From (C.29) and (C.30)
    k1 = math.sqrt(1 - constant_e**2) * abs(r_eb_e[2])
    k2 = constant_e**2 * constant_a
    beta = math.sqrt(r_eb_e[0] ** 2 + r_eb_e[1] ** 2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta
    # From (C.31)
    P = 4 / 3 * (E * F + 1)
    # From (C.32)
    Q = 2 * (E**2 - F**2)
    # From (C.33)
    D = P**3 + Q**2
    # From (C.34)
    V = (math.sqrt(D) - Q) ** (1 / 3) - (math.sqrt(D) + Q) ** (1 / 3)
    # From (C.35)
    G = 0.5 * (math.sqrt(E**2 + V) + E)
    # From (C.36)
    T = math.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G
    # From (C.37)
    Lat = np.sign(r_eb_e[2]) * math.atan2(
        (1 - T**2), (2 * T * math.sqrt(1 - constant_e**2))
    )
    # From (C.38)
    h = (beta - constant_a * T) * math.cos(Lat) + (
        r_eb_e[2] - np.sign(r_eb_e[2]) * constant_a * math.sqrt(1 - constant_e**2)
    ) * math.sin(Lat)
    p_b_llh = np.array([[float(Lat)], [float(Long)], [float(h)]])
    if len(arg) == 1:
        return p_b_llh
    if len(arg) == 2:
        v_eb_e = arg[1]
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        R_e_n = np.array(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        # Transform velocity using (2.73)
        v_eb_n = R_e_n @ v_eb_e
        return p_b_llh, v_eb_n
    if len(arg) == 3:
        v_eb_e = arg[1]
        R_b_e = arg[2]
        cos_lat = math.cos(Lat)
        sin_lat = math.sin(Lat)
        cos_lon = math.cos(Long)
        sin_lon = math.sin(Long)
        R_e_n = np.array(
            [
                [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
                [-sin_lon, cos_lon, 0],
                [-cos_lat * cos_lon, -cos_lat * sin_lon, -sin_lat],
            ]
        )
        # Transform velocity using (2.73)
        v_eb_n = R_e_n @ v_eb_e
        R_b_n = R_e_n @ R_b_e
        return p_b_llh, v_eb_n, R_b_n


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小
    Defined in :numref:`sec_calculus`"""


def set_axes(axes, titie, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴
    Defined in :numref:`sec_calculus`"""
    axes.set_title(titie)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


def plot(
    X,
    Y=None,
    title=None,
    xlabel=None,
    ylabel=None,
    legend=None,
    xlim=None,
    ylim=None,
    xscale="linear",
    yscale="linear",
    fmts=None,
    figsize=(3.5, 2.5),
    axes=None,
):
    """绘制数据点
    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []
    set_figsize(figsize)
    axes = axes if axes else plt.gca()
    # 如果`X`有一个轴，输出True
    def has_one_axis(X):
        return (
            hasattr(X, "ndim")
            and X.ndim == 1
            or isinstance(X, list)
            and not hasattr(X[0], "__len__")
        )

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    if fmts:
        for x, y, fmt in zip(X, Y, fmts):
            if len(x):
                axes.plot(x, y, fmt)
            else:
                axes.plot(y, fmt)
    else:
        for x, y in zip(X, Y):
            if len(x):
                axes.plot(x, y)
            else:
                axes.plot(y)
    set_axes(axes, title, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    return axes


def plot3d(X, Y, Z, xlabel=None, ylabel=None, zlabel=None, figsize=(3.5, 2.5)):
    set_figsize(figsize)
    axes = plt.axes(projection="3d")
    axes.plot3D(X, Y, Z)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_ylabel(zlabel)
    axes.set_xscale("linear")
    axes.set_yscale("linear")
    axes.set_zscale("linear")
    return axes


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def norm_normal_dis(x):
    mean = torch.mean(x)
    std = torch.std(x)
    return (x-mean)/(std+1e-16)


def norm_normal_dis_2d(x):
    mean = torch.mean(x, dim=1)
    std = torch.std(x, dim=1)
    return (x-mean.unsqueeze(1))/std.unsqueeze(1)
# def dead_reckoning_ecef(
#     dt, est_pos_new, est_vel_new, est_C_b_e_new, imu, on_cpu: bool = False
# ):
#     if on_cpu:
#         result_dev = est_pos_new.device
#         dev = torch.device("cpu")
#     else:
#         result_dev = est_pos_new.device
#         dev = result_dev
#     # move initial data to dev
#     est_pos_new = est_pos_new.to(dev)
#     est_vel_new = est_vel_new.to(dev)
#     est_C_b_e_new = est_C_b_e_new.to(dev)
#     # imu = imu[:: int(dt * 100), :].to(dev)

#     old_r_eb_e_prev = est_pos_new
#     old_v_eb_e_prev = est_vel_new
#     old_C_b_e_prev = est_C_b_e_new
#     old_v_eb_e_new = None
#     old_v_eb_e_new = None
#     old_C_b_e_new = None
#     for i in range(imu.shape[0]):
#         f_ib_b = imu[i, 0:3]
#         omega_ib_b = imu[i, 3:6]
#         # ATTITUDE UPDATE
#         # From (2.145) determine the Earth rotation over the update interval
#         # C_Earth = C_e_i' * old_C_e_i
#         alpha_ie = torch.tensor(constant_omega_ie * dt, device=dev)
#         C_Earth = torch.tensor(
#             [
#                 [cos(alpha_ie), sin(alpha_ie), 0.0],
#                 [-sin(alpha_ie), cos(alpha_ie), 0.0],
#                 [0.0, 0.0, 1.0],
#             ],
#             device=dev,
#         )
#         # Calculate attitude increment, magnitude, and skew-symmetric matrix
#         alpha_ib_b = omega_ib_b * dt
#         # this produces a warning, we should fix it
#         # mag_alpha = torch.sqrt(alpha_ib_b.detach().T @ alpha_ib_b.detach())
#         mag_alpha = torch.sqrt(alpha_ib_b.T @ alpha_ib_b)
#         Alpha_ib_b = torch.tensor(
#             [
#                 [0, -alpha_ib_b[2], alpha_ib_b[1]],
#                 [alpha_ib_b[2], 0, -alpha_ib_b[0]],
#                 [-alpha_ib_b[1], alpha_ib_b[0], 0],
#             ],
#             device=dev,
#         )
#         # Obtain coordinate transformation matrix from the new attitude w.r.t. an
#         # inertial frame to the old using Rodrigues' formula, (5.73)
#         if mag_alpha > 1.0e-8:
#             C_new_old = (
#                 torch.eye(3, device=dev)
#                 + torch.sin(mag_alpha) / mag_alpha * Alpha_ib_b
#                 + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b @ Alpha_ib_b
#             )
#         else:
#             C_new_old = torch.eye(3, device=dev) + Alpha_ib_b
#         # Update attitude using (5.75)
#         old_C_b_e_new = C_Earth @ old_C_b_e_prev @ C_new_old
#         # SPECIFIC FORCE FRAME TRANSFORMATION
#         # Calculate the average body-to-ECEF-frame coordinate transformation
#         # matrix over the update interval using (5.84) and (5.85)
#         if mag_alpha > 1.0e-8:
#             ave_C_b_e = (
#                 old_C_b_e_prev
#                 @ (
#                     torch.eye(3, device=dev)
#                     + (1 - torch.cos(mag_alpha)) / mag_alpha ** 2 * Alpha_ib_b
#                     + (1 - torch.sin(mag_alpha) / mag_alpha)
#                     / mag_alpha ** 2
#                     * Alpha_ib_b
#                     @ Alpha_ib_b
#                 )
#                 - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ old_C_b_e_prev
#             )
#         else:
#             ave_C_b_e = (
#                 old_C_b_e_prev
#                 - 0.5 * skew_symmetric([0, 0, alpha_ie]).to(dev) @ old_C_b_e_prev
#             )
#         # Transform specific force to ECEF-frame resolving axes using (5.85)
#         f_ib_e = ave_C_b_e @ f_ib_b
#         # UPDATE VELOCITY
#         # From (5.36),
#         old_v_eb_e_new = old_v_eb_e_prev + dt * (
#             f_ib_e
#             + gravity_ECEF(old_r_eb_e_prev)
#             - 2 * skew_symmetric([0, 0, constant_omega_ie]).to(dev) @ old_v_eb_e_prev
#         )
#         # UPDATE CARTESIAN POSITION
#         # From (5.38),
#         old_r_eb_e_new = old_r_eb_e_prev + (old_v_eb_e_new + old_v_eb_e_prev) * 0.5 * dt
#         old_r_eb_e_prev = old_r_eb_e_new
#         old_v_eb_e_prev = old_v_eb_e_new
#         old_C_b_e_prev = old_C_b_e_new
#     return (
#         old_r_eb_e_new.to(result_dev),
#         old_v_eb_e_new.to(result_dev),
#         old_C_b_e_new.to(result_dev),
#     )
