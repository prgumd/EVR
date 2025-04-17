###############################################################################
#
# Configuration for uncalibrated clearing
#
# History:
# 01-03-25 - Levi Burner - Prepared file for release
#
########################################################################

from behaviors import SystemMode
import numpy as np

# Directory to save sequences in
sequences_directory = '/media/pi/26E4-ACFB/uncalibrated_clearing'

# Hardware settings
res_x = int(4608/3)
res_y = int(2592/3)
res = (res_x, res_y)
sensor_mode = 1
fps = 30.0

K_intrinsics = np.array([[706.86524047,   0.,         785.28273951 ],
                         [  0.,         708.68750649,  437.12065929],
                         [  0.,           0.,           1.        ]])
dist = np.array([-0.23668399, 0.05504181, -0.00068037, 0.00074841, -0.00577425])

scale_x = 1.0
scale_y = 0.125
shape_new = (int(res_y*scale_y), int(res_x*scale_x))
scale_new = (shape_new[0] / res_y, shape_new[1] / res_x)
K_scale = np.array([[scale_new[1], 0.0, 0.0],
                    [0.0, scale_new[0], 0.0],
                    [0.0, 0.0, 1.0]])
K_new = K_scale @ K_intrinsics

# drag and b are used in the robot's communication process to simulate acceleration control
# using the robot's underlying velocity controller. Since the robot's internal software uses
# meters, b is the ground truth conversion factor from embodied units to meters. Note that
# the uncalibrated clearing algorithm does not know what b is.
simulated_drag = 0.1
simulated_b = 1.0

data_buffer_duration = 2.0
sleep_duration = 0.001
u_delay_duration = 0.1

frame_buffer_size = 200
signal_buffer_size = 1000
camera_lag_threshold = 0.5

# Color Tracker settings
max_pixels_change = 30

# Drone lab
hsv_touchpoint = ((152, 175), ( 63, 255), (167, 255))
hsv_rdoor      = (( 20,  31), (200, 255), (177, 255))
hsv_ldoor      = (( 33,  49), (130, 255), (167, 255))

# PRG lab
# hsv_touchpoint = ((152, 169), ( 72, 255), (203, 255))
# hsv_rdoor      = ((110, 128), ( 79, 255), (170, 255))
# hsv_ldoor      = (( 33,  45), (145, 255), (180, 255))

# Sequence of behaviours
mode_sequence = [
    (SystemMode.STOP,           (90.0, 2.0, False, False)), # Camera faces forward, rest for 2 seconds before starting
    # (SystemMode.MEASURE_PATCH,  (90.0, 0, 10.0)),           # Camera faces forward, measure touch target for 10 seconds
    # (SystemMode.STOP_FIXATE,    (90.0, 0)),                 # Camera faces forward, stop using touch target and its (now) known size
    # (SystemMode.YAW_FIXATE,     (90.0, 180.0, 0)),          # Rotate camera from forward to right while fixating on touch target
    # (SystemMode.APPROACH_PATCH, (180.0, 0, 0.0, True)),     # Camera faces right, approach touch target and return, center on for 0.0, return to start
    # (SystemMode.YAW_FIXATE,     (180.0, 0.0, 0)),           # Turn camera from right to left while fixatign on touch target
    # (SystemMode.APPROACH_PATCH, (0.0, 0, 0.0, True)),       # Camera faces left, approach touch target and return, center on for 0.0 seconds, return to start
    # (SystemMode.STOP_FIXATE,    (0.0, 0)),                  # Camera faces left, stop using touch target and its (now) known size
    # (SystemMode.STOP,           (90.0, 0.5, False, False)), # Camera faces forward, wait 1 seconds
    (SystemMode.MEASURE_PATCH,  (90.0, 3, 10.0)),           # Camera faces forward (towards opening), measure opening size for 10 seconds
    (SystemMode.CHOOSE_ENTER_DOOR, (0, 2)),                 # If robot fits, skip 0 commands, if robot does not fit, skip next two commands
    (SystemMode.APPROACH_PATCH, (90.0, 3, 1.0, False)),     # If fits, enter door, center on for 1.0 seconds, do not return to start
    (SystemMode.EXIT, None),                                # If fits, exit program
    (SystemMode.STOP_FIXATE,    (90.0, 3)),                 # If does not fit, camera faces forward, stop using opening target and its (now) known size
    (SystemMode.STOP,           (90.0, 2.0, True, False)),  # If does not fit, say no by shaking camera
    (SystemMode.EXIT, None),                                # If does not fit, exit program
]

# Behaviour configurations
class MeasurePatchConfig:
    est_T = 3.0
    osc_hz = 1 / 3.0
    osc_a = 0.25 * (2*np.pi*osc_hz)**2 # Amplitude of sinusoidal acceleration
    min_sample_ratio = 0.95
    expected_fps = fps

class ApproachPatchConfig:
    approach_speed = 0.3
    pid_p = 1.0
    pid_d = 2.5
    pid_tau = 0.15
    T_settle = 3.0

class YawFixateConfig:
    yaw_rate = 75.0
    yaw_K = 0.0 # Disable closed loop yaw
    max_err_norm_pixels = 1.0 # Disable closed loop yaw

class StopFixateConfig:
    min_tstop = 0.1
    stop_v = 0.01
    stop_x = 0.1
    stop_T = 0.2
    z_filt_tau = 0.15
    K_stop = 1.5
    x_pid_p = 1.0
    x_pid_d = 2.5
    x_pid_tau = 0.15

class ChooseEnterDoorConfig:
    # When determining if the opening is large enough, lower bound the size of the opening
    # assuming the camera's turning radius is less than 15% of the body width
    # Note: the camera's turning radius could be measured by turning in place during fixation
    camera_turn_radius_ratio = 0.15 
