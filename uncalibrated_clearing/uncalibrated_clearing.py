###############################################################################
#
# Uncalibrated clearing demo for DJI Robomaster
#
# History:
# 01-03-25 - Levi Burner - Prepared file for release
#
###############################################################################

if __name__ == '__main__':
    import os
    os.environ['OPENBLAS_NUM_THREADS'] = '1' # When importing numpy, by default try to limit the number of threads
    os.environ['DISPLAY'] = ':0' # Use physical display when running over ssh
    import numpy as np
    import cv2
    cv2.setNumThreads(1) # When importing OpenCV, by default try to limit the number of threads
else:
    import numpy as np
    import cv2

from behaviors import (SystemMode, MeasurePatch, ApproachPatch,
                       YawFixate, StopFixate, Stop,
                       size_constraint_init_jax)

from vme_research.hardware.pan_tilt_servo import PanTiltServo
from vme_research.hardware.pi_camera import PiCamera
from vme_research.hardware.djirobomaster import RobomasterIntAcc
from vme_research.messaging.shared_ndarray import SharedNDArrayPubSub, SharedNDArrayPool, SharedNDArrayPipe
from vme_research.hardware.record import Record, Load, make_sequence_directory
from vme_research.algorithms.color_tracker import ColorTracker
from vme_research.algorithms.sample_buffer import SampleBuffer

import argparse
import json
from multiprocessing import Value, Queue, Pool
from queue import Empty
import time

SignalFieldsOptions = {
    'name': 'Signal',
    'version': '0.0.1',
    'fields': [{'name': 'x', 'type': str(np.ndarray), 'split': False}],
    'append_fields': []
}

class TimeSource:
    def __init__(self, sim=False, t0=None):
        self.sim = sim
        self.last_wall_time = time.time()
        if self.sim:
            self.t0 = t0
            if self.t0 is None:
                self.t = Value('f', 0.0)
            else:
                self.t = Value('f', self.t0)
        else:
            self.t0 = time.time()

    def time(self):
        if not self.sim:
            return time.time() - self.t0
        else:
            return self.t.value

    def time_set(self, t):
        self.t.value = t

    def time_increment(self, dt):
        self.t.value += dt
        wall_time = time.time()
        diff = wall_time - self.last_wall_time
        self.last_wall_time = wall_time

def linear_interp(t, sample_data):
    ts, xs = sample_data
    alpha = (t - ts[0]) / (ts[1] - ts[0])
    return alpha * (xs[1] - xs[0]) + xs[0]

# Get bounds of tracked object from a list of pixels
def get_bounds(indices, buffer, recorder, frame_t):
    if indices is not None:
        p_t = np.min(indices[:, 0])
        p_b = np.max(indices[:, 0])
        p_l = np.min(indices[:, 1])
        p_r = np.max(indices[:, 1])
        p_sample = np.array((p_l, p_r, p_t, p_b), dtype=np.float64)
        buffer.append(frame_t, p_sample)
        if recorder is not None: recorder.pub(frame_t, (p_sample,))

def get_bounds_door(ldoor_indices, rdoor_indices, buffer, recorder, frame_t):
    if ldoor_indices is not None and rdoor_indices is not None:
        p_t1 = np.min(ldoor_indices[:, 0])
        p_b1 = np.max(ldoor_indices[:, 0])
        # _ = np.min(ldoor_indices[:, 1])
        p_l = np.max(ldoor_indices[:, 1])

        p_t2 = np.min(rdoor_indices[:, 0])
        p_b2 = np.max(rdoor_indices[:, 0])
        p_r = np.min(rdoor_indices[:, 1])
        # _ = np.max(rdoor_indices[:, 1])

        p_t = (p_t2 + p_t1) / 2.0
        p_b = (p_b2 + p_b1) / 2.0

        p_sample = np.array((p_l, p_r, p_t, p_b), dtype=np.float64)
        buffer.append(frame_t, p_sample)
        if recorder is not None: recorder.pub(frame_t, (p_sample,))

def test_robomaster(cfg, sequence, save, save_signals, t_skip=None):
    # import jax here so all processes do not have to import jax, which takes a while
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "all")
    from vme_research.algorithms.phi_constraint import jaccel_z_scale_constraint_match_phi_tf
    size_constraint_init_jax(jaccel_z_scale_constraint_match_phi_tf, cfg.MeasurePatchConfig)

    # Determine if loading or saving and what the sequence directory is
    loading = True if not save and not save_signals and sequence is not None else False
    if loading:
        sequence_directory = sequence
        print('loading from', sequence_directory)
    elif save or save_signals:
        sequence_directory = make_sequence_directory(cfg.sequences_directory)
        print('saving to', sequence_directory)
    else:
        sequence_directory = None

    # Construct the global time source object
    time_source = TimeSource(sim=loading)
    if loading and t_skip is not None: time_source.time_set(t_skip)

    # Create loading and recording objects
    def loader_if(cond, name):
        if not cond or sequence_directory is None: return None
        return Load(os.path.join(sequence_directory, name))
    loader_camera   = loader_if(loading, 'camera')
    loader_pan_tilt = loader_if(loading, 'pan_tilt')
    loader_robo_u   = loader_if(loading, 'u') # TODO what is this in relation to u_recorder

    def recorder_if(cond, name, time_source, fields_options):
        if not cond or sequence_directory is None: return None
        return Record(os.path.join(sequence_directory, name), time_source=time_source, fields_options=fields_options)
    recorder_camera         = recorder_if(save, 'pan_tilt',   time_source, None)
    recorder_pan_tilt       = recorder_if(save, 'pan_tilt',   time_source, None)
    recorder_robo_u         = recorder_if(save, 'robomaster', time_source, None)
    recorder_touch          = recorder_if(save_signals, 'touch_target',   time_source, SignalFieldsOptions)
    recorder_leftd          = recorder_if(save_signals, 'left_door',      time_source, SignalFieldsOptions)
    recorder_rightd         = recorder_if(save_signals, 'right_door',     time_source, SignalFieldsOptions)
    recorder_door           = recorder_if(save_signals, 'door',           time_source, SignalFieldsOptions)
    embodied_state_recorder = recorder_if(save_signals, 'embodied_state', time_source, SignalFieldsOptions)
    u_recorder              = recorder_if(save_signals, 'u',              time_source, SignalFieldsOptions) # TODO what is this in relation to loader_robo_u?
    setpoint_recorder       = recorder_if(save_signals, 'approach_data',  time_source, SignalFieldsOptions)
    recorder_odometry       = recorder_if(save_signals, 'robomaster_odometry', time_source, None)
    signal_recorders_to_shutdown = [recorder_touch, recorder_leftd, recorder_rightd, recorder_door,
                                    embodied_state_recorder, u_recorder, setpoint_recorder]

    # Construct IPC objects for hardware processes
    stop = Value('i', 0)
    ndarray_pool = SharedNDArrayPool((*cfg.shape_new, 3), np.uint8, cfg.frame_buffer_size, close_queue=Queue())
    def camera_close(t, shm_frame):
        ndarray_pool.close(shm_frame.id)
    camera_buffer = SampleBuffer(sample_close=camera_close)
    camera_pub_sub = SharedNDArrayPubSub(zero_copy=True)
    pan_tilt_pub_sub_u = SharedNDArrayPipe(sample_data=(np.zeros((3,)),), max_messages=cfg.signal_buffer_size)
    u_pub_sub = SharedNDArrayPipe(sample_data=(np.zeros((4,)),), max_messages=cfg.signal_buffer_size)
    u_buffer = SampleBuffer(sample_interp=linear_interp)

    # Construct camera, pan_tilt, and robomaster hardware processes
    camera           = PiCamera(        stop, time_source, loader=loader_camera,   recorder=recorder_camera,   pub_sub=camera_pub_sub,
                                ndarray_pool=ndarray_pool, K=cfg.K_intrinsics, dist=cfg.dist, K_new=cfg.K_new, shape_new=cfg.shape_new,
                                res=cfg.res, fps=cfg.fps, sensor_mode=cfg.sensor_mode)
    pan_tilt         = PanTiltServo(    stop, time_source, loader=loader_pan_tilt, recorder=recorder_pan_tilt, pub_sub_u=pan_tilt_pub_sub_u)
    robomaster_accel = RobomasterIntAcc(stop, time_source, loader=loader_robo_u,   recorder=recorder_robo_u,   pub_sub_u=u_pub_sub,
                                        recorder_odometry=recorder_odometry, drag=cfg.simulated_drag, b=cfg.simulated_b)

    # Construct color tracking objects and buffers
    touchpoint_tracker = ColorTracker(cfg.hsv_touchpoint, max_pixels_change=cfg.max_pixels_change, name='touchpoint', sliders=False)
    ldoor_tracker      = ColorTracker(cfg.hsv_ldoor, max_pixels_change=cfg.max_pixels_change, name='ldoor', sliders=False)
    rdoor_tracker      = ColorTracker(cfg.hsv_rdoor, max_pixels_change=cfg.max_pixels_change, name='rdoor', sliders=False)
    touchpoint_buffer = SampleBuffer(sample_interp=linear_interp)
    ldoor_buffer      = SampleBuffer(sample_interp=linear_interp)
    rdoor_buffer      = SampleBuffer(sample_interp=linear_interp)
    door_buffer       = SampleBuffer(sample_interp=linear_interp)
    bounds_buffers = [touchpoint_buffer, ldoor_buffer, rdoor_buffer, door_buffer]

    # Initialize state machine
    mode_sequence_i = 0
    system_mode      = cfg.mode_sequence[mode_sequence_i][0]
    assert system_mode == SystemMode.STOP
    system_mode_vars = cfg.mode_sequence[mode_sequence_i][1]
    t0 = time_source.time()
    cam_heading = system_mode_vars[0]
    stop_duration = system_mode_vars[1]
    stop_mode = Stop(time_source, t0, cam_heading, stop_duration)

    # Variables for the main loop
    patch_sizes = []
    initialize_sensors = True
    last_frame_t = None
    characteristic_scale_d = None
    d_over_z0 = None
    ldoor_indices = None
    rdoor_indices = None
    running_main_loop = True
    half_body_sizes = []
    half_body_sizes = [0.08309855585, 0.09513195329] # b = 1.0
    # half_body_sizes = [0.05428092534, 0.05605256026] # b = 2.0
    # half_body_sizes = [0.1675156044,  0.1929610126 ] # b = 0.5
    body_size = 0.0

    # Helper function for main loop
    def loop_sleep():
        if loading:
            time_source.time_increment(cfg.sleep_duration)
        else:
            time.sleep(cfg.sleep_duration)

    try:
        pan_tilt.start()
        robomaster_accel.start()
        camera.start()

        while running_main_loop:
            times = []
            frames_retrieved = 0
            # Get new frames and trim the frame buffer
            while camera_pub_sub.size() > 0:
                try:
                    receive_t, frame_t, shm_frame = camera_pub_sub.get(timeout=cfg.sleep_duration)
                    frames_retrieved+=1
                except Empty:
                    time.sleep(cfg.sleep_duration)
                    continue
                camera_buffer.append(frame_t, shm_frame)
            if (frames_retrieved > 1):
                print('skipped', frames_retrieved-1, 'frames')
            if len(camera_buffer) > 0:
                latest_frame_t, _ = camera_buffer[-1]
                camera_buffer.trim(latest_frame_t - cfg.data_buffer_duration)
            if len(camera_buffer) < 2:
                # print('not enough frames to run')
                loop_sleep()
                continue

            # Check for skipped frames in buffer and reset time if necessary
            if np.abs(camera_buffer[-1][0] - camera_buffer[-2][0]) > cfg.camera_lag_threshold:
                print('Sensor lag detected, resetting t0')
                initialize_sensors = True
            if initialize_sensors:
                t_now = time_source.time()
                cam_tf_inc = camera_buffer.set_tf(t_now)
                touchpoint_buffer.inc_t(cam_tf_inc)
                ldoor_buffer.inc_t(cam_tf_inc)
                rdoor_buffer.inc_t(cam_tf_inc)
                door_buffer.inc_t(cam_tf_inc)
                if last_frame_t is not None:
                    last_frame_t += cam_tf_inc
                initialize_sensors = False

            # Get latest frame and check that it is new
            frame_t, shm_frame = camera_buffer.get(camera_buffer[-1][0])
            frame = shm_frame.x
            if last_frame_t is not None and frame_t == last_frame_t:
                loop_sleep()
                continue
            last_frame_t = frame_t

            # Color target tracking
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            touchpoint_indices = touchpoint_tracker.update(frame_hsv, convert_to_hsv=False, visualize=False)
            ldoor_indices      = ldoor_tracker     .update(frame_hsv, convert_to_hsv=False, visualize=False)
            rdoor_indices      = rdoor_tracker     .update(frame_hsv, convert_to_hsv=False, visualize=False)
            get_bounds(touchpoint_indices, touchpoint_buffer, recorder_touch,  frame_t)
            get_bounds(ldoor_indices,      ldoor_buffer,      recorder_leftd,  frame_t)
            get_bounds(rdoor_indices,      rdoor_buffer,      recorder_rightd, frame_t)
            get_bounds_door(ldoor_indices, rdoor_indices, door_buffer, recorder_door, frame_t)

            # Run current behaviour
            if system_mode == SystemMode.MEASURE_PATCH:
                finished, characteristic_scale_d, z_over_d0_new, u, u_gimbal = measure_patch.update()
                if finished:
                    patch_sizes.append(float(characteristic_scale_d))
                    if characteristic_scale_d is None:
                        print('Failed to measure characteristic scale, exiting')
                        running_main_loop=False
                    if d_over_z0 is None:
                        d_over_z0 = z_over_d0_new
            elif system_mode == SystemMode.APPROACH_PATCH:
                finished, half_body_size, u, u_gimbal = approach_patch.update()
                if finished:
                    half_body_sizes.append(float(half_body_size))
                    body_size += half_body_size
            elif system_mode == SystemMode.YAW_FIXATE:
                finished, _, u, u_gimbal = yaw_fixate.update()
            elif system_mode == SystemMode.STOP_FIXATE:
                finished, _, u, u_gimbal = stop_fixate.update()
            elif system_mode == SystemMode.STOP:
                finished, _, u, u_gimbal = stop_mode.update()

            # If current behaviour finished, switch to the next one
            while finished:
                mode_sequence_i += 1
                system_mode = cfg.mode_sequence[mode_sequence_i][0]
                system_mode_vars = cfg.mode_sequence[mode_sequence_i][1]
                t0 = time_source.time()
                if system_mode == SystemMode.MEASURE_PATCH:
                    bounds_buffers[system_mode_vars[1]].trim(t0)
                    measure_patch = MeasurePatch(time_source, t0,
                        bounds_buffer=bounds_buffers[system_mode_vars[1]],
                        u_buffer=u_buffer, K=cfg.K_new, cam_heading=system_mode_vars[0],
                        t_measure=system_mode_vars[2], state_recorder=embodied_state_recorder,
                        jaccel_z_scale_constraint_tf=jaccel_z_scale_constraint_match_phi_tf,
                        cfg=cfg.MeasurePatchConfig)
                    finished=False
                elif system_mode == SystemMode.APPROACH_PATCH:
                    approach_patch = ApproachPatch(time_source, t0,
                        d=characteristic_scale_d, # Assume last d is the correct d to approach
                        d_over_z0=d_over_z0, # Use d_over_z0 to go to Z0 after touching the patch
                        bounds_buffer=bounds_buffers[system_mode_vars[1]], u_buffer=u_buffer, K=cfg.K_new,
                        cam_heading=system_mode_vars[0], T_center=system_mode_vars[2], return_start=system_mode_vars[3],
                        setpoint_recorder=setpoint_recorder, cfg=cfg.ApproachPatchConfig)
                    finished=False
                elif system_mode == SystemMode.YAW_FIXATE:
                    yaw_fixate = YawFixate(time_source, t0,
                        last_cam_heading=system_mode_vars[0], new_cam_heading=system_mode_vars[1],
                        bounds_buffer=bounds_buffers[system_mode_vars[2]],
                        K=cfg.K_new, cfg=cfg.YawFixateConfig)
                    finished=False
                elif system_mode == SystemMode.STOP_FIXATE:
                    stop_fixate = StopFixate(time_source, t0, cam_heading=system_mode_vars[0], 
                        bounds_buffer=bounds_buffers[system_mode_vars[1]],
                        K=cfg.K_new, d=characteristic_scale_d, cfg=cfg.StopFixateConfig)
                    finished=False
                elif system_mode == SystemMode.STOP:
                    stop_mode = Stop(time_source, t0,
                        cam_heading=system_mode_vars[0], duration=system_mode_vars[1],
                        no=system_mode_vars[2], yes=system_mode_vars[3])
                    finished=False
                elif system_mode == SystemMode.CHOOSE_ENTER_DOOR:
                    door_size = characteristic_scale_d # Assume the last measured thing was the door

                    # See config.py for explanation
                    alpha = cfg.ChooseEnterDoorConfig.camera_turn_radius_ratio
                    required_door_size = 2 * alpha / (1 - 2 * alpha) * body_size + 2*np.max(half_body_sizes)

                    if door_size > required_door_size:
                        print('DOOR LARGE ENOUGH, ENTERING', door_size, required_door_size, body_size)
                        mode_sequence_i += system_mode_vars[0] # Increment by X if true
                        finished=True
                    else:
                        print('DOOR TOO SMALL, NOT ENTERING', door_size, required_door_size, body_size)
                        mode_sequence_i += system_mode_vars[1] # Increment by X if false
                        finished=True
                elif system_mode == SystemMode.EXIT:
                    finished=False
                    running_main_loop=False

            # Send the control commands
            # Delay the control commands by u_delay_duration seconds to limit jitter
            u[0] += cfg.u_delay_duration
            u_gimbal += cfg.u_delay_duration
            u_buffer.append(u[0], u[1:])
            u_pub_sub.pub((u,))
            if u_gimbal is not None: pan_tilt_pub_sub_u.pub((u_gimbal,))
            if u_recorder is not None: u_recorder.pub(u[0], (u,))

            # Visualize what the robot is tracking on put on the screen
            frame_display = frame
            if touchpoint_indices is not None:
                frame_display[touchpoint_indices[:, 0], touchpoint_indices[:, 1], :] = (0, 0, 255)
            if ldoor_indices is not None:
                frame_display[ldoor_indices [:, 0], ldoor_indices [:, 1], :] = (0, 255, 0)
            if rdoor_indices is not None:
                frame_display[rdoor_indices [:, 0], rdoor_indices [:, 1], :] = (0, 255, 0)
            if len(door_buffer) > 0:
                _, (p_l, p_r, p_t, p_b) = door_buffer[-1]
                frame_display = np.ascontiguousarray(frame_display)
                cv2.rectangle(frame_display, (int(p_l), int(p_t)), (int(p_r), int(p_b)), (0, 0, 255), thickness=-1)

            cv2.imshow('uncalibrated clearing', frame_display)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            if loading: time_source.time_increment(cfg.sleep_duration)
        # End while loop
    except KeyboardInterrupt:
        pass

    stop.value = 1
    camera.join()
    pan_tilt.join()
    print('waiting for robomaster')
    robomaster_accel.join()

    if save_signals:
        measurements_file = os.path.join(sequence_directory, 'measurements.json')
        measurements_dict = {
            'patch_sizes': patch_sizes,
            'half_body_sizes': half_body_sizes,
        }
        with open(measurements_file, 'w', encoding='utf-8') as f:
            json.dump(measurements_dict, f, ensure_ascii=False, indent=2)
        print('measurements')
        from pprint import pprint
        pprint(measurements_dict)

        import shutil
        config_file = os.path.join(sequence_directory, 'config.py')
        shutil.copy(os.path.abspath(cfg.__file__), config_file)

    for signal_recorder in signal_recorders_to_shutdown:
        if signal_recorder is not None:
            signal_recorder.close()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_signals', action='store_true')
    parser.add_argument('--sequence', type=str, help='Path to sequence folder')
    parser.add_argument('--tskip', type=float, help='Time to skip in recording')
    args = parser.parse_args()

    import config as cfg
    test_robomaster(cfg, args.sequence, args.save, args.save_signals, args.tskip)
