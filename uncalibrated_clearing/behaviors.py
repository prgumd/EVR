###############################################################################
#
# Behaviours (motion controllers) for uncalibrated clearing
#
# History:
# 01-03-25 - Levi Burner - Prepared file for release
#
###############################################################################

from vme_research.algorithms.simple_pid import SimplePID
import numpy as np
from enum import Enum

SystemMode = Enum('SystemMode',
    ['MEASURE_PATCH',
     'APPROACH_PATCH',
     'YAW_FIXATE',
     'STOP_FIXATE',
     'STOP',
     'CHOOSE_ENTER_DOOR',
     'EXIT'
    ])

def size_constraint(x_c_buffer, u_buffer, K, state_recorder, jaccel_z_scale_constraint_tf, T, tf, expected_fps):
    # Sample the buffer at equal intervals
    ts = np.arange(tf-T, tf, 1.0 / expected_fps)
    x_c_list   = [x_c_buffer.get(t) for t in ts]
    x_c_list_t = [sample[0] for sample in x_c_list]
    x_c_list   = [sample[1] for sample in x_c_list]
    u_list     = [u_buffer.get(t) for t in ts]
    u_list_t   = [sample[0] for sample in u_list]
    u_list     = [sample[1] for sample in u_list]
    x_c_t = np.array(x_c_list_t)
    x_c   = np.array(x_c_list)
    u_t   = np.array(u_list_t)
    u     = np.array(u_list)

    d_over_z = ((x_c[:, 1] - x_c[:, 0]) / K[0, 0])
    z_over_d = 1 / d_over_z
    accel_z = -u[:, 0]

    x, res, residuals_full = jaccel_z_scale_constraint_tf(x_c_t, z_over_d, accel_z)
    if state_recorder is not None: state_recorder.pub(x_c_t[-1], (x,))
    # d = 0.15
    # print('z_over_d', z_over_d[-1], 'd', d, 'z', z_over_d[-1] * d)
    return x[0]

# Attempt to trigger compilation of kernel ahead of time
def size_constraint_init_jax(jaccel_z_scale_constraint_tf, cfg):
    N = np.arange(0.0, cfg.est_T, 1.0 / cfg.expected_fps).shape[0]
    x_c_t = np.linspace(0.0, 1.0, N)
    z_over_d = np.linspace(0.0, 1.0, N)
    accel_z = np.linspace(0.0, 1.0, N)
    jaccel_z_scale_constraint_tf(x_c_t, z_over_d, accel_z)

class MeasurePatch:
    def __init__(self, time_source, t0, bounds_buffer, u_buffer, K, cam_heading, t_measure, state_recorder,
                 jaccel_z_scale_constraint_tf, cfg):
        self.time_source = time_source
        self.t0 = t0
        self.bounds_buffer = bounds_buffer
        self.u_buffer = u_buffer
        self.K = K
        self.cam_heading = cam_heading
        self.t_measure = t_measure
        self.state_recorder = state_recorder
        self.jaccel_z_scale_constraint_tf = jaccel_z_scale_constraint_tf
        self.cfg = cfg
        self.d_over_z0 = None

    def update(self):
        # make sure there are some samples
        buffers_good = True
        if len(self.bounds_buffer) == 0 or len(self.u_buffer) == 0:
            print('empty buffers')
            buffers_good = False
        else:
            # make sure there is data for the entire interval
            tmin = max(self.bounds_buffer[ 0][0], self.u_buffer[ 0][0])
            tf   = min(self.bounds_buffer[-1][0], self.u_buffer[-1][0])

        if buffers_good and tf - tmin < self.cfg.est_T:
            # print('buffers do not cover interval')
            buffers_good = False

        # Make sure that large numbers of frames were not dropped
        if buffers_good:
            x_c_list = self.bounds_buffer.get(tf-self.cfg.est_T, tf)
            expected_samples = self.cfg.est_T * self.cfg.expected_fps
            if len(x_c_list[0]) < self.cfg.min_sample_ratio * expected_samples:
                print('x_c_list too many skipped samples', len(x_c_list[0]), expected_samples, self.cfg.min_sample_ratio * expected_samples)
                buffers_good = False

        if buffers_good:
            d = size_constraint(self.bounds_buffer, self.u_buffer, self.K, self.state_recorder, self.jaccel_z_scale_constraint_tf, self.cfg.est_T, tf, self.cfg.expected_fps)
        else:
            d = None

        if self.d_over_z0 is None and len(self.bounds_buffer) > 0:
            _, x_c = self.bounds_buffer[-1]
            self.d_over_z0 = ((x_c[1] - x_c[0]) / self.K[0, 0])

        # Move back and forth
        t_now = self.time_source.time()
        t_signal = t_now - self.t0

        u_cam = np.array((self.cfg.osc_a*np.cos(2*np.pi*self.cfg.osc_hz * t_signal), 0.0))

        # print(t_signal, u_cam[0])

        theta = -(self.cam_heading - 90.0) * np.pi / 180.0
        R_body_cam = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
        u_body = R_body_cam @ u_cam

        w_z = 0.0
        u = np.array((t_now, *u_body, w_z))
        u_gimbal = np.array((t_now, 90.0, self.cam_heading))

        if self.time_source.time() - self.t0 > self.t_measure:
            print('Patch Size!', d)
            return True, d, self.d_over_z0, u, u_gimbal
        else:
            # print('Current size!', d)
            return False, None, self.d_over_z0, u, u_gimbal

class ApproachPatch:
    def __init__(self, time_source, t0, d, d_over_z0, bounds_buffer, u_buffer, K,
                 cam_heading, T_center, return_start,
                 setpoint_recorder, cfg):
        self.time_source = time_source
        self.t0 = t0
        self.d = d
        self.d_over_z0 = d_over_z0
        self.bounds_buffer = bounds_buffer
        self.u_buffer = u_buffer
        self.K = K
        self.cam_heading = cam_heading
        self.setpoint_recorder = setpoint_recorder
        self.cfg = cfg
        self.x_pid = SimplePID(k_p=self.cfg.pid_p, k_d=self.cfg.pid_d, tau=self.cfg.pid_tau)
        self.z_pid = SimplePID(k_p=self.cfg.pid_p, k_d=self.cfg.pid_d, tau=self.cfg.pid_tau)
        self.last_t_x_c = None
        self.body_size = None
        self.z_t = None
        self.z_t_backup = self.d / self.d_over_z0
        self.T_center = T_center
        self.T_approach = None
        self.T_backup = self.z_t_backup / self.cfg.approach_speed
        self.return_start = return_start
        self.backup_impulse = False

    def update(self):
        u = np.array((self.time_source.time(), 0.0, 0.0, 0.0))

        # Current patch position
        t_x_c, x_c = self.bounds_buffer[-1]
        x_over_z = (x_c[1] - x_c[0]) / 2.0 + x_c[0]
        x_over_z = (x_over_z - self.K[0, 2]) / self.K[0, 0]
        y_over_z = (x_c[3] - x_c[2]) / 2.0 + x_c[2]
        y_over_z = (y_over_z - self.K[1, 2]) / self.K[1, 1]
        w = ((x_c[1] - x_c[0]) / self.K[0, 0])
        z_over_d = 1 / w

        x = x_over_z * z_over_d * self.d
        y = y_over_z * z_over_d * self.d
        z = z_over_d * self.d

        if self.T_approach is None:
            self.T_approach = z / self.cfg.approach_speed
            self.z_t = z

        tnow = self.time_source.time()
        if self.last_t_x_c is not None:
            # Current patch position control to approach the patch at a constant rate
            tdelta = tnow - self.t0
            if tdelta < self.T_center: # For T_center seconds center on the target at the current distance
                z_t = self.z_t
                x_t = 0.0
            elif tdelta < self.T_center + self.T_approach: # Drive at the target and assume it was hit
                x_t = 0.0
                z_t = self.z_t - (tdelta - self.T_center) * self.cfg.approach_speed
            elif tdelta < self.T_center + self.T_approach + self.T_backup + self.cfg.T_settle:
                # Move back away from the target to the original position and stop
                z_t = self.z_t_backup * ((tdelta - self.T_center - self.T_approach) / self.T_backup)
                z_t = min(self.z_t_backup, z_t)
                x_t = 0.0
            else:
                z_t = self.z_t_backup
                x_t = 0.0

            self.x_pid.set_goal(x_t)
            self.z_pid.set_goal(z_t)

            u_x = self.x_pid.update(x, dt=t_x_c - self.last_t_x_c)
            u_z = self.z_pid.update(z, dt=t_x_c - self.last_t_x_c)

            if (self.return_start and not self.backup_impulse
                and tdelta > self.T_center + self.T_approach):
                u_z += 2*(self.cfg.approach_speed / (t_x_c - self.last_t_x_c))
                self.backup_impulse = True

            if self.body_size is None or z < self.body_size:
                self.body_size = z
            # print('{:0.3f} {:0.3f} {:0.3f} {:0.3f} {:0.3f}'.format(tdelta - self.T_center - self.T_approach, z_t, z, u_z, self.body_size))

            u_cam = np.array((-u_z, -u_x))
            theta = -(self.cam_heading - 90.0) * np.pi / 180.0
            R_body_cam = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
            u_body = R_body_cam @ u_cam
            u = np.array((tnow, *u_body, 0.0))

            if self.setpoint_recorder is not None:
                body_size_rec = self.body_size if self.body_size is not None else 0.0
                setpoint_recorder_sample = np.array((x_t, z_t, x, z, u_x, u_z, body_size_rec))
                self.setpoint_recorder.pub(tnow, (setpoint_recorder_sample,))
        else:
            u = np.array((tnow, 0.0, 0.0, 0.0))
        self.last_t_x_c = t_x_c

        u_gimbal = np.array((tnow, 90.0, self.cam_heading))

        if self.return_start:
            T_finished = self.T_center + self.T_approach + self.T_backup + self.cfg.T_settle
        else:
            T_finished = self.T_center + self.T_approach

        if tnow - self.t0 > T_finished:
            if self.return_start: print('Body size', self.body_size)
            return True, self.body_size, u, u_gimbal
        else:
            return False, None, u, u_gimbal

class YawFixate:
    def __init__(self, time_source, t0, last_cam_heading, new_cam_heading, bounds_buffer, K, cfg):
        self.time_source = time_source
        self.t0 = t0
        self.last_cam_heading = last_cam_heading
        self.new_cam_heading = new_cam_heading
        self.bounds_buffer = bounds_buffer
        self.K = K
        self.cfg = cfg
        self.yaw_duration = abs((self.new_cam_heading - self.last_cam_heading)) / self.cfg.yaw_rate

    def update(self):
        tnow = self.time_source.time() - self.t0

        # Yaw and fixate
        alpha = tnow / self.yaw_duration
        alpha_clipped = min(alpha, 1.0)

        cam_heading = self.new_cam_heading * alpha_clipped + self.last_cam_heading * (1.0 - alpha_clipped)
        u_gimbal = np.array((self.time_source.time(), 90.0, cam_heading))

        if alpha <= 1.0:
            w_z_ff = (np.pi / 180.0) * np.sign(self.new_cam_heading - self.last_cam_heading) * self.cfg.yaw_rate
        else:
            w_z_ff = 0.0

        # Yaw to fixate on the target
        if len(self.bounds_buffer) > 0:
            t_bounds, (p_l, p_r, p_t, p_b) = self.bounds_buffer[-1]
            mid_x = ((p_r - p_l) / 2.0) + p_l
            mid_x = (mid_x - self.K[0, 2]) / self.K[0, 0]
        else:
            mid_x = 0.0
        w_z = self.cfg.yaw_K * mid_x
        w_z_final = w_z + w_z_ff
        u = np.array((tnow+self.t0, 0.0, 0.0, w_z_final))
        # print('yaw', u[0], alpha, w_z_ff, w_z, w_z_final)

        if alpha > 1.0 and abs(mid_x) < self.cfg.max_err_norm_pixels:
            return True, None, u, u_gimbal
        else:
            return False, None, u, u_gimbal

class StopFixate:
    def __init__(self, time_source, t0, cam_heading, bounds_buffer, K, d, cfg):
        self.time_source = time_source
        self.t0 = t0
        self.cam_heading = cam_heading
        self.bounds_buffer = bounds_buffer
        self.K = K
        self.d = d
        self.cfg = cfg

        self.stop_v_T = None
        self.stop_x_T = None
        self.z_filt = None
        self.last_t_x_c = None
        self.x_pid = SimplePID(k_p=self.cfg.x_pid_p, k_d=self.cfg.x_pid_d, tau=self.cfg.x_pid_tau)

    def update(self):
        tnow = self.time_source.time() - self.t0

        # Current patch position
        t_x_c, x_c = self.bounds_buffer[-1]
        x_over_z = (x_c[1] - x_c[0]) / 2.0 + x_c[0]
        x_over_z = (x_over_z - self.K[0, 2]) / self.K[0, 0]
        y_over_z = (x_c[3] - x_c[2]) / 2.0 + x_c[2]
        y_over_z = (y_over_z - self.K[1, 2]) / self.K[1, 1]
        w = ((x_c[1] - x_c[0]) / self.K[0, 0])
        z_over_d = 1 / w

        x = x_over_z * z_over_d * self.d
        y = y_over_z * z_over_d * self.d
        z = z_over_d * self.d

        if self.z_filt is None:
            self.z_filt = z

        if self.last_t_x_c is not None:
            dt = t_x_c - self.last_t_x_c
            z_filt_update = (dt / self.cfg.z_filt_tau) * (z - self.z_filt)
            self.z_filt += z_filt_update
            dz_dt = z_filt_update / dt

            u_z = -self.cfg.K_stop * dz_dt
            self.x_pid.set_goal(0.0)
            u_x = self.x_pid.update(x, dt=t_x_c - self.last_t_x_c)
            # print(tnow, 'dz', dz_dt, 'x', x, 'u_z', u_z, 'u_x', u_x)
            # print(tnow, 'dz', dz_dt, 'dt', dt, 'z', z, 'z_filt', self.z_filt, 'x', x, 'u_z', u_z, 'u_x', u_x)

            u_cam = np.array((-u_z, -u_x))
            theta = -(self.cam_heading - 90.0) * np.pi / 180.0
            R_body_cam = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
            u_body = R_body_cam @ u_cam
            u = np.array((tnow+self.t0, *u_body, 0.0))
        else:
            dz_dt = 0.0
            u = np.array((tnow+self.t0, 0.0, 0.0, 0.0))
        self.last_t_x_c = t_x_c

        u_gimbal = np.array((tnow+self.t0, 90.0, self.cam_heading))

        if self.stop_v_T is None or abs(dz_dt) > self.cfg.stop_v: self.stop_v_T = tnow
        if self.stop_x_T is None or abs(x    ) > self.cfg.stop_x: self.stop_x_T = tnow

        if (tnow > self.cfg.min_tstop
            and tnow - self.stop_v_T > self.cfg.stop_T
            and tnow - self.stop_x_T > self.cfg.stop_T):
            return True, None, u, u_gimbal
        else:
            return False, None, u, u_gimbal

class Stop:
    def __init__(self, time_source, t0, cam_heading, duration, no=False, yes=False):
        self.time_source = time_source
        self.t0 = t0
        self.cam_heading = cam_heading
        self.duration = duration
        assert not (yes and no)
        self.no = no
        self.yes = yes

    def update(self):
        u = np.array((self.time_source.time(), 0.0, 0.0, 0.0))

        if not self.no and not self.yes:
            u_gimbal = np.array((self.time_source.time(), 90.0, self.cam_heading))
        elif self.no:
            t_sequence = self.time_source.time() - self.t0
            u_gimbal = np.array((self.time_source.time(), 90.0, 90.0 + 22.5 * np.sin(2*np.pi * t_sequence)))
        elif self.yes:
            t_sequence = self.time_source.time() - self.t0
            u_gimbal = np.array((self.time_source.time(), 90.0 + 22.5 * np.sin(2*np.pi * t_sequence), 90.0))

        if self.duration > 0 and self.time_source.time() - self.t0 > self.duration:
            return True, None, u, u_gimbal
        else:
            return False, None, u, u_gimbal
