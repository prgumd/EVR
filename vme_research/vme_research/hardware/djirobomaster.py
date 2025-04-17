from multiprocessing import Process
import time
import numpy as np

try:
    from robomaster import robot
except ModuleNotFoundError:
    print('Warning could not import DJI RoboMaster SDK')

from vme_research.algorithms.sample_buffer import SampleBuffer

RobomasterIntAccFieldsOptions = {
    'name': 'RobomasterIntAcc',
    'version': '0.0.1',
    'fields': [{'name': 'accel', 'type': str(np.ndarray), 'split': False}],
    'append_fields': []
}

RobomasterOdometryFieldsOptions = {
    'name': 'RobomasterIntAccOdometry',
    'version': '0.0.1',
    'fields': [{'name': 'odom', 'type': str(np.ndarray), 'split': False}],
    'append_fields': []
}

def linear_interp(t, sample_data):
    ts, xs = sample_data
    alpha = (t - ts[0]) / (ts[1] - ts[0])
    return alpha * (xs[1] - xs[0]) + xs[0]

class RobomasterIntAcc(Process):
    def __init__(self,
                 stop,
                 time_source,
                 drag=0.0,
                 b=1.0,
                 recorder=None, loader=None,
                 recorder_odometry=None,
                 pub_sub_u=None, pub_sub_sim=None):
        super(RobomasterIntAcc, self).__init__(daemon=False)

        self.stop = stop
        self.time_source = time_source
        self.drag = drag
        self.b = b
        self.recorder = recorder
        if self.recorder: self.recorder.set_fields_options(RobomasterIntAccFieldsOptions)
        self.recorder_odometry = recorder_odometry
        if self.recorder_odometry: self.recorder_odometry.set_fields_options(RobomasterOdometryFieldsOptions)
        self.loader = loader
        self.pub_sub_u = pub_sub_u
        self.pub_sub_sim = pub_sub_sim
        self.target_hz = 50.0 # TODO what should this be

        self.u_buffer = SampleBuffer(sample_interp=linear_interp)
        self.last_u_time = None
        self.dot_x = np.zeros((3,))

        self.last_pos = np.array((0.0, 0.0, 0.0))
        self.last_att = np.array((0.0, 0.0, 0.0))

    def run(self):
        try:
            if self.loader:
                self.run_loader()
            else:
                self.run_live()
        except KeyboardInterrupt:
            pass
        finally:
            if self.recorder:
                self.recorder.close()

            if self.recorder_odometry:
                self.recorder_odometry.close()

            if self.pub_sub_u:
                self.pub_sub_u.cleanup()

    def odom_pos_sub(self, data):
        try:
            x, y, z = data
            self.last_pos = np.array((x, y, z))
        except Exception as e:
            print(e)
            raise e

    def odom_att_sub(self, data):
        try:
            yaw, pitch, roll = data
            self.last_att = np.array((yaw, pitch, roll))
        except Exception as e:
            print(e)
            raise e

    def run_live(self):
        ep_chassis = None
        try:
            if not self.pub_sub_sim:
                ep_robot = robot.Robot()
                ep_robot.initialize(conn_type='rndis')
                ep_chassis = ep_robot.chassis
            else:
                assert False

            if self.recorder_odometry is not None:
                ep_chassis.sub_position(freq=50, callback=self.odom_pos_sub)
                ep_chassis.sub_attitude(freq=50, callback=self.odom_att_sub)

            last_t = self.time_source.time()
            while self.stop.value == 0:
                if self.pub_sub_u.size() > 0:
                    while self.pub_sub_u.size() > 0:
                        try:
                            data = self.pub_sub_u.get()
                        except Empty:
                            continue

                        if data is None:
                            time.sleep(0.001)
                            continue

                        t_accel = data[0][0]
                        accel   = data[0][1:4]
                        self.u_buffer.append(t_accel, accel)

                tnow = self.time_source.time() # Assumes time always increases
                if self.last_u_time is not None:
                    self.u_buffer.trim(tnow - 0.5)

                if len(self.u_buffer) == 0:
                    # print('RobomasterIntAcc empty buffers after trim', len(self.u_buffer), self.time_source.time())
                    time.sleep(1.0 / self.target_hz)
                    continue

                new_last_u_time, u = self.u_buffer.get(tnow)

                # if self.last_u_time is not None:
                #     print(1.0 / self.target_hz, self.time_source.time() - self.last_u_time)

                if tnow > new_last_u_time:
                    # print('driver', tnow, new_last_u_time, self.u_buffer[-1][0], 'empty')
                    print('robomaster u buffer empty', tnow, self.u_buffer[-1][0], len(self.u_buffer))
                    u = np.zeros((3,))
                self.last_u_time = new_last_u_time

                self.dot_x = self.dot_x + (1.0 / self.target_hz) * (self.b * u - self.drag * self.dot_x)

                if self.pub_sub_sim is None:
                    w_z_deg_s = u[2] * 180.0 / np.pi
                    ep_chassis.drive_speed(x=self.dot_x[0], y=self.dot_x[1], z=w_z_deg_s, timeout=1.0)
                else:
                    assert False

                if self.recorder is not None:
                    self.recorder.pub(self.last_u_time, (u, np.copy(self.dot_x)))

                if self.recorder_odometry is not None:
                    odom_sample = np.array((*self.last_pos, *self.last_att))
                    self.recorder_odometry.pub(tnow, (odom_sample,))

                # if self.pub_sub is not None:
                #     data = np.array((t_0, theta, t_1, tau))
                #     self.pub_sub.pub((data,))

                t_end = self.time_source.time()
                delta = t_end - last_t
                if delta < 1.0 / self.target_hz:
                    time.sleep((1.0 / self.target_hz) - delta)
                last_t = self.time_source.time()
        except KeyboardInterrupt:
            pass
        finally:
            if ep_chassis is not None:
                ep_chassis.drive_speed(x=0, y=0, z=0, timeout=0.5)
                if self.recorder_odometry:
                    ep_chassis.unsub_position()
                    ep_chassis.unsub_attitude()
                ep_robot.close()

    def run_loader(self):
        while self.stop.value == 0:
            time.sleep(0.1)
            continue

if __name__ == '__main__':
    from multiprocessing import Value
    from vme_research.messaging.shared_ndarray import SharedNDArrayPipe


    class TimeZeroSource:
        def __init__(self, t0=None):
            self.t0 = t0
            if self.t0 is None: self.t0 = time.time()

        def time(self):
            return time.time() - self.t0


    stop = Value('i', 0)
    time_source = TimeZeroSource()
    u_pub_sub = SharedNDArrayPipe(sample_data=(np.zeros((4,)),), max_messages=1000)
    robomaster_accel = RobomasterIntAcc(stop, time_source, drag=1.5, pub_sub=u_pub_sub)
    robomaster_accel.start()

    try:
        w = 0.25
        a = 0.5

        while time_source.time() < 8.0:
            t = time_source.time()

            if time_source.time() < 2.0:
                u = np.array((t+0.2, a, 0.0, 0.0))
                u_pub_sub.pub((u,))
            else:
                u = np.array((t+0.2, 0.0, 0.0, 0.0))
            # u = np.array((t+0.2, a*np.cos(2*np.pi*w * t), 0.0, 0.0))
            time.sleep(0.040)
    except KeyboardInterrupt:
        pass

    stop.value = True
    robomaster_accel.join()
