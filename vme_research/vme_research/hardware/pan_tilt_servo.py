from multiprocessing import Process
import time
import numpy as np
from vme_research.algorithms.sample_buffer import SampleBuffer
from queue import Empty

PanTiltServoFieldsOptions = {
    'name': 'PanTiltServo',
    'version': '0.0.1',
    'fields': [{'name': 'frame', 'type': str(np.ndarray), 'split': False}],
    'append_fields': []
}

def linear_interp(t, sample_data):
    ts, xs = sample_data
    alpha = (t - ts[0]) / (ts[1] - ts[0])
    return alpha * (xs[1] - xs[0]) + xs[0]

def convert_adc_to_degrees(adc, min, max):
  return 180 * (adc - min) / (max - min)

class PanTiltServo(Process):
    def __init__(self,
                 stop,
                 time_source,
                 recorder=None, loader=None, simulate=None,
                 pub_sub=None, pub_sub_u=None, pub_sub_sim=None):
        super(PanTiltServo, self).__init__(daemon=True)

        self.stop = stop
        self.time_source = time_source
        self.recorder = recorder
        if self.recorder: self.recorder.set_fields_options(PanTiltServoFieldsOptions)
        self.loader = loader
        self.pub_sub = pub_sub
        self.pub_sub_u = pub_sub_u
        self.pub_sub_sim = pub_sub_sim
        self.target_hz = 50
        assert self.target_hz >= 50.0 and self.target_hz <= 200.0
        self.u_buffer = SampleBuffer(sample_interp=linear_interp)
        self.last_u_time = None

    def run(self):
        try:
            if self.loader:
                self.run_loader()
            else:
                self.run_live()
        except KeyboardInterrupt: pass
        finally:
            if self.recorder:
                self.recorder.close()

            if self.pub_sub:
                self.pub_sub.cleanup()

    def run_live(self):
        try:
            from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_IIC as Board
            from DFRobot_RaspberryPi_Expansion_Board import DFRobot_Expansion_Board_Servo as Servo
        except ModuleNotFoundError:
            print('Warning could not import DFRobot_RaspberryPi_Expansion_Board')

        if not self.pub_sub_sim:
            self.board = Board(1, 0x10)
            self.servo = Servo(self.board)

            while self.board.begin() != self.board.STA_OK:
                print('failed to initialize dfrobot pi expansion')

            self.board.set_pwm_enable()                # Pwm channel need external powe
            self.board.set_pwm_frequency(self.target_hz)
            self.board.set_adc_enable()

        self.motor_0_min_adc = 3768.2
        self.motor_0_max_adc = 341.0
        self.motor_1_min_adc = 3757.24
        self.motor_1_max_adc = 341.0
        
        # print("m0min: ", self.motor_0_min_adc)
        # print("m0max: ", self.motor_0_max_adc)
        # print("m1min: ", self.motor_1_min_adc)
        # print("m1max: ", self.motor_0_max_adc)

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

                    t_u = data[0][0]
                    u   = data[0][1:]
                    self.u_buffer.append(t_u, u)

            tnow = self.time_source.time()
            if self.last_u_time is not None:
                self.u_buffer.trim(tnow-0.5)

            if len(self.u_buffer) > 0:
                new_last_u_time, u = self.u_buffer.get(tnow)

                # if self.last_u_time is not None:
                #     print(1.0 / self.target_hz, self.time_source.time() - self.last_u_time)

                self.last_u_time = new_last_u_time
                if tnow > new_last_u_time:
                    print('ptz u buffer empty', tnow, self.u_buffer[-1][0])
                else:
                    # a = 30.0
                    # w = 0.5 * 2 * np.pi
                    # t = self.time_source.time()
                    # motor_0_degrees = a * np.sin(w * t) + 90
                    # motor_1_degrees = a * np.cos(w * t) + 90
                    motor_0_degrees = 2.0 * (u[0] - 90.0) + 90.0 # TODO why
                    motor_1_degrees = 2.0 * (u[1] - 90.0) + 90.0
                    if not self.pub_sub_sim:
                        self.board.set_pwm_duty(0, ((1000.0 + 1000.0 * (motor_0_degrees / 180.0)) * 1e-6) * self.target_hz * 100)
                        self.board.set_pwm_duty(1, ((1000.0 + 1000.0 * (motor_1_degrees / 180.0)) * 1e-6) * self.target_hz * 100)
            # else:
            #     print('empty')

            if self.pub_sub_sim is None:
                adc_val_0 = self.board.get_adc_value(self.board.A0)
                t_0 = self.time_source.time()

                adc_val_1 = self.board.get_adc_value(self.board.A1)
                t_1 = self.time_source.time()
            else:
                t_0, adc_val_0, adc_val_1 = self.pub_sub_sim.get()
                t_1 = t_0

            theta = convert_adc_to_degrees(adc_val_0, self.motor_0_min_adc, self.motor_0_max_adc)
            tau = convert_adc_to_degrees(adc_val_1, self.motor_1_min_adc, self.motor_1_max_adc)

            if self.recorder is not None:
                self.recorder.pub(t_0, ((t_0, theta, t_1, tau),))

            if self.pub_sub is not None:
                data = np.array((t_0, theta, t_1, tau))
                self.pub_sub.pub((data,))

            delta = self.time_source.time() - last_t
            if delta < 1.0 / self.target_hz:
                time.sleep((1.0 / self.target_hz) - delta)
            last_t = self.time_source.time()

    def run_loader(self):
        while self.stop.value == 0:
            t = self.time_source.time()
            ret, t_0, data = self.loader.get(t)
            if ret:
                if self.pub_sub is not None:
                    self.pub_sub.pub(data)
            else:
                time.sleep(0.001)
