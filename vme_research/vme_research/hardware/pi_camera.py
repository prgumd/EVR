from multiprocessing import Process
import time

import cv2
import mujoco # TODO lazy imports...
import numpy as np

PiCameraFieldsOptions = {
    'name': 'PiCamera',
    'version': '0.0.1',
    'fields': [{'name': 'frame', 'type': str(np.ndarray), 'split': True}],
    'append_fields': [{'name': 'res', 'type': str(list), 'split': False},
                      {'name': 'K', 'type': str(list), 'split': False},
                      {'name': 'dist', 'type': str(list), 'split': False}]
}

class PiCamera(Process):
    def __init__(self,
                 stop,
                 time_source,
                 res=(640,480), fps=60, sensor_mode=None,
                 recorder=None, loader=None,
                 pub_sub=None, pub_sub_sim=None,
                 ndarray_pool = None,
                 K=None, dist=None, K_new=None, shape_new=None):
        super(PiCamera, self).__init__(daemon=True)

        self.stop = stop
        self.time_source = time_source
        # self.device = device
        self.res = res
        self.fps = fps
        self.sensor_mode = sensor_mode
        self.recorder = recorder
        if self.recorder: self.recorder.set_fields_options(PiCameraFieldsOptions)
        self.loader = loader
        self.pub_sub = pub_sub
        self.pub_sub_sim = pub_sub_sim

        assert not (self.pub_sub_sim and self.loader)

        self.ndarray_pool = ndarray_pool

        self.K = K
        self.dist = dist
        self.K_new = K_new
        self.shape_new = shape_new

        if self.K_new is not None:
            self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, np.eye(3), K_new, (shape_new[1], shape_new[0]), cv2.CV_32FC1)

    def post_process(self, receive_t, t, data):
        frame_t = t
        frame = data[0]

        if self.K_new is not None:
            if self.ndarray_pool is None:
                frame_out = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
            else:
                frame_out = self.ndarray_pool.get()
                cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR, dst=frame_out.x)
        else:
            frame_out = frame

        if self.pub_sub is not None:
            self.pub_sub.pub((receive_t, frame_t, frame_out), use_shm=[False, False, True])

    def run(self):
        try:
            if self.loader:
                self.run_loader()
            else:
                self.run_live()
        except KeyboardInterrupt: pass
        finally:
            if self.recorder:
                if self.K is not None:
                    append_values = [list(self.res), self.K.tolist(), self.dist.tolist()]
                else:
                    append_values = [(None, None), None, None]
                self.recorder.close(append_values)

            if self.pub_sub:
                self.pub_sub.cleanup()

    def run_live(self):
        cv2.setNumThreads(1)

        try:
            import os
            os.environ['LIBCAMERA_LOG_LEVELS'] = 'WARN'
            from picamera2 import Picamera2
        except ModuleNotFoundError:
            print('Warning could not import picamera2')

        if self.pub_sub_sim is None:
            self.cam = Picamera2()

            # import pprint
            # pprint.pprint(self.cam.sensor_modes)

            if self.sensor_mode:
                camera_config = self.cam.create_video_configuration(main={'format': 'RGB888', 'size': self.res}, buffer_count=6, raw=self.cam.sensor_modes[self.sensor_mode])
            else:
                camera_config = self.cam.create_video_configuration(main={'format': 'RGB888', 'size': self.res}, buffer_count=6)
            self.cam.configure(camera_config)
            self.cam.video_configuration.controls['FrameRate'] = self.fps
            self.cam.start('video')

        while self.stop.value == 0:
            if self.ndarray_pool is not None and self.K_new is None:
                if self.pub_sub_sim:
                    receive_t, frame_t, frame_cam = self.pub_sub_sim.get()
                    frame = self.ndarray_pool.get() # Unfortunately it seems zero copy is not possible
                    frame.x[:] = frame_cam.x
                else:
                    (frame_cam,), metadata = self.cam.capture_arrays() # This is blocking
                    receive_t = self.time_source.time()
                    frame_t = float(metadata['SensorTimestamp']) / 1e9
                    frame = self.ndarray_pool.get() # Unfortunately it seems zero copy is not possible
                    frame.x[:] = frame_cam
            else:
                if self.pub_sub_sim:
                    receive_t, frame_t, frame = self.pub_sub_sim.get()
                else:
                    (frame,), metadata = self.cam.capture_arrays() # This is blocking
                    receive_t = self.time_source.time()
                    frame_t = float(metadata['SensorTimestamp']) / 1e9

            if self.recorder is not None:
                self.recorder.pub(frame_t, (frame,))

            self.post_process(receive_t, frame_t, (frame,))

            # if self.last_frame_ts is not None:
            #     print((1.0 / (frame_t - self.last_frame_ts)))
            # self.last_frame_ts = frame_t

    def run_loader(self):
        while self.stop.value == 0:
            t = self.time_source.time()
            ret, frame_t, (frame,) = self.loader.get(t)

            if ret:
                if self.ndarray_pool and self.K_new is None:
                    frame_shm = self.ndarray_pool.get()
                    frame_shm.x[:] = frame
                    frame = frame_shm

                self.post_process(t, frame_t, (frame,)) # TODO handle receive t better here
            else:
                time.sleep(0.001)

class PiCameraSimulate():
    def __init__(self, pi_camera, mujoco_name):
        self.mujoco_name = mujoco_name
        self.pub_sub = pi_camera.pub_sub_sim
        self.cam_res = pi_camera.res
        self.fps = pi_camera.fps
        self.last_render_t = None

    def init(self, m, d):
        # Make all the things needed to render a simulated camera
        self.gl_ctx = mujoco.GLContext(*self.cam_res)
        self.gl_ctx.make_current()

        self.scn = mujoco.MjvScene(m, maxgeom=100)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.mujoco_name)

        self.vopt = mujoco.MjvOption()
        self.pert = mujoco.MjvPerturb()

        self.ctx = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_150)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)

        self.viewport = mujoco.MjrRect(0, 0, *self.cam_res)

    def callback(self, m, d):
        if self.last_render_t is None or d.time - self.last_render_t > 1.0 / self.fps:
            # Render the simulated camera
            mujoco.mjv_updateScene(m, d, self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)
            frame = np.empty((self.cam_res[1], self.cam_res[0], 3), dtype=np.uint8) # TODO shm?
            mujoco.mjr_readPixels(frame, None, self.viewport, self.ctx)
            frame = cv2.flip(frame, 0) # OpenGL renders with inverted y axis
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            self.pub_sub.pub((d.time, d.time, frame), use_shm=[False, False, True])
            self.last_render_t = d.time
