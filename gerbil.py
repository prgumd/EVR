import multiprocessing
from multiprocessing import Process
import traceback

from cvxopt import matrix, solvers
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

import mujoco
import mujoco.viewer as viewer
from mujoco.renderer import Renderer

class SimplePID:
  def __init__(self,k_p=1.,k_i=0.,k_d=0., dt=None, bounds=None,tau=0.1, tau_in=None, tau_setpoint=None):
    self.k_p=k_p; self.k_i=k_i; self.k_d=k_d;self.bd=bounds
    self.dt=dt
    self.goal = 0.0; self.response=0.0
    self.deriv_filter=0.0; self.integral=0.0
    self.error_filter=0.0
    self.setpoint_filter=0.0
    self.tau=tau
    self.tau_in=tau_in
    self.tau_setpoint=tau_setpoint
  def set_goal(self,goal):
    self.goal=goal
  def update(self,value,dt=None):
    if dt is not None:
      self.dt = dt

    if self.tau_setpoint is not None:
      self.setpoint_filter += (self.dt/self.tau_setpoint)*(self.goal - self.setpoint_filter)
    else:
      self.setpoint_filter = self.goal

    if self.tau_in is not None:
      self.error_filter += (self.dt/self.tau_in)*((self.setpoint_filter - value) - self.error_filter)
      error = self.error_filter
    else:
      error = self.setpoint_filter - value

    last_deriv_filter = self.deriv_filter
    self.deriv_filter += (self.dt/self.tau)*(error - self.deriv_filter)
    deriv = (self.deriv_filter - last_deriv_filter) / self.dt

    self.integral += self.dt*error
    self.response = self.k_p*(error + self.k_i*self.integral + self.k_d*deriv)
    if self.bd is not None:
      self.response=max(self.bd*-1,min(self.bd,self.response))
    return self.response
  def reset(self):
    self.deriv_filter=0.0;self.integral=0.0
  def __str__(self) -> str:
    return f"({self.deriv_filter:.3f}, {self.goal:.3f}, {self.response:.3f}) "


def lin_reg(focal_length, t_camera, line_pos, t_force, force_vector):
  # scale height measurements to be relative to the first measurement
  line_pos = line_pos - line_pos[0]
  # scale measurements by camera focal length
  line_pos /= focal_length

  dt = t_force[1] - t_force[0]
  d_velocity = np.cumsum(force_vector) * dt
  d_position = np.cumsum(d_velocity) * dt

  sampled_d_position = np.interp(t_camera, t_force, d_position) # Sample d_position at t_camera
  camera_matrix = np.stack((line_pos,
                            -(t_camera - t_force[0]),
                            0.5*np.square(t_camera - t_force[0])), axis=1) # Construct camera matrix
  x, res, rank, s = np.linalg.lstsq(camera_matrix, sampled_d_position, rcond=None)
  # print(x)
  # print(np.linalg.norm(res))
  return x

def lin_reg_impulse(focal_length, t_camera, line_pos, t_force, force_vector):
  # scale height measurements to be relative to the first measurement
  line_pos = line_pos - line_pos[0]
  # scale measurements by camera focal length
  line_pos /= focal_length

  dt = t_force[1] - t_force[0]
  d_velocity = scipy.integrate.cumulative_trapezoid(force_vector, initial=0.0) * dt
  d_position = scipy.integrate.cumulative_trapezoid(d_velocity, initial=0.0) * dt

  sampled_d_position = np.interp(t_camera, t_force, d_position) # Sample d_position at t_camera

  # dt_camera = t_camera[1] - t_camera[0]
  dt_camera = np.mean(np.diff(t_camera))
  print('dt_camera', dt_camera)
  n = int(np.ceil(2.0 / dt_camera)) # 1 second impulse response
  T_sampled_d_position = scipy.linalg.convolution_matrix(sampled_d_position, n=n, mode='full')
  T_sampled_d_position = T_sampled_d_position[(n-1):-(n - 1), :]
  # print('T_sampled_d_position')
  # print(T_sampled_d_position)

  taus = np.logspace(-4, 0, num=1000)
  tau_matrix = []
  for tau in taus:
    t_exp = np.array([dt_camera*i for i in range(0, n)])
    tau_vector = np.exp(-t_exp / tau)
    tau_matrix.append(tau_vector)
  tau_matrix = np.stack(tau_matrix, axis=1)

  # print(sampled_d_position.shape, T_sampled_d_position.shape, line_pos.shape)
  camera_matrix = np.hstack(((t_camera               - t_force[0]).reshape((-1, 1))[(n-1):, :],
                             -0.5*np.square(t_camera - t_force[0]).reshape((-1, 1))[(n-1):, :],
                             T_sampled_d_position @ tau_matrix)) # Construct camera matrix
  # Don't use least squares directly for numerical reasons
  # Instead, project into the column space of A even if not doing ridge regression
  # x, res, rank, s = np.linalg.lstsq(camera_matrix, line_pos[n-1:], rcond=-1)
  lamb = 0.00
  A = camera_matrix
  b = line_pos[(n-1):]
  P = np.identity(A.shape[1])
  P[0, 0] = 0.0 # Don't penalize velocity and gravity terms
  P[1, 1] = 0.0
  # x, res, rank, s = np.linalg.lstsq(A.T @ A + lamb * P, A.T @ b, rcond=None)
  # print('x')

  Q = (A.T @ A + lamb * P)
  q = -b.T @ A
  # A = np.stack((v1, v2), axis=0)
  # b = np.array([s, v_l])
  # Constrain impulse response coefficients to be positive
  G = -np.eye(Q.shape[0])
  h = np.zeros((Q.shape[0],))
  G = G[2:, :]
  h = h[2:]
  options = { 'show_progress': False }
  sol = solvers.qp(matrix(Q), matrix(q),
                   #A=matrix(A), b=matrix(b),
                   G=matrix(G), h=matrix(h),
                   options=options)
  qp_x = np.array((*sol['x'],))

  # print('qp')
  # print(sol)
  # print(qp_x)
  # print(qp_x - x)
  # exit(0)
  x = qp_x
  print('x raw', x)

  coefs = x[2:]
  # print('coefs', coefs)
  g = tau_matrix @ coefs
  dc_gain = np.sum(g)
  print('dc_gain_inv', 1.0/dc_gain)

  g = g / dc_gain
  d = 1.0 / dc_gain
  v0 = x[0] / dc_gain
  gb = x[1] / dc_gain


  # plt.figure()
  # plt.subplot(311)
  # plt.plot(t_camera, line_pos, label='measured')
  # plt.plot(t_camera[(n-1):], camera_matrix @ qp_x, label='predicted')
  # plt.legend()
  # plt.grid()
  # plt.subplot(312)
  # plt.plot(t_force, force_vector, label='u applied')
  # plt.legend()
  # plt.grid()

  # plt.subplot(313)
  # plt.plot(t_camera[(n-1):], camera_matrix[:, 0] * qp_x[0], label='pos')
  # plt.plot(t_camera[(n-1):], camera_matrix[:, 1] * qp_x[1],  label='vel')
  # # plt.plot(t_camera[(n-1):], camera_matrix[:, 2:] * qp_x[2:], label='conv')
  # plt.legend()
  # plt.grid()

  # plt.figure()
  # # plt.plot(taus, coefs, '-x')
  # plt.plot(g)
  # plt.show()

  # print(d, v0, gb)
  # print(np.linalg.norm(b - A @ x))
  # x[0] = 0.0
  # print(np.linalg.norm(b - A @ x))
  return (d, v0, gb), g

def solve_u(s, v_l, g_camera, T, dt_camera, dt):
  # t_interp = np.arange(0.0, T, dt)
  # t_interp = np.arange(0.0, (g_camera.shape[0] - 1)*dt_camera, dt)
  t_interp = np.array([i*dt for i in range(g_camera.shape[0] * int(np.floor(dt_camera / dt)))])
  t_camera = np.array([i*dt_camera for i in range(g_camera.shape[0])])
  g = np.interp(t_interp, t_camera, g_camera) # Sample g_camera at finer intervals
  g = g * np.sum(g_camera) / np.sum(g)

  # print('g')
  # print(g)
  print('G(0j)', np.sum(g), g.shape[0]*dt)

  # print('g_camera')
  # print(g_camera)
  # print('G_camera(0j)', np.sum(g_camera), g_camera.shape[0]*dt_camera, g.shape[0] / g_camera.shape[0], dt / dt_camera)

  # Truncate g to the section needed
  n = int(np.ceil(T / dt))
  T_g = scipy.linalg.convolution_matrix(g, n=n, mode='full')
  T_g = T_g[:-(g.shape[0]-1), :]

  print('T_g')
  print(T_g)
  print(T_g.shape)

  t_u = np.array([dt*i for i in range(T_g.shape[0])])
  # print('t_u')
  # print(t_u)
  v2 = np.sum(T_g, axis=0) * dt
  v1 = np.sum(np.cumsum(T_g, axis=0) * dt, axis=0) * dt

  print('v2')
  print(v2)
  print(v2.shape)

  print('v1')
  print(v1)
  print(v1.shape)

  # Closed form without inequality constraint or
  # A = np.array((((np.linalg.norm(v1)**2, np.dot(v1, v2)),
  #                (np.dot(v1, v2), np.linalg.norm(v2)**2))))
  # l = np.linalg.solve(A, np.array((s, v_l)))
  # print('l', l)
  # u = l[0] * v1 + l[1] * v2
  # print('u')
  # print(u)
  # print('diff')

  # Q = np.eye(T_g.shape[0])

  Qsqrt = np.zeros((T_g.shape[0], T_g.shape[0]))
  for i in range(T_g.shape[0]):
    for j in range(T_g.shape[0]):
      if i == j:
        Qsqrt[i, j] = -1.0
      if i + 1 == j:
        Qsqrt[i, j] = 1.0
  print('Qsqrt')
  print(Qsqrt)
  Q = Qsqrt.T @ Qsqrt
  Q[-1, -1] = 0.0

  q = np.zeros((T_g.shape[0],))

  first_var = np.ones(v1.shape)
  first_var[1:] = 0.0
  last_var = np.ones(v1.shape)
  last_var[:-1] = 0.0

  # A = []
  # b = []
  # A.append(first_var)
  # b.append(0.0)
  # for i in range(500):
  #   constraint = np.zeros((v1.shape[0],))
  #   constraint[v1.shape[0] - i - 1] = 1
  #   A.append(constraint)
  #   b.append(0.0)
  # A.append(v1)
  # A.append(v2)
  # b.append(s)
  # b.append(v_l)
  # A = np.array(A)
  # b = np.array(b)

  A = np.stack((first_var, last_var, v1, v2), axis=0)
  b = np.array([0.0, 0.0, s, v_l])
  G = -np.eye(T_g.shape[0])
  h = np.zeros((T_g.shape[0],))
  options = { 'show_progress': False }
  sol = solvers.qp(matrix(Q), matrix(q), A=matrix(A), b=matrix(b), G=matrix(G), h=matrix(h), options=options)
  qp_x = np.array((*sol['x'],))

  print('qp')
  print(sol)
  print(qp_x)
  # print(qp_x - u)
  u = qp_x #+ gb

  # print(s, v1 @ u)
  # print(v_l, v2 @ u)

  a_predicted = np.convolve(g, u, mode='full')[:-(g.shape[0]-1)]
  v_predicted = np.cumsum(a_predicted) * dt

  # print(t_u.shape)
  # print(a_predicted.shape)
  # plt.plot(t_u, a_predicted, label='pred')
  # plt.plot(t_u, u, label='u')
  # plt.legend()
  # plt.grid()
  # plt.show()

  v_l_check = np.sum(a_predicted) * dt
  v_l_check_no_g = np.sum(u) * dt
  print('v_l_check', v_l, v_l_check, v_l_check_no_g)

  s_check      = np.sum(np.cumsum((np.convolve(g, u, mode='full')[:-(g.shape[0]-1)])) * dt) * dt
  s_check_no_g = np.sum(np.cumsum(u) * dt) * dt
  print('s_check', s, s_check, s_check_no_g)
  # exit(0)

  return t_u, u, a_predicted, v_predicted

def traceback_on_exception(func):
    def wrapper(*args, **kwargs):
        try:
          return func(*args, **kwargs)
        except Exception as e:
          print(traceback.format_exc())
          raise e
    return wrapper

class Jumper:
  def __init__(self, renderer):
    self.renderer = renderer

    self.start_time = 0
    self.render_time = None
    self.t_force_vector = []
    self.acceleration_vector = []
    self.t_camera = []
    self.camera_pos_matrix = []
    self.render_counter = 0
    self.motor_vector = []
    self.coef = 0
    self.last_coef = 0

    self.target_position = None
    self.measure_velocity = -0.3
    self.bottom_position = None
    self.top_position = None
    self.low_pass_piston = None
    self.low_pass_piston_tau = 0.2


    # Jumping settings
    self.launch_angle = 20.0
    self.actual_launch_angle_rad = self.launch_angle
    self.jump_traj = []
    self.jump_vel = []

    self.leg_pid = SimplePID(k_p=200.0, k_i=0.01, k_d=0.4, tau=0.002, tau_setpoint=0.1)

    self.u_gravity = 0.0
    self.u_gravity_ramp = 20.0

    self.t_u = None
    self.u = None

    self.t_phi = []
    self.Phi_W = []

    self.t_u_measure = []
    self.u_measure = []

    self.t_ddotx = []
    self.ddotx = []

    # self.coefs = None

    self.measure_vars = []

  @traceback_on_exception
  def lift(self, m, d):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    if d.time - self.start_time > 0.5:
      piston_pos = d.joint('left_piston').qpos[0]
      if self.bottom_position is None:
        self.bottom_position = piston_pos
      
      self.u_gravity += m.opt.timestep * self.u_gravity_ramp

      if abs(piston_pos - self.bottom_position) > 0.01:
        self.start_time = d.time
        self.leg_pid.integral = -self.u_gravity / self.leg_pid.k_i / self.leg_pid.k_p
        mujoco.set_mjcb_control(lambda m, d: self.measure(m, d))

      d.actuator('left piston') .ctrl[0] = -self.u_gravity
      d.actuator('right piston').ctrl[0] = -self.u_gravity

  # @traceback_on_exception
  # def settle(self, m, d):
  #   piston_pos = d.joint('left_piston').qpos[0]
  #   self.leg_pid.set_goal(-0.25)
  #   u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
  #   d.actuator('left piston') .ctrl[0] = u
  #   d.actuator('right piston').ctrl[0] = u

  @traceback_on_exception
  def measure(self, m, d):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    if d.time - self.start_time > 0.5:
      piston_pos = d.joint('left_piston').qpos[0]

      if self.target_position is None:
        self.target_position = piston_pos

      self.target_position += m.opt.timestep * self.measure_velocity
      self.leg_pid.set_goal(self.target_position)
      u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
      d.actuator('left piston') .ctrl[0] = u
      d.actuator('right piston').ctrl[0] = u

      if self.bottom_position is None or piston_pos > self.bottom_position:
        self.bottom_position = piston_pos

      if self.top_position is None or piston_pos < self.top_position:
        self.top_position = piston_pos

      if self.low_pass_piston is None:
        self.low_pass_piston = piston_pos

      self.low_pass_piston += m.opt.timestep * (piston_pos - self.low_pass_piston) / (self.low_pass_piston_tau)
      high_pass_piston = piston_pos - self.low_pass_piston

      if d.time - self.start_time > 2.5 and abs(high_pass_piston) < 0.1:
        self.start_time = d.time
        print("top position", self.top_position, "bottom position", self.bottom_position)
        mujoco.set_mjcb_control(lambda m, d: self.camera_control(m, d))

  @traceback_on_exception
  def camera_control(self, m, d):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    # Set the initial position of the joint at the first second
    settle_T = 4.0
    piston_pos = d.joint('left_piston').qpos[0]

    if d.time - self.start_time < settle_T:
      alpha = min(1.0, ((d.time - self.start_time) / (0.5 * settle_T)))
      target_pos = alpha*(self.top_position + self.bottom_position) / 2 + (1 - alpha) * self.top_position

      self.leg_pid.set_goal(target_pos)
      u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
      d.actuator('left piston') .ctrl[0] = u
      d.actuator('right piston').ctrl[0] = u

    # oscillate the camera for oscillation_T seconds
    # while taking measurements
    oscillation_T  = 5.0
    oscillation_hz = 1.0
    if d.time - self.start_time > settle_T and d.time - self.start_time < settle_T + oscillation_T:  
      target_position = (((self.top_position + self.bottom_position) / 2)
                         + (0.25/2.0) * (self.top_position - self.bottom_position) 
                            * np.sin(oscillation_hz * 2*np.pi*(d.time - self.start_time - settle_T)))
      piston_pos = d.joint('left_piston').qpos[0]

      self.leg_pid.set_goal(target_position)
      u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
      d.actuator('left piston') .ctrl[0] = u
      d.actuator('right piston').ctrl[0] = u

      self.t_u_measure.append(d.time)
      self.u_measure.append(u)

      # Take measurements
      self.t_force_vector.append(d.time)
      self.acceleration_vector.append(d.sensor('acceleration').data[2])

      self.t_ddotx.append(d.time)
      self.ddotx.append(d.sensor('acceleration').data[2])

      self.motor_vector.append(-u)

      if self.render_time is None or d.time - self.render_time > 1/60.0:
        self.renderer.update_scene(d, camera=0)
        image = self.renderer.render()

        # Show the simulated camera image
        image     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # obtain mask to isolate highlighted edge 
        lower_red = np.array([0, 128, 128])
        upper_red = np.array([5, 255, 255])
        mask = cv2.inRange(image_hsv, lower_red, upper_red)
        thresholded_image = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("threshold", thresholded_image)
        # cv2.waitKey(1)

        # Measure height by averaging height of all red pixels in the middle of the frame
        indices = np.argwhere(mask == 255)
        heights = indices[:, 0]
        if len(heights) > 0: 
          line_position = np.mean(heights)
          t_frame = d.time #- self.start_time - settle_T
          self.t_camera.append(d.time)
          self.camera_pos_matrix.append(line_position)

          self.t_phi.append(d.time)
          self.Phi_W.append(line_position)

        self.render_time = d.time

    # Perform linear regression after data is gathered
    if d.time - self.start_time > settle_T + oscillation_T:
      self.t_camera = np.array(self.t_camera)
      self.camera_pos_matrix = np.array(self.camera_pos_matrix)
      self.t_force_vector = np.array(self.t_force_vector)
      self.acceleration_vector = np.array(self.acceleration_vector)
      self.motor_vector = np.array(self.motor_vector)
      focal_length = (self.renderer.height / 2) / np.tan(np.radians(m.cam_fovy[0] / 2))

      print('accel')
      coef       = lin_reg(focal_length, self.t_camera, self.camera_pos_matrix, self.t_force_vector, self.acceleration_vector)
      # print('motor before')
      # _ = lin_reg(focal_length, self.t_camera, self.camera_pos_matrix, self.t_force_vector, self.motor_vector)
      print('motor')
      coef_motor, g = lin_reg_impulse(focal_length, self.t_camera, self.camera_pos_matrix, self.t_force_vector, self.motor_vector)

      stroke_distance       = np.max(self.camera_pos_matrix * coef[0] / focal_length)      - np.min(self.camera_pos_matrix * coef[0] / focal_length)
      stroke_distance_motor = np.max(self.camera_pos_matrix * coef_motor[0]/ focal_length) - np.min(self.camera_pos_matrix * coef_motor[0] / focal_length)

      print('coef_motor / coef accel')
      print(coef_motor / coef, stroke_distance_motor / stroke_distance)
      print('coef_motor')
      print(coef_motor, stroke_distance_motor)
      print('coef accel')
      print(coef, stroke_distance)
      # exit(0)
      coef_ratio = coef_motor[0] / coef[0]

      dt_camera = np.mean(np.diff(self.t_camera))

      plot_data = {}
      plot_data['t_phi'] = self.t_phi
      plot_data['phi'] = self.Phi_W
      plot_data['t_u_measure'] = self.t_u_measure
      plot_data['u_measure'] = self.u_measure
      plot_data['t_ddotx'] = self.t_ddotx
      plot_data['ddot_x'] = self.ddotx
      plot_data['g_over_g0'] = g
      plot_data['coef_motor'] = coef_motor
      np.savez_compressed('plot_data.npz', **plot_data)

      self.start_time = d.time
      mujoco.set_mjcb_control(lambda m, d: self.jump(m, d, coef_motor, stroke_distance_motor, coef_ratio, g, dt_camera))

  @traceback_on_exception
  def jump(self, m, d, model_coef, stroke_distance, coef_ratio, g, dt_camera):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    distance, v_0, grav_bias = model_coef

    # set launch angle
    launch_angle_delay = 2.0
    if d.time - self.start_time < launch_angle_delay:
      current_launch_angle = min(self.launch_angle, ((d.time - self.start_time) / (0.8*launch_angle_delay))*self.launch_angle)
      d.actuator('left ankle').ctrl[0]  = -current_launch_angle * np.pi / 180.0
      d.actuator('right ankle').ctrl[0] = -current_launch_angle * np.pi / 180.0
      d.actuator('left piston').ctrl[0]  = 0
      d.actuator('right piston').ctrl[0] = 0
      self.actual_launch_angle_rad = -d.joint('left_ankle').qpos[0]
      d.actuator('left piston') .ctrl[0] = -0.9*np.cos(self.actual_launch_angle_rad) * grav_bias
      d.actuator('right piston').ctrl[0] = -0.9*np.cos(self.actual_launch_angle_rad) * grav_bias
    else:

      self.jump_traj.append((d.time, *d.body('central_body').xpos))
      self.jump_vel.append((d.time, *d.sensor('velocity').data))

      if self.t_u is None:
        # calculate projectile motion
        self.t_land = np.sqrt(2 * distance / (grav_bias * np.tan(self.actual_launch_angle_rad)))
        self.velocity_l = (distance / (self.t_land * np.sin(self.actual_launch_angle_rad)))
        self.acceleration_l = (self.velocity_l ** 2) / (2 * stroke_distance)
        # self.acceleration_time = self.velocity_l / self.acceleration_l
        self.acceleration_time = 0.25
        # u = acceleration_l + np.cos(self.actual_launch_angle_rad) * grav_bias

        print('stroke', stroke_distance)
        print('velocity_l', self.velocity_l)
        print('acceleration_l', self.acceleration_l)
        print('launch angle', self.actual_launch_angle_rad * 180.0 / np.pi)

        self.t_u, self.u, self.a_predicted, self.v_predicted = solve_u(stroke_distance, self.velocity_l, g, self.acceleration_time, dt_camera, m.opt.timestep)

      if self.start_time + launch_angle_delay + self.acceleration_time >= d.time:
        t_now = d.time - self.start_time - launch_angle_delay
        u_t = np.interp(t_now, self.t_u, self.u) + np.cos(self.actual_launch_angle_rad) * grav_bias
        # print('launch', t_now, self.t_u[0], self.t_u[-1], u_t)

        # a_pred_t = np.interp(t_now, self.t_u, self.a_predicted)  + np.cos(self.actual_launch_angle_rad) * grav_bias
        # v_pred_t = np.interp(t_now, self.t_u, self.v_predicted)
        # print(t_now, u_t, -a_pred_t, d.actuator('left piston').force[0],
        #       v_pred_t, np.linalg.norm(d.sensor('velocity').data) * coef_ratio, d.act)

        # print('angle', self.actual_launch_angle_rad * 180.0 / np.pi, -d.joint('left_ankle').qpos[0] * 180.0 / np.pi)

        d.actuator('left piston') .ctrl[0] = -u_t
        d.actuator('right piston').ctrl[0] = -u_t
        # d.actuator('left piston').ctrl[0]  = -self.acceleration_l
        # d.actuator('right piston').ctrl[0] = -self.acceleration_l

        a_b = d.sensor('acceleration').data
        v_b = d.sensor('velocity').data
      elif self.start_time + launch_angle_delay + self.acceleration_time + 4.0 > d.time:
        # Set leg forces to 0
        d.act[:] = 0 # Zero out first order dynamics, as if "switching off" leg
        d.actuator('left piston').ctrl[0]  = 0.0
        d.actuator('right piston').ctrl[0] = 0.0
        # print(self.velocity_l, np.linalg.norm(d.sensor('velocity').data) * coef_ratio, d.act)
      else:
        print("v_l", self.velocity_l,
              "a_l", self.acceleration_l,
              # "u", u,
              "T", self.acceleration_time,
              "d", stroke_distance)
        # mujoco.set_mjcb_control(None)
        mujoco.set_mjcb_control(lambda m, d: self.plot_results(
            m, d, distance, v_0, grav_bias, self.actual_launch_angle_rad,
            self.velocity_l, self.acceleration_l, self.acceleration_time, coef_ratio, g, self.t_u, self.u))

  def plot_results(self, m, d, distance, v_0, grav_bias, launch_angle_rad, velocity_l, acceleration_l, acceleration_time, coef_ratio, g, t_u, u_jump):
    jump = np.array(self.jump_traj)
    jump_vel = np.array(self.jump_vel)
    plot(jump, jump_vel, launch_angle_rad, velocity_l, grav_bias, coef_ratio, g, t_u, u_jump)
    mujoco.set_mjcb_control(None)

def plot(*args):
  p = Process(target=plot_data_proc, args=(*args,))
  p.start()
  # p.join()

def plot_data_proc(jump, jump_vel, launch_angle_rad, velocity_l, grav_bias, coef_ratio, g, t_u, u_jump):
  t = jump[:, 0] - jump[0, 0]
  dt = t[1] - t[0]

  jump_theory_doty = np.sin(launch_angle_rad) * velocity_l + t * 0
  jump_theory_dotz = np.cos(launch_angle_rad) * velocity_l - t * grav_bias

  jump_theory_y = jump[0, 2] + np.cumsum(jump_theory_doty) * dt
  jump_theory_z = jump[0, 3] + np.cumsum(jump_theory_dotz) * dt

  jump_theory_y = jump_theory_y / coef_ratio
  jump_theory_z = jump_theory_z / coef_ratio

  jump[:, 2] = jump[:, 2] - jump[0, 2]
  jump[:, 3] = jump[:, 3] - jump[0, 3]

  jump_theory_y = jump_theory_y - jump_theory_y[0]
  jump_theory_z = jump_theory_z - jump_theory_z[0]

  plt.figure()
  # plt.subplot(311)
  # plt.plot(t, jump[:, 1], label='x')
  # plt.legend()
  # plt.grid()
  plt.subplot(211)
  plt.plot(t, jump[:, 2], label='y')
  plt.plot(t+t_u[-1], jump_theory_y, label='yt')
  plt.legend()
  plt.grid()
  plt.subplot(212)
  plt.plot(t, jump[:, 3], label='z')
  plt.plot(t+t_u[-1], jump_theory_z, label='zt')
  plt.legend()
  plt.grid()

  plt.figure()
  plt.plot(jump[:, 2], jump[:, 3], label='acheived')
  plt.plot(jump_theory_y, jump_theory_z, label='target')
  plt.grid()
  plt.legend()

  plt.figure()
  plt.plot(t, np.gradient(jump[:, 2]) / dt, label='achieved vy')
  plt.plot(t, np.gradient(jump[:, 3]) / dt, label='achieved vz')
  plt.plot(t, jump_theory_doty / coef_ratio, label='target vy')
  plt.plot(t, jump_theory_dotz / coef_ratio, label='target vz')
  plt.grid()
  plt.legend()

  # plt.figure()
  # plt.subplot(211)
  # plt.plot(t, np.linalg.norm(jump_vel[:, 1:], axis=1) * coef_ratio)

  # plt.subplot(212)
  # plt.plot(t, np.linalg.norm(np.gradient(jump_vel[:, 1:], axis=0) / dt * coef_ratio, axis=1))

  plt.figure()
  plt.subplot(211)
  plt.plot((0.0, *np.cumsum(g)))
  # plt.plot((0.0, *np.cumsum(g)))
  plt.subplot(212)
  plt.plot(t_u, u_jump)

  plot_data = {}
  plot_data['t'] = t
  plot_data['y'] = jump[:, 2]
  plot_data['z'] = jump[:, 3]
  plot_data['y_theory'] = jump_theory_y
  plot_data['z_theory'] = jump_theory_z

  plot_data['t_u'] = t_u
  plot_data['u_jump'] = u_jump
  np.savez_compressed('plot_data_jump.npz', **plot_data)

  plt.show()

@traceback_on_exception
def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('gerbil.xml')
  d = mujoco.MjData(m)

  if m is not None:
    renderer = Renderer(m, width=600, height=600)
    jumper = Jumper(renderer)
    jumper.start_time = d.time
    mujoco.set_mjcb_control(lambda m, d: jumper.lift(m, d))

  return m , d

if __name__ == '__main__':
  multiprocessing.set_start_method('spawn')
  viewer.launch(loader=load_callback)
