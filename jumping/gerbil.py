import os
if __name__ == '__main__': os.environ['OPENBLAS_NUM_THREADS'] = '1'
import traceback

from cvxopt import matrix, solvers
import cv2
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer as viewer
from mujoco.renderer import Renderer
import numpy as np
import scipy

def traceback_on_exception(func):
  def wrapper(*args, **kwargs):
    try:
      return func(*args, **kwargs)
    except Exception as e:
      print(traceback.format_exc())
      raise e
  return wrapper

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

def lin_reg_impulse(focal_length, t_camera, line_pos, t_force, force_vector, force_offset=0.0):
  # scale height measurements to be relative to the first measurement
  line_pos = line_pos - line_pos[0]

  # scale measurements by camera focal length
  line_pos /= focal_length

  # Offseting the applied by force by an amount approximately equal to the
  # force necessary to overcome gravity greatly improves the numerics of the
  # least squares problem. After solving the least squares problem the result
  # is transformed to remove the effect the offset
  force_vector = np.copy(force_vector) + force_offset

  dt = t_force[1] - t_force[0]
  d_velocity = np.cumulative_sum(force_vector, include_initial=True)[:-1] * dt
  d_position = np.cumulative_sum(d_velocity[1:], include_initial=True) * dt

  # Consider time constants ranging from half the frame period to 0.5 seconds
  dt_camera = np.mean(np.diff(t_camera))
  taus = np.logspace(np.log10(dt_camera*0.5), np.log10(0.5), num=50)
  # taus = [0.01, 0.1, 0.5]
  # taus = [0.1]

  # Make sure the impulse response is 8 time constants greater the highest time constant
  T_impulse = 8*taus[-1] # exp(-8) ~= 0.0335%
  n = int(np.ceil(T_impulse / dt))
  d_position_conv = scipy.linalg.convolution_matrix(d_position, n=n, mode='full')
  d_position_conv = d_position_conv[(n-1):-(n-1), :] # Trim zero padded regions

  tau_matrix = []
  for tau in taus:
    t_exp = np.array([dt*i for i in range(0, n)])
    tau_vector = np.exp(-t_exp / tau)
    tau_matrix.append(tau_vector)
  tau_matrix = np.stack(tau_matrix, axis=1)
  # Normalize each impulse response to have dc gain of 1
  tau_matrix /= np.sum(tau_matrix, axis=0)

  # Calculate convolution of double integral with each impulse response
  d_position_conv_tau = d_position_conv @ tau_matrix

  # Sample at each camera time that is within a non zero padded region
  n_cam = int(np.ceil(T_impulse / dt_camera))
  d_position_conv_tau_sampled = []
  for i in range(d_position_conv_tau.shape[1]):
    d_position_conv_tau_sampled.append(np.interp(t_camera[n_cam-1:], t_force[n-1:], d_position_conv_tau[:, i]))
  d_position_conv_tau_sampled = np.stack(d_position_conv_tau_sampled, axis=1)

  # Set up and solve optimization for vel, gravity, and impulse response coefficients
  # divided by characteristic scale.
  # Don't use least squares directly for numerical reasons
  # Instead, solve the normal equation directly
  A = np.hstack((#np.ones((t_camera.shape[0],))        .reshape((-1, 1))[(n_cam-1):, :],
                 (t_camera               - t_force[0]).reshape((-1, 1))[(n_cam-1):, :],
                 -0.5*np.square(t_camera - t_force[0]).reshape((-1, 1))[(n_cam-1):, :],
                 d_position_conv_tau_sampled))
  b = line_pos[(n_cam-1):]

  Q = (A.T @ A)
  q = -b.T @ A
  # Constrain impulse response coefficients to be positive
  G = -np.eye(Q.shape[0])
  h = np.zeros((Q.shape[0],))
  G = G[2:, :]
  h = h[2:]

  options = { 'show_progress': False }
  sol = solvers.qp(matrix(Q), matrix(q),
                   G=matrix(G), h=matrix(h),
                   options=options)
  xstar_qp = np.array((*sol['x'],))
  xstar_qp_orig = np.copy(xstar_qp) # For plotting

  # Undo the effect of force offset
  xstar_qp[1] += -force_offset * np.sum(tau_matrix @ xstar_qp[2:])

  # Transform estimated quantities to the embodied scale
  v0_over_d = xstar_qp[0]
  gb_over_d = xstar_qp[1]
  g_over_d_coefs = xstar_qp[2:]
  g_over_d = tau_matrix @ g_over_d_coefs
  d_inv = np.sum(g_over_d) # WLOG assume that the DC gain is 1 in embodied units

  d = 1.0 / d_inv
  v0 = v0_over_d * d
  gb = gb_over_d * d
  g = g_over_d * d

  # plt.figure()
  # plt.subplot(211)
  # plt.plot(t_camera, line_pos, label='measured')
  # plt.plot(t_camera[(n_cam-1):], A @ xstar_qp_orig, label='predicted')
  # plt.legend()
  # plt.grid()
  # plt.subplot(212)
  # plt.plot(t_force, force_vector, label='u applied')
  # plt.legend()
  # plt.grid()

  # plt.figure()
  # plt.plot(t_exp, g)

  # plt.figure()
  # plt.plot(taus, xstar_qp_orig[2:], label='c_i', marker='x')
  # plt.show()
  # exit(0)

  return (d, v0, gb), g

def solve_u(s, v_l, g, T, dt):
  # print('G(0j)', np.sum(g))

  # Truncate convolution with g to the section needed
  n = int(np.ceil(T / dt))
  T_g = scipy.linalg.convolution_matrix(g, n=n, mode='full')
  T_g = T_g[:-(g.shape[0]-1), :]

  t_u = np.array([dt*i for i in range(T_g.shape[0])])
  v2 = np.sum(T_g, axis=0) * dt
  v1 = np.sum(np.cumsum(T_g, axis=0) * dt, axis=0) * dt

  Qsqrt = np.zeros((T_g.shape[0], T_g.shape[0]))
  for i in range(T_g.shape[0]):
    for j in range(T_g.shape[0]):
      if i == j:
        Qsqrt[i, j] = -1.0
      if i + 1 == j:
        Qsqrt[i, j] = 1.0
  Q = Qsqrt.T @ Qsqrt
  Q[-1, -1] = 0.0

  q = np.zeros((T_g.shape[0],))

  first_var = np.ones(v1.shape)
  first_var[1:] = 0.0
  last_var = np.ones(v1.shape)
  last_var[:-1] = 0.0

  A = np.stack((first_var, last_var, v1, v2), axis=0)
  b = np.array([0.0, 0.0, s, v_l])
  G = -np.eye(T_g.shape[0])
  h = np.zeros((T_g.shape[0],))
  options = { 'show_progress': False }
  sol = solvers.qp(matrix(Q), matrix(q), A=matrix(A), b=matrix(b), G=matrix(G), h=matrix(h), options=options)
  qp_x = np.array((*sol['x'],))

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

  # v_l_check = np.sum(a_predicted) * dt
  # v_l_check_no_g = np.sum(u) * dt
  # print('v_l_check', v_l, v_l_check, v_l_check_no_g)

  # s_check      = np.sum(np.cumsum((np.convolve(g, u, mode='full')[:-(g.shape[0]-1)])) * dt) * dt
  # s_check_no_g = np.sum(np.cumsum(u) * dt) * dt
  # print('s_check', s, s_check, s_check_no_g)
  # exit(0)

  return t_u, u, a_predicted, v_predicted

class Jumper:
  def __init__(self, renderer, simulated_quant_shift=None, b_scale=None,
               oscillation_T=None, oscillation_hz=None, amplitude=None):
    self.renderer = renderer
    self.simulated_quant_shift = simulated_quant_shift
    self.b_scale = b_scale
    self.oscillation_T  = oscillation_T
    self.oscillation_hz = oscillation_hz
    self.amplitude = amplitude

    if self.oscillation_T is None:
      self.oscillation_T = 10.0
    if self.oscillation_hz is None:
      self.oscillation_hz = 1.0
    if self.amplitude is None:
      self.amplitude = 1.0

    # Render the onboard camera without shadows because of a shadow acne in recent mujoco==3.3.1
    scene = self.renderer.scene
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

    self.finished = False
    self.start_time = 0
    self.render_time = None
    self.t_force_vector = []
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

    self.leg_pid = SimplePID(k_p=200.0, k_i=0.01, k_d=0.4, tau=0.002, tau_setpoint=0.1)

    self.u_gravity = 0.0
    self.u_gravity_ramp = 20.0

    self.t_u = None
    self.u = None

    # For plots
    # self.t_phi = []
    # self.Phi_W = []

    # self.t_u_measure = []
    # self.u_measure = []

    # self.t_ddotx = []
    # self.ddotx = []

    self.gt_v0 = None

    self.jump_traj_t = []
    self.t_jump_end = None
    self.jump_traj = []
    # self.jump_vel = []
    self.jump_traj_u = []

  @traceback_on_exception
  def lift(self, m, d):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    if d.time - self.start_time > 0.5:
      piston_pos = d.joint('left_piston').qpos[0]
      
      self.u_gravity += m.opt.timestep * self.u_gravity_ramp

      self.bottom_position = 0.0 # Joint limit of the piston
      if -(piston_pos - self.bottom_position) > 0.0:
        self.start_time = d.time
        self.leg_pid.integral = -self.u_gravity / self.leg_pid.k_i / self.leg_pid.k_p
        mujoco.set_mjcb_control(lambda m, d: self.measure(m, d))

      d.actuator('left piston') .ctrl[0] = -self.u_gravity
      d.actuator('right piston').ctrl[0] = -self.u_gravity

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
        # print("top position", self.top_position, "bottom position", self.bottom_position)
        mujoco.set_mjcb_control(lambda m, d: self.camera_control(m, d))

  @traceback_on_exception
  def camera_control(self, m, d):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    # Set the initial position of the joint at the first second
    settle_T = 4.0
    piston_pos = d.joint('left_piston').qpos[0]

    # Get the ground truth distance from camera to target for error bounds checking
    pos_line_w = d.site('highlighted_edge').xpos
    t_wc = d.camera('cam').xpos
    R_wc = d.camera('cam').xmat.reshape((3, 3))
    pos_line_c = R_wc.T @ (pos_line_w - t_wc)
    focal_length = ((self.renderer.height - 1) / 2) / np.tan(np.radians(m.cam_fovy[0] / 2))
    pixel_offset = np.array([(self.renderer.width - 1) / 2, (self.renderer.height - 1) / 2])
    gt_pixel_pos = focal_length * np.array([pos_line_c[0] / pos_line_c[2], pos_line_c[1] / pos_line_c[2]])
    gt_pixel_pos += pixel_offset

    if d.time - self.start_time < settle_T:
      alpha = min(1.0, ((d.time - self.start_time) / (0.5 * settle_T)))
      target_pos = alpha*(self.top_position + self.bottom_position) / 2 + (1 - alpha) * self.top_position

      self.leg_pid.set_goal(target_pos)
      u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
      d.actuator('left piston') .ctrl[0] = u
      d.actuator('right piston').ctrl[0] = u

    # oscillate the camera for oscillation_T seconds
    # while taking measurements
    if d.time - self.start_time > settle_T and d.time - self.start_time < settle_T + self.oscillation_T:  
      target_position = (((self.top_position + self.bottom_position) / 2)
                         + (self.amplitude*0.25/2.0) * (self.top_position - self.bottom_position) 
                            * np.sin(self.oscillation_hz * 2*np.pi*(d.time - self.start_time - settle_T)))

      self.leg_pid.set_goal(target_position)
      u = self.leg_pid.update(piston_pos, dt=m.opt.timestep)
      d.actuator('left piston') .ctrl[0] = u
      d.actuator('right piston').ctrl[0] = u

      # self.t_u_measure.append(d.time)
      # self.u_measure.append(u)

      # Take measurements
      self.t_force_vector.append(d.time)

      # self.t_ddotx.append(d.time)
      # self.ddotx.append(d.sensor('acceleration').data[2])

      self.motor_vector.append(-u)

      if self.gt_v0 is None:
        self.gt_v0 = d.body('central_body').cvel[5]

      if self.render_time is None or d.time - self.render_time > 1/60.0:
        self.render_time = d.time
        if self.simulated_quant_shift is None:
          self.renderer.update_scene(d, camera='cam')
          image = self.renderer.render()
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
          # cv2.imshow("image", image)
          # cv2.waitKey(0)

          # obtain mask to isolate highlighted edge 
          image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          lower_red = np.array([0, 128, 128])
          upper_red = np.array([10, 255, 255])
          mask = cv2.inRange(image_hsv, lower_red, upper_red)
          # cv2.imshow("mask", mask)
          # cv2.waitKey(0)

          # # Measure height by averaging height of all red pixels in the middle of the frame
          indices = np.argwhere(mask == 255)
          heights = indices[:, 0]
          if len(heights) == 0:
            raise Exception('No red pixels detected in the image and lin_reg_impulse does not support skipped frames')
          else:
            line_position = np.mean(heights)
            # t_frame = d.time #- self.start_time - settle_T
            self.t_camera.append(d.time)
            self.camera_pos_matrix.append(line_position)
            # self.t_phi.append(d.time)
            # self.Phi_W.append(line_position)
        else:
          # Simulate an integer pixel position with a quantization boundaries controlled by self.simulated_quant_shift
          line_position = np.round(gt_pixel_pos[1] + self.simulated_quant_shift) - self.simulated_quant_shift
          assert np.abs(line_position - gt_pixel_pos[1]) <= 0.5 # Max shift due to quantization is 0.5 pixels

          self.t_camera.append(d.time)
          self.camera_pos_matrix.append(line_position)
          # self.t_phi.append(d.time)
          # self.Phi_W.append(line_position)

    # Perform linear regression after data is gathered
    if d.time - self.start_time > settle_T + self.oscillation_T:
      self.t_camera = np.array(self.t_camera)
      self.camera_pos_matrix = np.array(self.camera_pos_matrix)
      self.t_force_vector = np.array(self.t_force_vector)
      self.motor_vector = np.array(self.motor_vector)        

      coef_motor, g = lin_reg_impulse(focal_length, self.t_camera, self.camera_pos_matrix, self.t_force_vector, self.motor_vector, force_offset=-self.u_gravity)

      # Scale jump stroke by oscillation amplitude scalar so jump controls are about the same
      stroke_distance_motor = np.max(self.camera_pos_matrix * coef_motor[0]/ focal_length) - np.min(self.camera_pos_matrix * coef_motor[0] / focal_length)
      stroke_distance_motor = (stroke_distance_motor / self.amplitude)

      plot_data = {}
      # plot_data['t_phi'] = self.t_phi
      # plot_data['phi'] = self.Phi_W
      # plot_data['t_u_measure'] = self.t_u_measure
      # plot_data['u_measure'] = self.u_measure
      # plot_data['t_ddotx'] = self.t_ddotx
      # plot_data['ddot_x'] = self.ddotx
      plot_data['coef_motor'] = coef_motor
      plot_data['g_over_g0'] = g
      if self.simulated_quant_shift is None:
        # Save gt values for plotting
        gt_b = m.actuator('left piston').gainprm[0] * 2 / m.body('central_body').mass # 2 legs/body mass, neglect mass of legs, as in paper
        gt_d = d.site('highlighted_edge').xpos[1]
        gt_grav = np.abs(m.opt.gravity[2])
        bounds_info = {}
        bounds_info['gt_b'] = gt_b # meters / embodied
        bounds_info['gt_d'] = gt_d # meters
        bounds_info['gt_grav'] = gt_grav # meters
        bounds_info['gt_v0_over_d'] = self.gt_v0 / gt_d # multiples of d
        bounds_info['gt_gb_over_d'] = gt_grav / gt_d # multiples of d
        bounds_info['gt_d_inv'] = gt_b / gt_d # 1 / embodied
        bounds_info['gt_d'] = gt_d / gt_b # embodied
        plot_data['bounds_info'] = bounds_info
      self.plot_data = plot_data

      self.start_time = d.time
      mujoco.set_mjcb_control(lambda m, d: self.jump(m, d, coef_motor, stroke_distance_motor, g))

  @traceback_on_exception
  def jump(self, m, d, model_coef, stroke_distance, g):
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 0] = 0.0
    d.mocap_pos[m.body_mocapid[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'platform_end')], 2] = -3.0

    distance, v_0, grav_bias = model_coef

    # set launch angle
    launch_angle_delay = 3.0
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

      if self.t_u is None:
        # calculate projectile motion
        self.t_land = np.sqrt(2 * distance / (grav_bias * np.tan(self.actual_launch_angle_rad)))
        self.velocity_l = (distance / (self.t_land * np.sin(self.actual_launch_angle_rad)))
        self.acceleration_l = (self.velocity_l ** 2) / (2 * stroke_distance)
        # self.acceleration_time = self.velocity_l / self.acceleration_l
        self.acceleration_time = 0.25
        # print('stroke', stroke_distance)
        # print('velocity_l', self.velocity_l)
        # print('acceleration_l', self.acceleration_l)
        # print('launch angle', self.actual_launch_angle_rad * 180.0 / np.pi)

        self.t_u, self.u, self.a_predicted, self.v_predicted = solve_u(stroke_distance, self.velocity_l, g, self.acceleration_time, m.opt.timestep)

      # Record data for plotting
      self.jump_traj_t.append(d.time)
      self.jump_traj.append(np.copy(d.body('central_body').xpos))
      self.jump_traj_u.append(d.actuator('left piston').ctrl[0])

      if self.start_time + launch_angle_delay + self.acceleration_time >= d.time:
        t_now = d.time - self.start_time - launch_angle_delay
        u_t = np.interp(t_now, self.t_u, self.u) + np.cos(self.actual_launch_angle_rad) * grav_bias
        d.actuator('left piston') .ctrl[0] = -u_t
        d.actuator('right piston').ctrl[0] = -u_t

      elif self.start_time + launch_angle_delay + self.acceleration_time + 4.0 > d.time:
        # Set leg forces to 0
        d.act[:] = 0 # Zero out first order dynamics, as if "switching off" leg
        d.actuator('left piston').ctrl[0]  = 0.0
        d.actuator('right piston').ctrl[0] = 0.0
        if self.t_jump_end == None:
          self.t_jump_end = d.time

      # Wait for jump to complete before saving data and exiting
      else:
        self.jump_traj_t = np.array(self.jump_traj_t)
        self.jump_traj = np.array(self.jump_traj)
        self.jump_traj_u = np.array(self.jump_traj_u)

        self.jump_info = {
          't': self.jump_traj_t,
          't_jump_end': self.t_jump_end,
          'x': self.jump_traj,
          'u': self.jump_traj_u,
        }

        mujoco.set_mjcb_control(None)
        self.finished = True

def build_simulation(res=200, shift=None, dist=None, grav=None, tau=None, b_scale=None,
                     oscillation_T=None, oscillation_hz=None, amplitude=None):
  spec = mujoco.MjSpec.from_file('gerbil.xml')
  if dist is not None:
    spec.body('platform_end').pos[1] = dist
  if grav is not None:
    spec.option.gravity[2] = -grav
  if tau is not None:
    spec.actuator('left piston').dynprm[0] = tau
    spec.actuator('right piston').dynprm[0] = tau
  if b_scale is not None:
    spec.actuator('left piston').gainprm[0] = b_scale
    spec.actuator('right piston').gainprm[0] = b_scale

  m = spec.compile()
  d = mujoco.MjData(m)

  renderer = Renderer(m, width=res, height=res)
  jumper = Jumper(renderer, simulated_quant_shift=shift, 
                  oscillation_T=oscillation_T, oscillation_hz=oscillation_hz,
                  amplitude=amplitude)
  jumper.start_time = d.time
  mujoco.set_mjcb_control(lambda m, d: jumper.lift(m, d))
  return m, d, jumper

@traceback_on_exception
def load_callback(m=None, d=None, build_simulation=build_simulation):
  mujoco.set_mjcb_control(None)
  m, d, jumper = build_simulation()
  mujoco.set_mjcb_control(jumper.lift)
  return m, d

def run_interactive():
  # multiprocessing.set_start_method('spawn')
  viewer.launch(loader=lambda m=None, d=None: load_callback(m, d, build_simulation))

if __name__ == '__main__':
  cv2.setNumThreads(1)
  import matplotlib
  matplotlib.use('TkAgg') # Makes showing debug plots work in control callback work on Linux
  np.set_printoptions(suppress=False, linewidth=200)
  run_interactive()
