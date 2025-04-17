###############################################################################
#
# Solve various problems using the Phi constraint and Jax
#
# History:
# 06-18-24 - Levi Burner - Created file
#
###############################################################################

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jR

import jaxopt
from jaxopt import GaussNewton

def jcumulative_int(dt, x):
    return dt * jnp.cumsum(x)

def jcumulative_trapezoid(dt, x):
    steps = 0.5 * (x[:-1] + x[1:])
    return jnp.concatenate((np.array([0.0,]), jcumulative_int(dt, steps)))

def jaccel_xyz_scale_constraint(times, x_over_z, y_over_z, z_over_d, accel_x, accel_y, accel_z):
    # z_over_z0 = z_over_z0 / z_over_z0[0]

    x_over_d = x_over_z * z_over_d
    # dx_over_d -= dx_over_d[0]

    y_over_d = y_over_z * z_over_d
    # dy_over_d -= dy_over_d[0]

    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]
    # int_accel_x  = jcumulative_int(dt, accel_x)
    # iint_accel_x = jcumulative_int(dt, int_accel_x)

    # int_accel_y  = jcumulative_int(dt, accel_y)
    # iint_accel_y = jcumulative_int(dt, int_accel_y)

    # int_accel_z  = jcumulative_int(dt, accel_z)
    # iint_accel_z = jcumulative_int(dt, int_accel_z)

    int_accel_x  = jcumulative_trapezoid(dt, accel_x)
    iint_accel_x = jcumulative_trapezoid(dt, int_accel_x)

    int_accel_y  = jcumulative_trapezoid(dt, accel_y)
    iint_accel_y = jcumulative_trapezoid(dt, int_accel_y)

    int_accel_z  = jcumulative_trapezoid(dt, accel_z)
    iint_accel_z = jcumulative_trapezoid(dt, int_accel_z)

    Px = x_over_d
    Py = y_over_d
    Pz = z_over_d

    # TODO R and D1 are basically static
    # It should be possible to precompute most of ATA
    R = -(times - times[0])

    # TODO Flag?
    # D1 = 0.5 * jnp.square(R)
    Da_x = iint_accel_x
    Da_y = iint_accel_y
    Da_z = iint_accel_z

    pad  = jnp.zeros((times.shape[0],), dtype=times.dtype)
    ones = jnp.ones ((times.shape[0],), dtype=times.dtype)

    A = jnp.stack((jnp.concatenate((  Px,  Py,  Pz)), # d column
                   jnp.concatenate((ones, pad, pad)), # x axis initial velocity
                   jnp.concatenate(( pad,ones, pad)), # y axis initial velocity
                   jnp.concatenate(( pad, pad,ones)), # z axis initial velocity
                   jnp.concatenate((   R, pad, pad)), # x axis initial velocity
                   jnp.concatenate(( pad,   R, pad)), # y axis initial velocity
                   jnp.concatenate(( pad, pad,   R)), # z axis initial velocity
                   # jnp.concatenate((  D1, pad, pad)), # x axis gravity bias
                   # jnp.concatenate(( pad,  D1, pad)), # y axis gravity bias
                   # jnp.concatenate(( pad, pad,  D1)), # z axis gravity bias
                 ), axis=1)
    b = jnp.concatenate((Da_x, Da_y, Da_z)).reshape((A.shape[0],))

    ATA = A.T @ A

    def matvec_A(x):
      return  jnp.dot(ATA, x)

    x = jaxopt.linear_solve.solve_cholesky(matvec_A, A.T @ b)

    residuals_full = b - A @ x
    res = jnp.sum(jnp.square(residuals_full.flatten()))

    return x.flatten(), res, residuals_full

def accel_xyz_scale_constraint_tf(times, x_over_z, y_over_z, z_over_d, accel_x, accel_y, accel_z):
    ret = jaccel_xyz_scale_constraint(
        jnp.flip(times), jnp.flip(x_over_z), jnp.flip(y_over_z), jnp.flip(z_over_d),
        jnp.flip(accel_x), jnp.flip(accel_y), jnp.flip(accel_z))
    return ret
jaccel_xyz_scale_constraint_tf = jax.jit(accel_xyz_scale_constraint_tf)


def jaccel_z_scale_constraint(times, z_over_d, accel_z):
    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]

    int_accel_z  = jcumulative_trapezoid(dt, accel_z)
    iint_accel_z = jcumulative_trapezoid(dt, int_accel_z)

    Pz = z_over_d

    # TODO R and D1 are basically static
    # It should be possible to precompute most of ATA
    R = -(times - times[0])

    Da_z = iint_accel_z

    pad  = jnp.zeros((times.shape[0],), dtype=times.dtype)
    ones = jnp.ones ((times.shape[0],), dtype=times.dtype)

    A = jnp.stack((Pz, # d column
                   ones, # initial position
                   R # initial velocity
                  ), axis=1)
    b = Da_z.reshape((A.shape[0],))

    ATA = A.T @ A

    def matvec_A(x):
      return  jnp.dot(ATA, x)

    x = jaxopt.linear_solve.solve_cholesky(matvec_A, A.T @ b)

    residuals_full = b - A @ x
    res = jnp.sum(jnp.square(residuals_full.flatten()))

    return x.flatten(), res, residuals_full

def accel_z_scale_constraint_tf(times, z_over_d, accel_z):
    ret = jaccel_z_scale_constraint(
        jnp.flip(times), jnp.flip(z_over_d), jnp.flip(accel_z))
    return ret
jaccel_z_scale_constraint_tf = jax.jit(accel_z_scale_constraint_tf)

def jaccel_z_scale_constraint_match_phi(times, z_over_d, accel_z):
    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]

    int_accel_z  = jcumulative_trapezoid(dt, accel_z)
    iint_accel_z = jcumulative_trapezoid(dt, int_accel_z)

    Pz = z_over_d

    # TODO R and D1 are basically static
    # It should be possible to precompute most of ATA
    R = -(times - times[0])

    Da_z = iint_accel_z

    pad  = jnp.zeros((times.shape[0],), dtype=times.dtype)
    ones = jnp.ones ((times.shape[0],), dtype=times.dtype)

    A = jnp.stack((Da_z, # double int acceleration
                   ones, # initial position
                   R # initial velocity
                  ), axis=1)
    b = Pz.reshape((A.shape[0],))

    ATA = A.T @ A

    def matvec_A(x):
      return  jnp.dot(ATA, x)

    x = jaxopt.linear_solve.solve_cholesky(matvec_A, A.T @ b)

    residuals_full = b - A @ x
    res = jnp.sum(jnp.square(residuals_full.flatten()))

    d = 1.0 / x.flatten()[0]
    x0 = x.flatten()[1] * d
    dx0 = x.flatten()[2] * d

    return jnp.array((d, x0, dx0)), res, residuals_full

def jaccel_z_scale_constraint_match_phi_tf(times, z_over_d, accel_z):
    ret = jaccel_z_scale_constraint_match_phi(
        jnp.flip(times), jnp.flip(z_over_d), jnp.flip(accel_z))
    return ret
jaccel_z_scale_constraint_match_phi_tf = jax.jit(jaccel_z_scale_constraint_match_phi_tf)

def jaccel_xyz_phi_constraint(times, x_over_z, y_over_z, z_over_z0, accel_x, accel_y, accel_z):
    z_over_z0 = z_over_z0 / z_over_z0[0]

    dx_over_z0 = x_over_z * z_over_z0
    dx_over_z0 -= dx_over_z0[0]

    dy_over_z0 = y_over_z * z_over_z0
    dy_over_z0 -= dy_over_z0[0]

    # TODO all times are needed but they are assumed to be evenly spaced
    dt = times[1] - times[0]
    # int_accel_x  = jcumulative_int(dt, accel_x)
    # iint_accel_x = jcumulative_int(dt, int_accel_x)

    # int_accel_y  = jcumulative_int(dt, accel_y)
    # iint_accel_y = jcumulative_int(dt, int_accel_y)

    # int_accel_z  = jcumulative_int(dt, accel_z)
    # iint_accel_z = jcumulative_int(dt, int_accel_z)

    int_accel_x  = jcumulative_trapezoid(dt, accel_x)
    iint_accel_x = jcumulative_trapezoid(dt, int_accel_x)

    int_accel_y  = jcumulative_trapezoid(dt, accel_y)
    iint_accel_y = jcumulative_trapezoid(dt, int_accel_y)

    int_accel_z  = jcumulative_trapezoid(dt, accel_z)
    iint_accel_z = jcumulative_trapezoid(dt, int_accel_z)

    Px = dx_over_z0
    Py = dy_over_z0
    Pz = z_over_z0 - 1

    # TODO R and D1 are basically static
    # It should be possible to precompute most of ATA
    R = -(times - times[0])

    D1 = 0.5 * jnp.square(R)
    Da_x = iint_accel_x
    Da_y = iint_accel_y
    Da_z = iint_accel_z

    pad = jnp.zeros((times.shape[0],), dtype=times.dtype)

    A = jnp.stack((jnp.concatenate((  Px,  Py,  Pz)), # initial Z column
                   jnp.concatenate((   R, pad, pad)), # x axis initial velocity
                   jnp.concatenate(( pad,   R, pad)), # y axis initial velocity
                   jnp.concatenate(( pad, pad,   R)), # z axis initial velocity
                   jnp.concatenate((  D1, pad, pad)), # x axis gravity bias
                   jnp.concatenate(( pad,  D1, pad)), # y axis gravity bias
                   jnp.concatenate(( pad, pad,  D1)), # z axis gravity bias
                 ), axis=1)
    b = jnp.concatenate((Da_x, Da_y, Da_z)).reshape((A.shape[0],))

    ATA = A.T @ A

    def matvec_A(x):
      return  jnp.dot(ATA, x)

    x = jaxopt.linear_solve.solve_cholesky(matvec_A, A.T @ b)

    residuals_full = b - A @ x
    res = jnp.sum(jnp.square(residuals_full.flatten()))

    return x.reshape((7,)), res, residuals_full

def accel_xyz_phi_constraint_tf(times, x_over_z, y_over_z, z_over_z0, accel_x, accel_y, accel_z):
    ret = jaccel_xyz_phi_constraint(
        jnp.flip(times), jnp.flip(x_over_z), jnp.flip(y_over_z), jnp.flip(z_over_z0),
        jnp.flip(accel_x), jnp.flip(accel_y), jnp.flip(accel_z))
    return ret
jaccel_xyz_phi_constraint_tf = jax.jit(accel_xyz_phi_constraint_tf)

def jbound_R_hull_to_Z(x_i, x_j, y_i, y_j, R_cm, p_m, smooth_approx=False):
  # inward facing normals that define the viewing frustum
  i_x_normal = jnp.array(( 1.0,  0.0,  x_i))
  j_x_normal = jnp.array((-1.0,  0.0, -x_j))
  i_y_normal = jnp.array(( 0.0, -1.0,  y_i))
  j_y_normal = jnp.array(( 0.0,  1.0, -y_j)) 

  p_c = (R_cm @ p_m.T)

  if not smooth_approx:
    i_x = jnp.argmin(i_x_normal @ p_c)
    j_x = jnp.argmin(j_x_normal @ p_c)
    i_y = jnp.argmin(i_y_normal @ p_c)
    j_y = jnp.argmin(j_y_normal @ p_c)

    p_m_i_x = p_m[i_x]
    p_m_j_x = p_m[j_x]
    p_m_i_y = p_m[i_y]
    p_m_j_y = p_m[j_y]
  else:
    d_i_x = i_x_normal @ p_c
    d_j_x = j_x_normal @ p_c
    d_i_y = i_y_normal @ p_c
    d_j_y = j_y_normal @ p_c

    # min_d_i_x = jnp.min(d_i_x)
    # min_d_j_x = jnp.min(d_j_x)
    # min_d_i_y = jnp.min(d_i_y)
    # min_d_j_y = jnp.min(d_j_y)

    # tau = 1e-3
    # alpha_i_x = jnp.exp(-(d_i_x - min_d_i_x) / tau)
    # alpha_j_x = jnp.exp(-(d_j_x - min_d_j_x) / tau)
    # alpha_i_y = jnp.exp(-(d_i_y - min_d_i_y) / tau)
    # alpha_j_y = jnp.exp(-(d_j_y - min_d_j_y) / tau)

    # alpha_i_x /= jnp.linalg.norm(alpha_i_x)
    # alpha_j_x /= jnp.linalg.norm(alpha_j_x)
    # alpha_i_y /= jnp.linalg.norm(alpha_i_y)
    # alpha_j_y /= jnp.linalg.norm(alpha_j_y)

    tau = 1e-6
    alpha_i_x = jax.nn.softmax(-d_i_x / tau)
    alpha_j_x = jax.nn.softmax(-d_j_x / tau)
    alpha_i_y = jax.nn.softmax(-d_i_y / tau)
    alpha_j_y = jax.nn.softmax(-d_j_y / tau)

    p_m_i_x = alpha_i_x @ p_m
    p_m_j_x = alpha_j_x @ p_m
    p_m_i_y = alpha_i_y @ p_m
    p_m_j_y = alpha_j_y @ p_m

  v_ij_m_x = p_m_j_x - p_m_i_x
  v_ij_m_y = p_m_j_y - p_m_i_y

  v_ij_c_x = R_cm @ v_ij_m_x
  v_ij_c_y = R_cm @ v_ij_m_y

  # TODO there is a coordinate frame issue
  z_i_x = (-x_j * v_ij_c_x[2] - v_ij_c_x[0]) / -(x_i - x_j)
  z_i_y = ( y_j * v_ij_c_y[2] - v_ij_c_y[1]) /  (y_i - y_j)

  p_m_c = jnp.zeros((3,))
  v_ic_m_x = p_m_c - p_m_i_x
  v_ic_m_y = p_m_c - p_m_i_y

  v_ic_c_x = R_cm @ v_ic_m_x
  v_ic_c_y = R_cm @ v_ic_m_y

  z_c_x = z_i_x + v_ic_c_x[2]
  z_c_y = z_i_y + v_ic_c_y[2]

  # TODO gauruntee z_c_x and z_c_y positive
  return z_c_x, z_c_y, z_i_x, z_i_y

def jbound_R_hull_to_Z_iterate(R_hd, x_c, hull_points):
  z_c_x, z_c_y, z_i_x, z_i_y = jbound_R_hull_to_Z(
    x_c[1], x_c[2], x_c[3], x_c[4], R_hd, hull_points)

  x_over_z = -(x_c[1] + x_c[2]) / 2
  y_over_z =  (x_c[3] + x_c[4]) / 2
  # z_over_s = jnp.sqrt(jnp.abs(z_c_x * z_c_y))
  z_over_s = jnp.sqrt(z_c_x * z_c_y)

  return x_over_z, y_over_z, z_over_s
vjbound_R_hull_to_Z_iterate = jax.vmap(jbound_R_hull_to_Z_iterate, in_axes=[0, 0, None])

# https://en.wikipedia.org/wiki/Axis%E2%80%93angle_representation#Exponential_map_from_so(3)_to_SO(3)
def jso3_to_SO3(w_mag):
  theta = jnp.linalg.norm(w_mag) + 1e-11 # TODO is this necessary? What is a better way?
  w = w_mag / theta
  K = jnp.array([[0, -w[2], w[1]],
                 [w[2], 0, -w[0]],
                 [-w[1], w[0], 0]])
  R = np.eye(3) + jnp.sin(theta) * K + (1 - jnp.cos(theta)) * (K @ K)
  return R
vjso3_to_SO3 = jax.vmap(jso3_to_SO3, in_axes=[0,])

def jcumulative_matmul_back_right(i, val):
  A_out, A = val
  A_out = A_out.at[-i-2, :, :].set(A_out[-i-1, :, :] @ A[-i-1, :, :])
  return A_out, A

def jphi_constraint_body_u_hull_impulse(Rhd0, wb, h, x_c, u_d, Rhd00, h0, hull_points, l_wb, l_h0):
  hat_T  = jnp.convolve(u_d[:, 1], h[0, :], mode='valid')
  hat_wx = jnp.convolve(u_d[:, 2], h[1, :], mode='valid') + wb[0]
  hat_wy = jnp.convolve(u_d[:, 3], h[2, :], mode='valid') + wb[1]
  hat_wz = jnp.convolve(u_d[:, 4], h[3, :], mode='valid') + wb[2]

  zeros = jnp.zeros((hat_T.shape[0], 2))
  ia_d  = jnp.column_stack((u_d[-hat_T.shape[0]:, 0], zeros, hat_T))
  w_d   = jnp.column_stack((u_d[-hat_T.shape[0]:, 0], hat_wx, hat_wy, hat_wz))

  Rd1d2 = vjso3_to_SO3(((w_d[:-1, 0] - w_d[1:, 0]) * w_d[1:, 1:4].T).T)

  R_hd = jnp.zeros((Rd1d2.shape[0]+1, 3, 3))
  R_hd = R_hd.at[-1, :, :].set(Rhd0)
  R_hd, _ = jax.lax.fori_loop(0, Rd1d2.shape[0], jcumulative_matmul_back_right, (R_hd, Rd1d2))

  ia_h = (R_hd @ jnp.atleast_3d(ia_d[:, 1:4])).squeeze()

  # Calculate Phi from x_c (bounding box), hull_points, R_hd
  x_c_trimmed = x_c[-R_hd.shape[0]:]
  x_over_z, y_over_z, z_over_s = vjbound_R_hull_to_Z_iterate(R_hd, x_c_trimmed, hull_points)

  ret = accel_xyz_phi_constraint_tf(
      times     = x_c_trimmed[:, 0],
      x_over_z  = x_over_z,
      y_over_z  = y_over_z,
      z_over_z0 = z_over_s, # This function converts to z_over_z0 internally
      accel_x   = ia_h [:, 0],
      accel_y   = ia_h [:, 1],
      accel_z   = ia_h [:, 2])

  (Z_tf, dotX_tf, dotY_tf, dotZ_tf, a_b_x, a_b_y, a_b_z), res, full_res = ret

  # print('accel_xyz_phi_constraint_tf', ret)

  hat_pd0_h = jnp.array((x_over_z[-1]*Z_tf,
                         y_over_z[-1]*Z_tf,
                                      Z_tf))
  hat_vd0_h = jnp.array((dotX_tf, dotY_tf, dotZ_tf))
  hat_ad0_h = ia_h[-1, 0:3] - jnp.array((a_b_x, a_b_y, a_b_z))

  l_h0 = 10.0
  l_h = 1000.0
  l_Rhd = 1.0
  residuals = full_res.flatten()
  residuals = jnp.concatenate((residuals, jnp.sqrt(l_wb) * wb))
  residuals = jnp.concatenate((residuals, jnp.sqrt(l_h0) * (h - h0).flatten()))
  residuals = jnp.concatenate((residuals, jnp.array([jnp.sqrt(l_h) * (jnp.sum(h[i, :]) - 1) for i in range(4)])))
  residuals = jnp.concatenate((residuals, jnp.sqrt(l_Rhd) * (Rhd0 - Rhd00).flatten()))

  # TODO why is jax sometimes returning the negative?
  return -jnp.sign(Z_tf) * hat_pd0_h, -jnp.sign(Z_tf) * hat_vd0_h, -jnp.sign(Z_tf) * hat_ad0_h, residuals


def jphi_constraint_body_u_hull_impulse_theta(theta, x_c, u_d, Rhd0, wb0, h0, l_wb, l_h0, hull_points, opt_R, opt_wb, opt_h):
  # Rhd = jR.from_rotvec(theta[:3]).as_matrix()
  # wb  = theta[3:6] if opt_wb else wb0
  # if   opt_h and     opt_wb: h = theta[6:].reshape(h0.shape)
  # elif opt_h and not opt_wb: h = theta[3:].reshape(h0.shape)
  # else:                      h = h0

  i = 0
  if opt_R:
    Rhd = jR.from_rotvec(theta[i:i+3]).as_matrix()
    i+=3
  else:
    Rhd = Rhd0

  if opt_wb:
    wb = theta[i:i+3]
    i+=3
  else:
    wb = wb0

  if opt_h:
    h = theta[i:i+h0.size].reshape(h0.shape)
    i+=h0.size
  else:
    h = h0

  _, _, _, loss = jphi_constraint_body_u_hull_impulse(Rhd, wb, h, x_c, u_d, Rhd0, h0, hull_points, l_wb, l_h0)

  return loss

def jphi_constraint_body_u_hull_optimize_impulse(x_c, u_d, Rhd0=jnp.eye(3), wb0=jnp.zeros((3,)), h0=jnp.ones((4,1)),
                                                 l_wb=10.0, l_h0=100.0, hull_points=None,
                                                 opt_R=True, opt_wb=True, opt_h=True):
  Rhd0_rotvec = jR.from_matrix(Rhd0).as_rotvec() # TODO, let rotvec always start from 0

  theta0 = None
  if opt_R:  theta0 = Rhd0_rotvec  if theta0 is None else jnp.concatenate((theta0, Rhd0_rotvec))
  if opt_wb: theta0 = wb0          if theta0 is None else jnp.concatenate((theta0, wb0))
  if opt_h:  theta0 = h0.flatten() if theta0 is None else jnp.concatenate((theta0, h0.flatten()))

  if theta0 is not None:
    gn = GaussNewton(residual_fun=
      lambda theta: jphi_constraint_body_u_hull_impulse_theta(theta,
         x_c, u_d, Rhd0, wb0, h0, l_wb, l_h0, hull_points, opt_R, opt_wb, opt_h))
    res = gn.run(theta0)

    theta_star = res.params
  else:
    res = None
    theta_star = theta0

  i = 0
  if opt_R:
    Rhd0star = jR.from_rotvec(theta_star[i:i+3]).as_matrix()
    i+=3
  else:
    Rhd0star = Rhd0

  if opt_wb:
    wb_star = theta_star[i:i+3]
    i+=3
  else:
    wb_star = wb0

  if opt_h:
    h_star = theta_star[i:i+h0.size].reshape(h0.shape)
    i+=h0.size
  else:
    h_star = h0

  return res, Rhd0star, wb_star, h_star, jphi_constraint_body_u_hull_impulse(Rhd0star, wb_star, h_star, x_c, u_d, Rhd0, h0, hull_points, l_wb, l_h0)

jphi_constraint_body_u_hull_optimize_impulse = jax.jit(jphi_constraint_body_u_hull_optimize_impulse, static_argnames=('opt_R', 'opt_wb', 'opt_h'))
