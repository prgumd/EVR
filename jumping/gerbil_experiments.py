import os
if __name__ == '__main__': os.environ['OPENBLAS_NUM_THREADS'] = '1'
import multiprocessing
import pickle
import time

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from tqdm import tqdm

# Uncomment to produce LaTeX style plots exactly as in the paper
# import scienceplots
# plt.style.use('science')

from gerbil import build_simulation

def build_res(res, shift=None): return build_simulation(res=res, shift=shift)
def build_dist(dist, shift=None): return build_simulation(dist=dist, shift=shift)
def build_grav(grav, shift=None): return build_simulation(grav=grav, shift=shift)
def build_tau(tau, shift=None): return build_simulation(tau=tau, shift=shift)
def build_b_scale(b_scale, shift=None): return build_simulation(b_scale=b_scale, shift=shift)
def build_oscillation_T(oscillation_T, shift=None): return build_simulation(oscillation_T=oscillation_T, shift=shift)

def run_jumper(m, d, jumper, run_viewer=False, sleep=False):
  if run_viewer:
    with mujoco.viewer.launch_passive(m, d) as viewer:
      while not jumper.finished:
        mujoco.mj_step(m, d)
        viewer.sync()
        if sleep: time.sleep(0.001)
  else:
    while not jumper.finished:
      mujoco.mj_step(m, d)

def collect_data(xvalues, pixel_shifts, file, builder, run_viewer=False, sleep=False):
  cv2.setNumThreads(1)

  # Collect data from each xvalue
  plot_datas = []
  jump_datas = []
  for x in tqdm(xvalues, desc=f'Running {file}'):
    m, d, jumper = builder(x)
    run_jumper(m, d, jumper, run_viewer=run_viewer, sleep=sleep)
    if jumper.plot_data:
      plot_datas.append(jumper.plot_data)
      jump_datas.append(jumper.jump_info)
    del jumper # Force the renderer to close
  with open(f'data/plot_datas_{file}.pickle', 'wb') as f:
    pickle.dump(plot_datas, f)
  with open(f'data/plot_datas_{file}_jump.pickle', 'wb') as f:
    pickle.dump(jump_datas, f)

  # Collect data at a bunch of pixel shifts for each resolution
  pixel_shift_plot_datas = []
  jump_datas = []
  for x in tqdm(xvalues, desc=f'Running {file} pixel shifts'):
    for shift in pixel_shifts:
      m, d, jumper = builder(x, shift=shift)
      run_jumper(m, d, jumper)
      if jumper.plot_data:
        pixel_shift_plot_datas.append(jumper.plot_data)
        jump_datas.append(jumper.jump_info)
      del jumper
  with open(f'data/plot_datas_{file}_pixel_shift.pickle', 'wb') as f:
    pickle.dump(pixel_shift_plot_datas, f)

def plot_bounds(fig, ax, xvalues, pixel_shifts, hat, gt, errorbar_data, key):
  d_means = []
  d_mins = []
  d_maxs = []
  for i, res in enumerate(xvalues):
    res_d = []
    for j, shift in enumerate(pixel_shifts):
      plot_index = i * len(pixel_shifts) + j
      plot_data = errorbar_data[plot_index]
      d = key(plot_data)
      res_d.append(d)
    d_means.append(np.mean(res_d))
    d_mins.append(np.min(res_d))
    d_maxs.append(np.max(res_d))
  d_means = np.array(d_means)
  d_mins = np.array(d_mins)
  d_maxs = np.array(d_maxs)

  ax.fill_between(xvalues, d_mins, d_maxs, alpha=0.2)
  ax.plot(xvalues, gt, marker='x', label='gt')
  ax.plot(xvalues, hat, label='est')

def load_and_plot(fig, axes, xvalues, plot_datas, pixel_shift_plot_datas, pixel_shifts, xlog=False):
  hat_gb    = [p['coef_motor'][2] for p in plot_datas]
  gt_gb     = [p['bounds_info']['gt_gb_over_d'] / p['bounds_info']['gt_d_inv'] for p in plot_datas]
  hat_d     = [p['coef_motor'][0]       for p in plot_datas]
  gt_d      = [p['bounds_info']['gt_d'] for p in plot_datas]
  plot_bounds(fig, axes[0], xvalues, pixel_shifts, hat_d, gt_d,
              pixel_shift_plot_datas, key=lambda p: p['coef_motor'][0])
  plot_bounds(fig, axes[1], xvalues, pixel_shifts, hat_gb, gt_gb,
              pixel_shift_plot_datas, key=lambda p: p['coef_motor'][2])

  if xlog:
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
  axes[0].grid(True)
  axes[1].grid(True)
  return fig, axes

def make_plot(fig, axes, xvalues, pixel_shifts, xlog, file, title):
  with open(f'data/plot_datas_{file}.pickle', 'rb') as f:
    plot_datas = pickle.load(f)
  with open(f'data/plot_datas_{file}_pixel_shift.pickle', 'rb') as f:
    pixel_shift_plot_datas = pickle.load(f)
  fig, axes = load_and_plot(fig, axes, xvalues, plot_datas, pixel_shift_plot_datas, pixel_shifts, xlog)
  axes[0].set_title(title)

def load_and_plot_jump(fig, axes, xvalues, plot_datas, file, legends):
  for i, p in enumerate(plot_datas):
    jump_t = p['t']
    jump_x = p['x']
    jump_u = p['u']
    axes[0].plot(jump_x[:, 1], jump_x[:, 2] - jump_x[0, 2], label=f'{file}={xvalues[i]:0.2f}')
    axes[0].set_ylim([-0.2, 3.5])

    t_jump_end = p['t_jump_end']
    i_jump_end = np.searchsorted(jump_t, t_jump_end) + 1
    axes[1].plot(jump_t[:i_jump_end] - jump_t[0], -jump_u[:i_jump_end], label=legends[i])
  axes[1].legend(loc='upper left')
  axes[0].grid(True)
  axes[1].grid(True)
  return fig, axes

def make_jump_plot(fig, axes, xvalues, pixel_shifts, file, title, legends):
  with open(f'data/plot_datas_{file}_jump.pickle', 'rb') as f:
    plot_datas = pickle.load(f)
  fig, axes = load_and_plot_jump(fig, axes, xvalues, plot_datas, file, legends)
  axes[0].set_title(title)

def run_experiments():
  multiprocessing.set_start_method('spawn')
  os.makedirs('data', exist_ok=True)

  # Full experiment
  N_steps = 4
  resolutions = [100, 200, 300, 600]
  initial_distances = np.linspace(1.0, 4.0, N_steps)
  gravities = 9.81 * np.linspace(0.5, 2.0, N_steps)
  taus = np.linspace(0.05, 0.2, N_steps)
  b_scales = np.linspace(0.5, 2.0, N_steps)
  oscillation_Ts = [6.0, 8.0, 10.0, 15.0]
  oscillation_hzs = np.linspace(0.5, 2.0, N_steps)
  amplitudes = np.linspace(0.3, 1.0, N_steps)
  pixel_shifts = np.linspace(-0.5, 0.5, 25+1)

  processes = []
  processes.append(multiprocessing.Process(target=collect_data, args=(resolutions, pixel_shifts, 'res', build_res)))
  processes.append(multiprocessing.Process(target=collect_data, args=(initial_distances, pixel_shifts, 'dist', build_dist)))
  processes.append(multiprocessing.Process(target=collect_data, args=(gravities, pixel_shifts, 'grav', build_grav)))
  processes.append(multiprocessing.Process(target=collect_data, args=(taus, pixel_shifts, 'tau', build_tau)))
  processes.append(multiprocessing.Process(target=collect_data, args=(b_scales, pixel_shifts, 'b', build_b_scale)))
  processes.append(multiprocessing.Process(target=collect_data, args=(oscillation_Ts, pixel_shifts, 'oscillation_T', build_oscillation_T)))
  [p.start() for p in processes]
  [p.join() for p in processes]

  rows = 2
  cols = 6
  fig, axes = plt.subplots(rows, cols, figsize=(cols*2, 4))#, sharey='row')
  axes_cols = [[axes[i][j] for i in range(rows)] for j in range(cols)]

  # distance, distance and b sharey
  axes[0][5].sharey(axes[0][0])

  # distance, tau, res, oscT, gravity sharey
  axes[0][2].sharey(axes[0][1])
  axes[0][3].sharey(axes[0][1])
  axes[0][4].sharey(axes[0][1])

  # grav, distance, tau, res, osc_T share Y 
  axes[1][1].sharey(axes[1][0])
  axes[1][2].sharey(axes[1][0])
  axes[1][3].sharey(axes[1][0])
  # grav, gravity, b sharey
  axes[1][5].sharey(axes[1][4])

  distance_variation = 'Jump Distance $d$'
  tau_variation = "Time Constant $1/\\alpha$"
  res_variation = "Camera Resolution"
  oscillation_period = "Oscillation Period $T$"
  gravity_variation = "Gravity $g_b$"
  actuator_gain = "Actuator Gain $b$"

  b_gt = 2 / 10 # 2 legs, 10kg
  b_plot = b_scales * b_gt

  # Estimation plot legends
  make_plot(fig, axes_cols[0], initial_distances, pixel_shifts, False, 'dist', distance_variation)
  make_plot(fig, axes_cols[1], taus, pixel_shifts, False, 'tau', tau_variation)
  make_plot(fig, axes_cols[2], resolutions, pixel_shifts, False, 'res', res_variation)
  make_plot(fig, axes_cols[3], oscillation_Ts, pixel_shifts, False, 'oscillation_T', oscillation_period)
  make_plot(fig, axes_cols[4], gravities, pixel_shifts, False, 'grav', gravity_variation)
  make_plot(fig, axes_cols[5], b_plot, pixel_shifts, False, 'b', actuator_gain)

  hat_d = "$\\hat{d}$"
  hat_gb= "$\\hat{g}_b$"
  def set_legend(i, j, var, loc='best'):
    handles, labels = axes[i][j].get_legend_handles_labels()
    axes[i][j].legend(handles, ["gt", var], loc=loc)
  set_legend(0, 0, hat_d, loc='upper left')
  set_legend(1, 0, hat_gb, loc='upper left')

  axes[1][0].set_xlabel('Jump Distance $d$ (m)')
  axes[1][1].set_xlabel('Time Constant $1/\\alpha$ (s)')
  axes[1][2].set_xlabel('Camera Resolution (pixels)')
  axes[1][3].set_xlabel('Oscillation Period $T$ (s)')
  axes[1][4].set_xlabel('Gravity $g_b$ (m/s$^2$)')
  axes[1][5].set_xlabel('Actuator Gain $b$')
  axes[0][0].set_ylabel('$\\hat{d}$ (embodied)')
  axes[1][0].set_ylabel('$\\hat{g}_b$ (embodied)')
  fig.align_ylabels((axes[0][0], axes[1][0]))
  # fig.suptitle('Estimated Jump Distance and Gravitational Strength versus Experimental Parameters')
  fig.tight_layout()
  fig.savefig('jumping_estimation_results.png', dpi=300)


  d_legends    = [f"$d={x:0.2f}$" for x in initial_distances]
  tau_legends  = [f"$1/\\alpha={x:0.2f}$" for x in taus]
  res_legends  = [ "$\\mathrm{res}=" + f"{x}$" for x in resolutions]
  osc_legends  = [f"$T={x:0.2f}$" for x in oscillation_Ts]
  grav_legends = [f"$g={x:0.2f}$" for x in gravities]
  b_legends    = [f"$b={x:0.2f}$" for x in b_plot]
  rows = 2
  cols = 6
  fig, axes = plt.subplots(rows, cols, figsize=(cols*2, 4), sharey='row')
  axes_cols = [[axes[i][j] for i in range(rows)] for j in range(cols)]
  make_jump_plot(fig, axes_cols[0], initial_distances, pixel_shifts, 'dist', distance_variation, d_legends)
  make_jump_plot(fig, axes_cols[1], taus, pixel_shifts, 'tau', tau_variation, tau_legends)
  make_jump_plot(fig, axes_cols[2], resolutions, pixel_shifts, 'res', res_variation, res_legends)
  make_jump_plot(fig, axes_cols[3], oscillation_Ts, pixel_shifts, 'oscillation_T', oscillation_period, osc_legends)
  make_jump_plot(fig, axes_cols[4], gravities, pixel_shifts, 'grav', gravity_variation, grav_legends)
  make_jump_plot(fig, axes_cols[5], b_plot, pixel_shifts, 'b', actuator_gain, b_legends)

  y_meters = "$y$ (m)"
  z_meters = "$z$ (m)"
  t_sec = "$t$ (sec)"
  axes[0][0].set_xlabel(z_meters)
  axes[0][1].set_xlabel(z_meters)
  axes[0][2].set_xlabel(z_meters)
  axes[0][3].set_xlabel(z_meters)
  axes[0][4].set_xlabel(z_meters)
  axes[0][5].set_xlabel(z_meters)
  axes[1][0].set_xlabel(t_sec)
  axes[1][1].set_xlabel(t_sec)
  axes[1][2].set_xlabel(t_sec)
  axes[1][3].set_xlabel(t_sec)
  axes[1][4].set_xlabel(t_sec)
  axes[1][5].set_xlabel(t_sec)

  axes[0][0].set_ylabel(y_meters)
  axes[1][0].set_ylabel("$u$ (embodied)")
  fig.align_ylabels((axes[0][0], axes[1][0]))
  # fig.suptitle('Jump Trajectories and Jump Control Signal versus Experimental Parameters')
  fig.tight_layout()
  fig.savefig('jumping_jump_results.png', dpi=300)

  plt.show()

if __name__ == '__main__':
  np.set_printoptions(suppress=False, linewidth=200)
  run_experiments()
