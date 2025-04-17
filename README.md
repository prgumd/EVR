# Embodied Visuomotor Representation

[Webpage](https://prg.cs.umd.edu/EVR), [arXiv](https://arxiv.org/abs/2410.00287)


## Uncalibrated Jumping Simulation
To run: `python3 jumping/gerbil.py`. An interactive simulation window will open that allows you to drag the jump target, and change the strength of gravity. Make sure to pause the simulation before the measurement oscillations start, adjust these parameters, then let the simulation run.

Requires: matplotlib, numpy, scipy, mujoco, cvxopt

The results should look like this video.

[![Uncalibrated Jumping Video](https://img.youtube.com/vi/6yPAL_Xx36Y/0.jpg)](https://www.youtube.com/watch?v=6yPAL_Xx36Y)

To run the experiments that produced the plots in the paper, run python3 gerbil_experiments.py.

## Uncalibrated Touching and Clearing Experiment

These codes under `uncalibrated_clearing` are designed to run on a custom robot built by combining a DJI robomaster with a Raspberry Pi 5 and a custom pan tilt servo camera mount. The are provided as a reference to any interested reader. These codes depend on the included `vme_research` package which can be installed by running `pip install -e ./vme_research` from the root of the directly.

The acheived behaviour looks like this:

[![Uncalibrated Touching and Clearing Video](https://img.youtube.com/vi/z20scBUrwrs/0.jpg)](https://www.youtube.com/watch?v=z20scBUrwrs)
