# Model-Based Diffusion for Trajectory Optimization

> Fork of [LeCAR-Lab/model-based-diffusion](https://github.com/LeCAR-Lab/model-based-diffusion). Used as a submodule by the [mrmbd](../README.md) project.

<div align="center">

[[Website]](https://lecar-lab.github.io/mbd/)
[[PDF]](https://drive.google.com/file/d/1kPjD79Cfr9spWulWNVFMRHqTE-mjbGAp/view?usp=sharing)
[[Arxiv]](https://arxiv.org/pdf/2407.01573)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- insert figure -->
<img src="assets/joint.gif" width="600px"/>

</div>

Model-based diffusion (MBD) is a **diffusion-based trajectory optimization** framework that employs a **dynamics model** to approximate the score function.
MBD outperforms existing methods (including RL) in terms of sample efficiency and generalization.

## Installation

To install the required packages without `cuda` support, run the following command:

```bash
git clone --depth 1 https://github.com/ddebenedittis/model-based-diffusion.git
pip install -e .
```

To install `mbd` with `cuda` support, run the following command:

```bash
pip install -e ".[cuda12]"
```

## Project Structure

```
mbd/
├── envs/          # Brax-based environments (hopper, ant, humanoid, car2d, pushT, ...)
├── planners/      # Trajectory optimization (MBD planner, path integral baselines)
├── blackbox/      # Black-box optimization variants
├── rl/            # RL baselines (Brax training)
├── scripts/       # Multi-seed runs & diffusion visualization
└── utils.py       # Shared utilities
```

## Usage

### Trajectory Optimization

```bash
python mbd/planners/mbd_planner.py --env_name $ENV_NAME
```

Available environments: `hopper`, `halfcheetah`, `walker2d`, `ant`, `humanoidrun`, `humanoidstandup`, `humanoidtrack`, `car2d`, `pushT`.

With demonstrations (supported for `humanoidtrack`, `car2d`):

```bash
python mbd/planners/mbd_planner.py --env_name $ENV_NAME --enable_demo
```

Multi-seed runs:

```bash
python mbd/scripts/run_mbd.py --env_name $ENV_NAME
```

Visualize the diffusion process (requires a completed planner run):

```bash
python mbd/scripts/vis_diffusion.py --env_name $ENV_NAME
```

### Black-box Optimization

```bash
python mbd/blackbox/mbd_opt.py
```

### Baselines

RL baseline:

```bash
python mbd/rl/train_brax.py --env_name $ENV_NAME
```

Zeroth-order trajectory optimization (MPPI, CEM, CMA-ES):

```bash
python mbd/planners/path_integral.py --env_name $ENV_NAME --update_method $MODE
```

## Acknowledgements

* This codebase's environment and RL implementation is built on top of [Brax](https://github.com/google/brax).

## BibTeX

```bibtex
@misc{pan2024modelbaseddiffusiontrajectoryoptimization,
      title={Model-Based Diffusion for Trajectory Optimization}, 
      author={Chaoyi Pan and Zeji Yi and Guanya Shi and Guannan Qu},
      year={2024},
      eprint={2407.01573},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.01573}, 
}
```