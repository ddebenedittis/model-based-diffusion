# Model-Based Diffusion for Trajectory Optimization

<div align="center">

[[Website]](https://lecar-lab.github.io/mbd/)
[[PDF]](https://drive.google.com/file/d/1kPjD79Cfr9spWulWNVFMRHqTE-mjbGAp/view?usp=sharing)
[[Arxiv]](https://arxiv.org/pdf/2407.01573)

[<img src="https://img.shields.io/badge/Backend-Jax-red.svg"/>](https://github.com/google/jax)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<!-- insert figure -->
<img src="assets/joint.gif" width="600px"/>

</div>

This repository contains the code for the paper "Model-based Diffusion for Trajectory Optimization".

Model-based diffusion (MBD) is a novel **diffusion-based trajectory optimization** framework that employs a **dynamics model** to approximate the score function. 
MBD outperforms existing methods (including RL) in terms of sample efficiency and generalization.

## Installation

To install the required packages without `cuda` support, run the following command:

```bash
git clone --depth 1 git@github.com:LeCAR-Lab/model-based-diffusion.git
pip install -e .
```

To install `mbd` with `cuda` support, run the following command:

```bash
pip install -e ".[cuda12]"
```

## Usage

### Model-based Diffusion for Trajectory Optimization

To run model-based diffusion to optimize a trajectory, run the following command:

```bash
python mbd/planners/mbd_planner.py --env_name $ENV_NAME
```

where `$ENV_NAME` is the name of the environment, you can choose from `hopper`, `halfcheetah`, `walker2d`, `ant`, `humanoidrun`, `humanoidstandup`, `humanoidtrack`, `car2d`, `pushT`.

To run model-based diffusion combined with demonstrations, run the following command:

```bash
python mbd/planners/mbd_planner.py --env_name $ENV_NAME --enable_demo
```

Currently, only the `humanoidtrack`, `car2d` support demonstrations.

To run multiple seeds, run the following command:

```bash
python mbd/scripts/run_mbd.py --env_name $ENV_NAME
```

To visualize the diffusion process, run the following command:

```bash
python mbd/scripts/vis_diffusion.py --env_name $ENV_NAME
```

Please make sure you have run the planner first to generate the data.

### Model-based Diffusion for Black-box Optimization

To run model-based diffusion for black-box optimization, run the following command:

```bash
python mbd/blackbox/mbd_opt.py
```

### Other Baselines

To run RL-based baselines, run the following command:

```bash
python mbd/rl/train_brax.py --env_name $ENV_NAME
```

To run other zeroth order trajectory optimization baselines, run the following command:

```bash
python mbd/planners/path_integral.py --env_name $ENV_NAME --update_method $MODE
```

where `$MODE` is the mode of the planner, you can choose from `mppi`, `cem`, `cma-es`.

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