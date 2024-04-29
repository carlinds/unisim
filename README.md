# Unisim

This is an unofficial re-implementation of [UniSim: A Neural Closed-Loop Sensor Simulator](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf).



## Installation
This is a plugin to [neurad-studio](https://github.com/georghess/neurad-studio). Please refer to the [neurad-studio documentation](https://github.com/georghess/neurad-studio?tab=readme-ov-file#1-installation-setup-the-environment) for information about prerequisites and dependencies before the installation of this plugin.


If you wish to develop on this plugin, simply clone the repository and install neurad-studio:

```bash
pip install git+https://github.com/georghess/neurad-studio.git
pip install -e .
```

If you just want to run UniSim in your existing neurad-studio environment, you can run it directly as any other method using the `ns-train` command:
```bash
ns-train unisim pandaset-data --data data/pandaset
```
and follow the instructions in the terminal.

## Usage

```bash
ns-train unisim pandaset-data --data data/pandaset
```

## Models

We provide a `unisim` model, which is our attempt at a faithful reimplementation. Note that the GAN loss is disabled by default, as there was a large degree of uncertainty in its implementation. We especially welcome any contributions in this area.

We also provide a `unisim++` model, which includes a number of tweaks/changes to the original model. These include:
- Enabling various improvements from NeuRAD, such as rolling shutter compensation and training with missing lidar points.
- Using a mipnerf-style gaussian approximation to compensate for the spatial extent of frustums.
- Replacing the first stage of unisim training with a learning rate warmup.
- Tuned losses and hyperparameters.
