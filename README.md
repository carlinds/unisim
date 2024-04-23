# Unisim

This is an unofficial re-implementation of [UniSim: A Neural Closed-Loop Sensor Simulator](https://openaccess.thecvf.com/content/CVPR2023/papers/Yang_UniSim_A_Neural_Closed-Loop_Sensor_Simulator_CVPR_2023_paper.pdf).

This is a plugin to [neurad-studio](https://github.com/geohess/neurad-studio).


## Installation

```bash
uv pip install git+https://github.com/georghess/neurad-studio.git
uv pip install -e .
```

## Usage

```bash
ns-train unisim pandaset-data --data data/pandaset
```