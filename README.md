# Selective Piecewise Constant Strain (SPCS) Kinematics

This repository provides a JAX implementation of the Selective Piecewise Constant Strain (SPCS) kinematics for continuum 
soft robots. SPCS combines Constant Strain (CS) and Piecewise Constant Strain (PCS) [[1]](#1) kinematics to allow for some strains
to stay constant throughout the entire robot, while other strains vary piecewise.
This allows us to minimize the degrees of freedom of the kinematic state while still being able to accurately model the
shape of the robot.

As we show in our paper (see below), SPCS is for example a good choice for robots based on Handed Shearing Auxetics (HSAs).
The twist and axial strains of the HSAs can be kept constant throughout the robot, while the bending and shear strains 
vary piecewise.

We provide an efficient implementation in JAX for evaluating the 3D forward kinematics of many points along the 
centerline of the robot.
We make use of the JAX autograd capabilities to implement a differential inverse kinematics algorithm that can be
for example used for shape sensing purposes.

## Citation
This kinematic model is part of the publication **Modelling Handed Shearing Auxetics:
Selective Piecewise Constant Strain Kinematics and Dynamic Simulation** presented at the 
_6th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2023)_. 
You can find the publication online on ~~IEEE Xplore~~.

Please use the following citation if you use our method in your (scientific) work:

```bibtex
@inproceedings{stolzle2023modelling,
  title={Modelling Handed Shearing Auxetics: Selective Piecewise Constant Strain Kinematics and Dynamic Simulation},
  author={St{\"o}lzle, Maximilian and Chin, Lillian and Truby, Ryan L. and Rus, Daniela and Della Santina, Cosimo},
  booktitle={2023 IEEE 6th International Conference on Soft Robotics (RoboSoft)},
  year={2023},
  organization={IEEE}
}
```

## Installation
The plugin can be installed from PyPI:

```bash
pip install jax-spcs-kinematics
```

or locally from the source code:

```bash
pip install .
```

## Getting started

We invite you to run the example script to demonstrate both the forward and inverse kinematics capabilities of the SPCS model:

```bash
python examples/spcs_example.py
```

## Usage

Below, you can find a minimal example of how to use the SPCS kinematic model.
We start with the initialization of the model:

```python
from jax import numpy as jnp
from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain

kinematics = SelectivePiecewiseConstantStrain(
    # specify the original length of each segment
    # in this case, the rod is composed of 2 segments of length 0.5 m each
    l0=jnp.array([0.5, 0.5]),
    # model twist and elongation to be constant across entire rod
    strain_selector_cs=jnp.array([False, False, True, False, False, True]),
    # model the bending and shear strains to be constant across each segment (i.e. piecewise constant)
    strain_selector_pcs=jnp.array([True, True, False, True, True, False])
)
```

The strain basis of the CS and PCS can then be verified:

```python
print(kinematics.strain_basis_cs)
print(kinematics.strain_basis_pcs)
```

which should result in:

\begin{equation}\small
\begin{split}
    B_\mathrm{CS} = \begin{bmatrix}
        0 & 0 & 1 & 0 & 0 & 0\\
        0 & 0 & 0 & 0 & 0 & 1\\
    \end{bmatrix}^\mathrm{T} \in \mathbb{R}^{6 \times 2},\\
    B_{\mathrm{PCS},i} = \begin{bmatrix}
        1 & 0 & 0 & 0 & 0 & 0\\
        0 & 1 & 0 & 0 & 0 & 0\\
        0 & 0 & 0 & 1 & 0 & 0\\
        0 & 0 & 0 & 0 & 1 & 0\\
    \end{bmatrix}^\mathrm{T} \in \mathbb{R}^{6 \times 4}.
\end{split}
\end{equation}

Next, we define a kinematic configuration of the robot. Following our choice of strain selectors, it has the following form:


## References
<a id="1">[1]</a> Renda, Federico, et al. "Discrete cosserat approach for multisection soft manipulator dynamics." 
IEEE Transactions on Robotics 34.6 (2018): 1518-1533.
