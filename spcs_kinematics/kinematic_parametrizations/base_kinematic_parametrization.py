from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import *


class BaseKinematicParametrization(ABC):
    def __init__(self, verbose: bool = False):
        super().__init__()

        self.verbose = verbose
        self.eps = 1.19209e-07

        self.state: jnp.array
        self.residual: jnp.array

    @abstractmethod
    def forward_kinematics(self, points: jnp.array) -> jnp.array:
        """
        Used to derive the transformation matrix from the state vector for a set of points along the backbone.
        :param: points: array of points s with shape (N, )
                points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
        :return: T: transformation matrix T for each point s as (4, 4, N)
        """
        pass

    @abstractmethod
    def inverse_kinematics(
        self,
        transformations: jnp.ndarray,
        points: jnp.ndarray,
        state_init: jnp.ndarray,
        num_iterations,
        translational_error_weight,
        rotational_error_weight,
        gamma: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        pass
