from abc import ABC, abstractmethod
import jax.numpy as jnp
from typing import *


class BaseKinematicParametrization(ABC):
    """
    Base class for all kinematic parametrizations
    Parameters:
        configuration: configuration vector of the kinematic parametrization of shape (n_q, )
        eps: small number to avoid division by zero
    """
    configuration: jnp.array
    eps = 1.19209e-07

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: bool flag to enable verbose output
        """
        super().__init__()

        self.verbose = verbose

    @abstractmethod
    def forward_kinematics(self, points: jnp.array) -> jnp.array:
        """
        Used to derive the transformation matrix from the configuration vector for a set of points along the backbone.
        Args:
            points: array of points s with shape (N, ). points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
        Returns
            T: SE(3) transformation matrix T for each point s as array of shape (4, 4, N)
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
        """
        Run differential inverse kinematics to find the configuration vector q that minimizes the error between the
        desired transformation matrices and the transformation matrices computed from the configuration vector.
        Args:
            transformations: array of desired SE(3) transformation matrices T for each point s as array of shape (4, 4, N)
            points: array of points s with shape (N, ). points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
            state_init: initial guess for the configuration vector q as array of shape (n_q, )
            num_iterations: number of iterations to run the differential inverse kinematics for
            translational_error_weight: weight for the translational error
            rotational_error_weight: weight for the rotational error
            gamma: array of shape (n_q, ) that the step size / learning rate
        """
        pass
