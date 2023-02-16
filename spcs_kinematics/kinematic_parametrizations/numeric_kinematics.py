from functools import partial
import jax
from jax import numpy as jnp
import progressbar
from typing import Callable, Tuple
import warnings


from .base import BaseKinematicParametrization
import spcs_kinematics.jax_math as jmath


class NumericKinematics(BaseKinematicParametrization):
    """
    Base class for kinematic parametrizations that are based on numerical evaluation of the forward kinematics.
    Parameters:
        configuration: array of shape (n_q, ) with kinematic configuration (i.e. the configuration of the system)
        transformation_fun: function that computes the SE(3) transformation of a set of points given a configuration
            Needs to have the signature transformation_fun(s, q) -> T
            where s is an array of shape (N, ), q is an array of shape (n_q, ), and T is an array of shape (4, 4, N)
            N is the number of points, n_q is the number of configuration variables
        pose_fun: function that computes the pose of a set of points given a configuration
            Needs to have the signature pose_fun(s, q) -> chi
            where s is an array of shape (N, ), q is an array of shape (n_q, ), and chi is an array of shape (7, N)
            N is the number of points, n_q is the number of configuration variables (i.e. the configuration of the system)
            chi[:, i] is the pose of the ith point in the form of a vector of shape (7, ) with the first three entries
            being the translation and the last three entries being the rotation in quaternion form
        analytical_jacobian_fun: function that computes the analytical Jacobian of the pose chi w.r.t. the configuration q
            Needs to have the signature analytical_jacobian_fun(s, q) -> J
            where s is an array of shape (N, ), q is an array of shape (n_q, ), and J is an array of shape (7, n_q, N)
            N is the number of points, n_q is the number of configuration variables (i.e. the configuration of the system)
            J[:, :, i] is the Jacobian of the ith point in the form of a matrix of shape (7, n_q)
    """

    configuration: jnp.array = None
    transformation_fun: Callable = None
    pose_fun: Callable = None
    analytical_jacobian_fun: Callable = None

    def forward_kinematics(
        self, points: jnp.array, configuration: jnp.array = None
    ) -> jnp.array:
        """
        Computes the forward kinematics for the given points and configuration
        Args:
            points: array of points s with shape (N, )
                points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
            configuration: array of shape (n_q, ) with kinematic configuration (i.e. the configuration of the system)
        Returns:
            T: transformation matrix T from the base to each point s as (4, 4, N)
        """
        if configuration is None:
            configuration = self.configuration
        return self.transformation_fun(points, configuration)

    def analytical_jacobian(
        self, points: jnp.array, configuration: jnp.array = None
    ) -> jnp.array:
        """
        Computes the analytical Jacobian of the pose chi w.r.t. the configuration q
        Args:
            points: array of points s with shape (N, )
                points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
            configuration: array of shape (n_q, ) with kinematic configuration (i.e. the configuration of the system)
        Returns:
            J: Jacobian of the pose chi w.r.t. the configuration q of shape (7, n_q, N)
        """
        if configuration is None:
            configuration = self.configuration
        return self.analytical_jacobian_fun(points, configuration)

    def inverse_kinematics(
        self,
        transformations: jnp.ndarray,
        points: jnp.ndarray,
        state_init: jnp.ndarray = None,
        num_iterations: int = 100,
        translational_error_weight: float = 1.0,
        rotational_error_weight: float = 1.0,
        gamma: jnp.ndarray = jnp.array(1e-3),
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes the inverse kinematics for the given points and SE(3) transformations
        Args:
            transformations: array of SE(3) transformations T from the base to each point s as (4, 4, N)
            points: array of points s with shape (N, )
                points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
            state_init: array of shape (n_q, ) with initial kinematic configuration (i.e. the configuration of the system)
            num_iterations: number of iterations to run the differential inverse kinematics for
            translational_error_weight: weight for the translational error during the optimization
            rotational_error_weight: weight for the rotational error during the optimization
            gamma: step size / learning rate of the gradient descent optimization
        Returns:
            q: array of shape (n_q, ) with optimized kinematic configuration (i.e. the configuration of the system)
            e_chi: array of shape (7, N) with the final pose error
            q_its: array of shape (num_iterations, n_q) with the kinematic configuration at each iteration
            e_chi_its: array of shape (num_iterations, 7, N) with the pose error at each iteration
        """

        # set the initial configuration
        q_init = state_init
        if q_init is None:
            q_init = jnp.zeros_like(self.configuration)
        q = q_init.copy()

        # compute goal pose
        chi_goal = jmath.vse3_to_quat_SE3(transformations, 2e1 * self.eps)
        # compute the first pose error
        e_chi = chi_goal - self.pose_fun(points, q)

        q_its, e_chi_its = [], []
        for it in progressbar.progressbar(range(num_iterations), redirect_stdout=True):
            q_temp, e_chi = inverse_kinematics_step(
                self.pose_fun,
                self.analytical_jacobian_fun,
                points,
                chi_goal,
                q,
                translational_error_weight,
                rotational_error_weight,
                gamma,
                self.eps,
            )

            if jnp.isnan(q_temp).any():
                warnings.warn("Encountered NaNs during differential inverse kinematics")
                break

            q = q_temp
            q_its.append(q)
            e_chi_its.append(e_chi)

            if self.verbose:
                print("it", it, "q", q)

        q_its = jnp.stack(q_its, axis=0)
        e_chi_its = jnp.stack(e_chi_its, axis=0)

        self.configuration = q
        return q, e_chi, q_its, e_chi_its


@partial(jax.jit, static_argnums=(0, 1))
def inverse_kinematics_step(
    pose_fun: Callable,
    analytical_jacobian_fun: Callable,
    points: jnp.ndarray,
    chi_goal: jnp.ndarray,
    q_init: jnp.ndarray,
    translational_error_weight: float,
    rotational_error_weight: float,
    gamma: jnp.ndarray,
    eps: float,
) -> jnp.array:
    # current pose in SE(3) of quaternion representation
    chi_current = pose_fun(points, q_init)

    # define the pose error as the difference in SE(3) poses
    e_chi = chi_goal - chi_current

    # weight the pose error using rotational and translational error weights
    weights = jnp.concatenate(
        [
            rotational_error_weight * jnp.ones((4,)),
            translational_error_weight * jnp.ones((3,)),
        ],
        axis=0,
    )
    weighted_pose_error = jnp.einsum("i,ij->ij", weights, e_chi)

    # reshape to 6N (vertically stack for each point)
    flattened_pose_error = weighted_pose_error.transpose((1, 0)).reshape((-1,))

    # compute the analytical jacobian matrices
    analytical_jac = analytical_jacobian_fun(points, q_init)

    # stack the jacobians of all points (e.g. multi-task of equal priority)
    # First transpose to get shape (N, 6, n_q) instead of (6, n_q, N)
    stacked_jac = analytical_jac.transpose((2, 0, 1))
    # Second reshape to get shape (N * 6, n_q)
    stacked_jac = stacked_jac.reshape((-1, stacked_jac.shape[-1]))

    # transpose the multi-task Jacobian to get shape (n_q, N * 6)
    stacked_jac_T = stacked_jac.transpose((1, 0))

    # compute the pseudo-inverse of the multi-task Jacobian
    # looks like the pseudo-inverse is badly conditioned
    # stacked_jac_pinv = jnp.linalg.pinv(stacked_jac, rcond=eps)

    # implement gradient descent
    delta_q = gamma * jnp.matmul(stacked_jac_T, flattened_pose_error)

    # update q according to the gradient descent
    q = q_init + delta_q

    return q, e_chi
