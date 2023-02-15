from functools import partial
import jax
from jax import numpy as jnp
import progressbar
from typing import Callable, Tuple
import warnings


from .base_kinematic_parametrization import BaseKinematicParametrization
import spcs_kinematics.jax_math as jmath


class NumericKinematics(BaseKinematicParametrization):
    state: jnp.array = None
    transformation_fun: Callable = None
    pose_fun: Callable = None
    analytical_jacobian_fun: Callable = None

    def forward_kinematics(
        self, points: jnp.array, state: jnp.array = None
    ) -> jnp.array:
        if state is None:
            state = self.state
        return self.transformation_fun(points, state)

    def analytical_jacobian(
        self, points: jnp.array, state: jnp.array = None
    ) -> jnp.array:
        if state is None:
            state = self.state
        return self.analytical_jacobian_fun(points, state)

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
        # set the initial configuration
        q_init = state_init
        if q_init is None:
            q_init = jnp.zeros_like(self.state)
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

        self.state = q
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
