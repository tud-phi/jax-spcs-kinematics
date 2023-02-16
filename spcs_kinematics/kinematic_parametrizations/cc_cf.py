import numpy as np

from . import BaseKinematicParametrization
from jax import jit
import jax.numpy as jnp


class ConstantCurvatureClosedForm(BaseKinematicParametrization):
    def __init__(self, L0: float, d: float = 1.0):
        super().__init__()

        # configuration: [Delta_x, Delta_y, delta_L]
        self.L0 = L0
        self.d = d
        self.configuration = jnp.zeros((3,))

    def forward_kinematics(self, points: jnp.array, configuration: jnp.array = None) -> jnp.array:
        """
        Computes the forward kinematics for the given points and configuration
        Args:
            points: array of points s with shape (N, )
                points are specified along the backbone in interval [0, L0]
                where L0 is the unelongated length of the rod
            configuration: array of shape (3, ) with the CC configuration as [Delta_x, Delta_y, delta_L]
        Returns:
            T: transformation matrix T from the base to each point s as (4, 4, N)
        """
        if configuration is None:
            configuration = self.configuration
        configuration_ss = jnp.repeat(
            jnp.expand_dims(configuration, axis=1), repeats=points.shape[0], axis=1
        )
        return compute_cc_forward_kinematics(configuration_ss, points, self.L0, self.d, self.eps)

    def inverse_kinematics(
        self,
        transformations: jnp.array,
        points: jnp.array,
    ) -> jnp.array:
        """
        Computes the inverse kinematics for the given points and transformation matrices
        Args:
            transformations: array of transformation matrices T from the base to each point s as (4, 4, N)
            points: array of points s with shape (N, )
        Returns:
            configuration: array of shape (3, ) with the CC configuration as [Delta_x, Delta_y, delta_L]
        """
        self.configuration = compute_cc_inverse_kinematics(
            transformations, points, self.L0, self.d, self.eps
        )
        return self.configuration

    def compute_residual(self) -> jnp.array:
        pass


@jit
def compute_cc_forward_kinematics(
    q: jnp.array, s: jnp.array, L0: float, d: float, eps: float
) -> jnp.array:
    """
    Used to derive the transformation matrices for each point s along the backbone.
    :param q: configuration vector of dimension (3, n)
    :param s: vector with points s along the backbone in interval [0, L0] and dimension (n,)
    :param L0: unelongated length of the rod
    :param d: radius of the rod
    :param eps: small number to avoid numerical issues
    :return: transformation matrix T for each point s as (4, 4, n)
    """

    # scale configuration to the respective point along the rod
    q = s / L0 * q

    # add eps to avoid division by zero
    q = q + eps

    Delta_norm = jnp.sqrt(q[0] ** 2 + q[1] ** 2)

    # s_i and c_i
    q_sin = jnp.sin(Delta_norm / d)
    q_cos = jnp.cos(Delta_norm / d)

    R1 = jnp.stack(
        [
            1 + q[0] ** 2 / Delta_norm ** 2 * (q_cos - 1),
            q[0] * q[1] / Delta_norm ** 2 * (q_cos - 1),
            q[0] / Delta_norm * q_sin,
        ],
        axis=0,
    )
    R2 = jnp.stack(
        [
            q[0] * q[1] / Delta_norm ** 2 * (q_cos - 1),
            1 + q[1] ** 2 / Delta_norm ** 2 * (q_cos - 1),
            q[1] / Delta_norm * q_sin,
        ],
        axis=0,
    )
    R3 = jnp.stack(
        [-q[0] / Delta_norm * q_sin, -q[1] / Delta_norm * q_sin, q_cos], axis=0
    )
    R = jnp.stack([R1, R2, R3], axis=0)

    t = (d * (s + q[2]) / (Delta_norm ** 2)) * jnp.stack(
        [q[0] * (1 - q_cos), q[1] * (1 - q_cos), Delta_norm * q_sin], axis=0
    )

    T = jnp.concatenate([R, t.reshape(3, 1, -1)], axis=1)
    T4 = np.repeat(
        jnp.array([[[0], [0], [0], [1]]], dtype=R.dtype), repeats=s.shape[0], axis=2
    )
    T = jnp.concatenate([T, T4], axis=0)

    return T


# @jit
def compute_cc_inverse_kinematics(
    T: jnp.array, s: jnp.array, L0: float, d: float, eps: float
) -> jnp.array:
    """
    Used to derive the configuration vector from the transformation matrices for a set of points along the backbone.
    :param T: transformation matrix from the base of the segment for each point s as (4, 4, n)
    :param s: vector with points s along the backbone in interval [0, L0] and dimension (n,)
    :param L0: unelongated length of the rod
    :param d: radius of the rod
    :param eps: small number to avoid numerical issues
    :return: q
    """
    # rotation matrix to each point
    R = T[:3, :3, :]
    # translation vector to each point
    t = T[:3, 3, :]

    # address numerical issues of the rotation matrix |R[2, 2]| >= 1
    R = R.at[2, 2].set(R[2, 2] - jnp.sign(R[2, 2]) * 10 ** 2 * eps)

    # extension of the rod [m]
    delta_L = t[2] * (jnp.arccos(R[2, 2]) / jnp.sin(jnp.arccos(R[2, 2]))) - s

    delta_c = (d / (s + delta_L)) * ((jnp.arccos(R[2, 2])) ** 2 / (1 - R[2, 2]))

    Delta_x = t[0] * delta_c
    Delta_y = t[1] * delta_c

    # configuration of each point (not scaled yet to the full rod)
    q = jnp.stack([Delta_x, Delta_y, delta_L], axis=0)

    # scale q to the full rod length
    q = (L0 / s) * q

    return q
