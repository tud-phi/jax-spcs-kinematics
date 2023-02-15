from jax import jit, vmap
import jax.numpy as jnp


@jit
def transpose_quaternion(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Transposes a quaternion. Equivalent to inverting the rotation.
    :param quat: quaternion of order (x, y, z, w)
    :return: quat: quaternion of order (x, y, z, w)
    """
    quat = jnp.stack([-quat[0], -quat[1], -quat[2], quat[3]], axis=0)
    return quat


@jit
def quaternion_addition(quat1: jnp.ndarray, quat2: jnp.ndarray) -> jnp.ndarray:
    """
    Adds two quaternions.
    :param quat1: quaternion of order (x, y, z, w) and shape (4, )
    :param quat2: quaternion of order (x, y, z, w) and shape (4, )
    :return: quat_sum: summed quaternion of order (x, y, z, w) and shape (4, )
    """
    # first method
    # # left matrix of quaternion
    # M_left = jnp.block([
    #     [quat1[3] * jnp.identity(3) + cross_prod_skew_matrix(quat1[0:3]), quat1[0:3].reshape(3, 1)],
    #     [-quat1[0:3].reshape(1, 3), quat1[3].reshape(1, 1)]
    # ])
    # quat_sum = jnp.matmul(M_left, quat2)

    # second method
    epsilon = (
        quat1[3] * quat2[0:3]
        + quat2[3] * quat1[0:3]
        + jnp.cross(quat1[0:3], quat2[0:3])
    )
    nu = quat1[3] * quat2[3] - jnp.dot(quat1[0:3], quat2[0:3])
    quat_sum = jnp.concatenate(
        [
            epsilon,
            nu.reshape(
                1,
            ),
        ],
        axis=0,
    )
    return quat_sum


@jit
def quaternion_orientation_error(quat1: jnp.ndarray, quat2: jnp.ndarray) -> jnp.ndarray:
    """
    Error formulation for orientations expressed in quaternions used primarily for inverse kinematics
    Based on page 140 on: Siciliano, Bruno, et al. "Differential kinematics and statics."
    Robotics: Modelling, Planning and Control (2009): 105-160.
    :param quat1: quaternion of order (x, y, z, w) and shape (4, )
    :param quat2: quaternion of order (x, y, z, w) and shape (4, )
    :return: quat_sum: summed quaternion of order (x, y, z, w) and shape (4, )
    """
    nu1, nu2 = quat1[3], quat2[3]  # scalar parts of quaternions
    epsilon1, epsilon2 = quat1[0:3], quat2[0:3]  # vector parts of quaternions

    Delta_epsilon = nu1 * epsilon2 - nu2 * epsilon1 - jnp.cross(epsilon2, epsilon1)

    return Delta_epsilon
