from jax import jit, vmap
import jax.numpy as jnp
from .jax_linear_algebra import cross_prod_skew_matrix


@jit
def rotmat_to_quat(R: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Converts a rotation matrix to a quaternion.
    :param R: rotation matrix
    :param eps: small number to avoid division by zero.
        Attention: Needs to be a bit bigger than usual. Try something like 1e-6
    :return: quat: quaternion of order (x, y, z, w)
    """

    quat_x = (
        0.5
        * jnp.sign(R[2, 1] - R[1, 2])
        * jnp.sqrt(R[0, 0] - R[1, 1] - R[2, 2] + 1 + eps)
    )
    quat_y = (
        0.5
        * jnp.sign(R[0, 2] - R[2, 0])
        * jnp.sqrt(-R[0, 0] + R[1, 1] - R[2, 2] + 1 + eps)
    )
    quat_z = (
        0.5
        * jnp.sign(R[1, 0] - R[0, 1])
        * jnp.sqrt(-R[0, 0] - R[1, 1] + R[2, 2] + 1 + eps)
    )
    quat_w = 0.5 * jnp.sqrt(R[0, 0] + R[1, 1] + R[2, 2] + 1 + eps)

    # quaternion
    quat = jnp.stack([quat_x, quat_y, quat_z, quat_w], axis=0)

    return quat


@jit
def quat_to_rotmat(quat: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a quaternion to a rotation matrix.
    :param: quat: quaternion of order (x, y, z, w)
    :param: eps: small number to avoid division by zero.
        Attention: Needs to be a bit bigger than usual. Try something like 1e-6
    :return: R: rotation matrix
    """
    quat_skew = cross_prod_skew_matrix(quat[0:3])

    R = (
        jnp.identity(3)
        + 2 * quat[3] * quat_skew
        + 2 * jnp.linalg.matrix_power(quat_skew, 2)
    )

    return R


@jit
def rotmat_to_euler_xyz(R: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a rotation matrix to euler angles.
    :param R: rotation matrix of shape (3, 3)
    :param eps: small number to avoid division by zero.
        Attention: Needs to be a bit bigger than usual. Try something like 1e-6
    :return: euler: euler angles of order XYZ of shape (3,)
    """

    # euler angles
    euler_x = jnp.arctan2(-R[1, 2], R[2, 2])
    euler_y = jnp.arctan2(R[0, 2] * jnp.cos(euler_x), R[2, 2])
    euler_z = jnp.arctan2(-R[0, 1], R[0, 0])

    euler = jnp.stack([euler_x, euler_y, euler_z], axis=0)

    return euler


@jit
def euler_xyz_to_rotmat(euler: jnp.ndarray) -> jnp.ndarray:
    """
    Converts euler angles to a rotation matrix.
    :param euler: euler angles of order XYZ of shape (3,)
    :return: R: rotation matrix of shape (3, 3)
    """

    # rotation matrices
    R_x = jnp.array(
        [
            [1, 0, 0],
            [0, jnp.cos(euler[0]), -jnp.sin(euler[0])],
            [0, jnp.sin(euler[0]), jnp.cos(euler[0])],
        ]
    )
    R_y = jnp.array(
        [
            [jnp.cos(euler[1]), 0, jnp.sin(euler[1])],
            [0, 1, 0],
            [-jnp.sin(euler[1]), 0, jnp.cos(euler[1])],
        ]
    )
    R_z = jnp.array(
        [
            [jnp.cos(euler[2]), -jnp.sin(euler[2]), 0],
            [jnp.sin(euler[2]), jnp.cos(euler[2]), 0],
            [0, 0, 1],
        ]
    )

    # rotation matrix
    R = jnp.matmul(R_z, jnp.matmul(R_y, R_x))

    return R
