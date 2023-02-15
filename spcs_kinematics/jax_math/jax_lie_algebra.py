from jax import jit, vmap
import jax.numpy as jnp

from .jax_linear_algebra import cross_prod_skew_matrix
from .jax_rotation import rotmat_to_quat, quat_to_rotmat


@jit
def inverse_transformation_matrix(T: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the inverse of a SE(3) transformation matrix
    :param T: Transformation matrix of shape (4, 4)
    :return: T_inv: Inverse of transformation matrix of shape (4, 4)
    """
    T_inv = jnp.block(
        [
            [T[:3, :3].T, -T[:3, :3].T @ T[:3, 3:]],
            [jnp.zeros((1, 3)), jnp.ones((1, 1))],
        ]
    )
    return T_inv


@jit
def screw_SE3_to_se3(screw: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Converts a screw vector in SE(3) to a transformation matrix in se(3)
    :param screw: screw vector of shape (6, )
    :param eps: small number to avoid division by zero
    :return: T: transformation matrix of shape (4, 4)
    """
    # construct skew matrix from rotational strains
    R = cross_prod_skew_matrix(screw[0:3])
    # construct translation vector
    t = screw[3:6]
    # construct se(3) matrix from xi
    T = jnp.block(
        [
            [R, t.reshape(3, 1)],
            [jnp.zeros((1, 3)), jnp.ones((1, 1))],
        ]
    )
    return T


@jit
def quat_SE3_to_se3(chi_quat: jnp.ndarray) -> jnp.ndarray:
    """
    Converts a pose vector in SE(3) with the rotation in quaternion representation to a transformation matrix in se(3)
    :param screw: pose vector of shape (7, ) consisting of
        chi = [quat_x, quat_y, quat_z, quat_w, t_x, t_y, t_z]
    :param eps: small number to avoid division by zero
    :return: T: transformation matrix of shape (4, 4)
    """
    # construct skew matrix from rotational strains
    R = quat_to_rotmat(chi_quat[0:4])
    # construct translation vector
    t = chi_quat[4:7]
    # construct se(3) matrix
    T = jnp.block(
        [
            [R, t.reshape(3, 1)],
            [jnp.zeros((1, 3)), jnp.ones((1, 1))],
        ]
    )
    return T


@jit
def se3_to_screw_SE3(T: jnp.array, eps: float):
    """
    Converts a transformation matrix in se(3) of dim R^4x4 to a screw vector in SE(3) of dim R^6
    :param T: transformation matrix in se(3)
    :param eps: small number to avoid division by zero
    :return: chi: screw vector in SE(3) consisting of the rotation vector and the translation vector:
        chi = (rot_vec^T, t^T)
    """
    # extract translation vector and rotation matrix
    t = T[0:3, 3]
    R = T[0:3, 0:3]

    # rotation angle theta while making sure that the input into arccos is in [-1, 1]
    theta = jnp.arccos((jnp.trace(R) - 1) / 2 - jnp.sign(jnp.trace(R)) * eps)
    # rotation axis n
    n = (
        jnp.stack(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ],
            axis=0,
        )
        / (2 * jnp.sin(theta) + eps)
    )
    # rotation vector
    rot_vec = theta * n

    screw = jnp.concatenate([rot_vec, t], axis=0)
    return screw


@jit
def se3_to_quat_SE3(g: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Converts a transformation matrix in se(3) of shape (4, 4)
    to a pose vector in SE(3) containing quaternions of shape (7, )
    :param g: transformation matrix in se(3) of shape (4, 4)
    :param eps: small number to avoid division by zero
    :return: chi: vector in SE(3) of shape (7, ) consisting of the rotational quaternion and the translation vector:
        chi = [quat_x, quat_y, quat_z, quat_w, t_x, t_y, t_z]
    """
    # extract translation vector and rotation matrix
    t = g[0:3, 3]
    R = g[0:3, 0:3]

    # compute quaternion
    quat = rotmat_to_quat(R, eps)

    # compute the SE(3) pose vector
    chi = jnp.concatenate([quat, t], axis=0)

    return chi


# vectorized version of the se3_to_quat_SE3 function
vse3_to_quat_SE3 = jit(vmap(se3_to_quat_SE3, in_axes=(2, None), out_axes=1))


@jit
def strain_SE3_to_se3(xi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Converts a strain in SE(3) to a strain matrix in se(3)
    :param xi: screw vector of shape (6, )
    :param eps: small number to avoid division by zero
    :return: xi_hat: transformation matrix of shape (4, 4)
    """
    # construct skew matrix from rotational strains
    k_tilde = cross_prod_skew_matrix(xi[0:3])
    # construct translational strain vector
    p = xi[3:6]
    # construct se(3) matrix from xi
    xi_hat = jnp.block(
        [
            [k_tilde, p.reshape(3, 1)],
            [jnp.zeros((1, 3)), jnp.zeros((1, 1))],
        ]
    )
    return xi_hat


@jit
def strain_adjoint(xi: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the adjoint map of the strain xi
    https://github.com/kucars/Discrete_Cosserat_CB/blob/master/src/dinamico_adj.m
    :param: xi: strain array in SE(3) of shape (6, )
    :return: ad_xi: adjoint map of lie group of shape (6, 6)
    """
    k_tilde = cross_prod_skew_matrix(xi[:3])  # skew matrix of rotational strain
    p_tilde = cross_prod_skew_matrix(xi[3:6])  # skew matrix of translation strain

    ad_xi = jnp.block([[k_tilde, jnp.zeros((3, 3))], [p_tilde, k_tilde]])
    return ad_xi


@jit
def strain_inv_Adjoint(s: jnp.ndarray, xi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Computes the inverse of the Adoint map of the transformation matrix g in se(3)
    https://github.com/kucars/Discrete_Cosserat_CB/blob/master/src/piecewise_invAdjoint.m
    :param: s: point on the curve [0, L0] of shape (1,)
    :param: xi: SE(3) array with strains of shape (6,)
    :param: eps: small number to avoid division by zero
    :return: inv_Adjoint: Inverse adjoint map of Lie group of shape (6, 6)
    """
    ad_xi = strain_adjoint(xi)

    # magnitude of rotational strain [rad / m]
    # equal to sqrt(k^T * k) where k is the angular strain vector
    theta = jnp.sqrt(jnp.dot(xi[:3], xi[:3]))

    inv_Adjoint = (
        jnp.identity(6)
        - (
            (3 * jnp.sin(s * theta) - s * theta * jnp.cos(s * theta))
            / (2 * theta + eps)
        )
        * ad_xi
        + (
            (4 - 4 * jnp.cos(s * theta) - s * theta * jnp.sin(s * theta))
            / (2 * theta ** 2 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 2)
        + (
            (jnp.sin(s * theta) - s * theta * jnp.cos(s * theta))
            / (2 * theta ** 3 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 3)
        + (
            (2 - 2 * jnp.cos(s * theta) - s * theta * jnp.sin(s * theta))
            / (2 * theta ** 4 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 4)
    )

    return inv_Adjoint
