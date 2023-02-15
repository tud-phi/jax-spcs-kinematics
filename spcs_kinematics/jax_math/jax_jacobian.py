from jax import jit, vmap
import jax.numpy as jnp

from .jax_linear_algebra import cross_prod_skew_matrix


@jit
def geometric_to_analytical_jacobian_quat(
    geometric_jacobian: jnp.ndarray, quat: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Converts a geometric to an analytical jacobian for the quaternion representation
    :param geometric_jacobian: Geometric jacobian of shape (6, n_q)
        The first three rows correspond to the angular velocity and the last three rows to the linear velocity
    :param quat: Current rotation in quaternion representation of shape (4, ) and order (x, y, z, w)
    :param eps: small number to avoid division by zero
    :return: analytical Jacobian of shape (7, n_q)
        The first four rows correspond to the quaternion derivative and the last three rows to the linear velocity
    """
    H = jnp.concatenate(
        [
            cross_prod_skew_matrix(quat[:3]) + quat[3] * jnp.identity(3),
            -quat[0:3].reshape(3, 1),
        ],
        axis=1,
    )

    # mapping matrices from geometric to analytical Jacobian
    E_R_quat_inv = 0.5 * H.T  # for rotation in quaternion representation
    E_t_inv = jnp.identity(3)  # for translation in Cartesian coordinate system
    E_inv = jnp.block([[E_R_quat_inv, jnp.zeros((4, 3))], [jnp.zeros((3, 3)), E_t_inv]])

    # compute the analytical Jacobian using the constructed mapping matrix E_inv
    analytical_jacobian = jnp.matmul(E_inv, geometric_jacobian)

    return analytical_jacobian


# vectorized version of the geometric_to_analytical_quat_jacobian function
vgeometric_to_analytical_jacobian_quat = jit(
    vmap(geometric_to_analytical_jacobian_quat, in_axes=(2, 1, None), out_axes=2)
)
