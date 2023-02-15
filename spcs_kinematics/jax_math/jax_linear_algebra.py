from jax import jit, vmap
import jax.numpy as jnp


@jit
def cross_prod_skew_matrix(phi: jnp.ndarray) -> jnp.ndarray:
    """
    Return the skew-symmetric matrix of a rotation vector phi.
    :param phi: rotation vector
    :return: skew-symmetric matrix
    """
    R = jnp.array(
        [
            [0.0, -phi[2], phi[1]],
            [phi[2], 0.0, -phi[0]],
            [-phi[1], phi[0], 0.0],
        ]
    )
    return R
