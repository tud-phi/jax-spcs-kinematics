from jax import jit, vmap, jacfwd, lax
import jax.numpy as jnp

from .jax_cs import constant_strain_expmap


@jit
def pcs_expmap(
    s: jnp.ndarray, l0: jnp.ndarray, xi: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the exponential map of a piecewise constant strain curve
    :param s: point on the curve [0, L0] of shape (1,)
    :param l0: un-extended length of the curve for each segment of shape (n_S, )
    :param xi: piecewise constant strain in SE(3) of shape (n_S, 6)
    :param eps: small number to avoid division by zero
    :return: g: exponential map in se(3) of shape (4, 4)
    """
    n_S = xi.shape[0]

    # cum-sum of unextended segment lengths
    L0 = jnp.cumsum(jnp.concatenate([jnp.zeros((1,)), l0]))

    def body_fun(_i, _g):
        _s_i = jnp.clip(
            s - lax.dynamic_slice(L0, (_i - 1,), (1,)),
            a_min=0.0,
            a_max=lax.dynamic_slice(l0, (_i - 1,), (1,)),
        )
        _xi = xi[_i - 1, :]
        _g = _g @ constant_strain_expmap(_s_i, _xi, eps)
        return _g

    g = lax.fori_loop(
        lower=1, upper=n_S + 1, body_fun=body_fun, init_val=jnp.identity(4)
    )

    return g
