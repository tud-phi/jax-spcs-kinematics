from jax import jit, vmap, jacfwd, lax
import jax.numpy as jnp

from .jax_cs import configuration_to_constant_strain, constant_strain_expmap
from .jax_lie_algebra import se3_to_quat_SE3
from .jax_pcs import pcs_expmap


@jit
def spcs_forward_kinematics(
    B_xi_cs: jnp.ndarray,
    B_xi_pcs: jnp.ndarray,
    xi0: jnp.ndarray,
    s: jnp.ndarray,
    l0: jnp.ndarray,
    q: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """
    Computes the forward kinematics of a selective piecewise constant strain curve
    :param B_xi_cs: strain basis for constant strain of shape (6, n_q_cs)
        basis for constant strain applied to entire rod
    :param B_xi_pcs: strain basis for piecewise constant strain of shape (6, n_q_pcs)
        basis for constant strain applied to each segment
    :param xi0: rest strain in SE(3) of shape (n_S + 1, 6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param l0: un-extended length of the curve for each segment of shape (n_S, )
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: T: exponential map in se(3) of shape (4, 4)
    """
    k = 0

    # the first entry is reserved for the azimuth angle at the base
    phi0 = q[k]
    k += 1

    # xi_cs: constant strain for entire rod: (6, )
    q_cs = q[k : k + B_xi_cs.shape[1]]
    xi_cs = configuration_to_constant_strain(B_xi_cs, xi0[0], q_cs)
    k += B_xi_cs.shape[1]

    vconfiguration_to_constant_strain = vmap(
        configuration_to_constant_strain, in_axes=(None, 0, 0), out_axes=0
    )

    # xi_pcs: constant strain for each segment: (n_S, 6)
    q_pcs = q[k:].reshape((-1, B_xi_pcs.shape[1]))
    xi_pcs = vconfiguration_to_constant_strain(B_xi_pcs, xi0[1:], q_pcs)

    # add xi_cs to each segment in xi_pcs
    xi = xi_pcs + xi_cs

    # compute transformation matrices based on phi0 and dphi_ds
    T_phi0 = jnp.array(
        [
            [jnp.cos(phi0), -jnp.sin(phi0), 0, 0],
            [jnp.sin(phi0), jnp.cos(phi0), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # compute exponential map
    g = pcs_expmap(s, l0, xi, eps)

    # convolute the two transformations
    T = jnp.matmul(T_phi0, g)

    return T


@jit
def spcs_forward_kinematics_quat_SE3(
    B_xi_cs: jnp.ndarray,
    B_xi_pcs: jnp.ndarray,
    xi0: jnp.ndarray,
    s: jnp.ndarray,
    l0: jnp.ndarray,
    q: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """
    Computes the forward kinematics of a constant strain curve and outputs the quaternion representation of SE(3)
    :param B_xi_cs: strain basis for constant strain of shape (6, n_q_cs)
        basis for constant strain applied to entire rod
    :param B_xi_pcs: strain basis for piecewise constant strain of shape (6, n_q_pcs)
        basis for constant strain applied to each segment
    :param xi0: rest strain in SE(3) of shape (n_S+1, 6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param l0: un-extended length of the curve for each segment of shape (n_S, )
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: chi: vector in SE(3) of shape (7, ) consisting of the rotational quaternion and the translation vector:
        chi = [quat_x, quat_y, quat_z, quat_w, t_x, t_y, t_z]
    """
    # compute the transformation matrix
    g = spcs_forward_kinematics(B_xi_cs, B_xi_pcs, xi0, s, l0, q, eps)

    # compute the quaternion representation of SE(3)
    chi = se3_to_quat_SE3(g, 2e1 * eps)

    return chi


@jit
def spcs_autodiff_analytical_quat_jacobian(
    B_xi_cs: jnp.ndarray,
    B_xi_pcs: jnp.ndarray,
    xi0: jnp.ndarray,
    s: jnp.ndarray,
    l0: jnp.ndarray,
    q: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """
    Computes the analytical Jacobian for a quaternion pose representation using jax autodiff
    :param B_xi_cs: strain basis for constant strain of shape (6, n_q_cs)
        basis for constant strain applied to entire rod
    :param B_xi_pcs: strain basis for piecewise constant strain of shape (6, n_q_pcs)
        basis for constant strain applied to each segment
    :param xi0: rest strain in SE(3) of shape (n_S+1, 6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param l0: un-extended length of the curve for each segment of shape (n_S, )
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: J: analytical jacobian of shape (7, n_q)
    """
    # make sure that the Jacobian does not become singular
    q = q + jnp.sign(q + eps) * 1e3 * eps

    chi_fun = lambda _q: spcs_forward_kinematics_quat_SE3(
        B_xi_cs, B_xi_pcs, xi0, s, l0, _q, 1e3 * eps
    )

    J = jacfwd(chi_fun)(q)

    return J
