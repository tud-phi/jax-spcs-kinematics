from jax import jit, vmap, jacfwd
import jax.numpy as jnp

from .jax_lie_algebra import (
    strain_SE3_to_se3,
    strain_inv_Adjoint,
    strain_adjoint,
    se3_to_quat_SE3,
)


def compute_strain_basis(
    strain_selector: jnp.ndarray = jnp.ones((6,), dtype=bool),
) -> jnp.ndarray:
    n_q = strain_selector.sum().item()
    strain_basis = jnp.zeros((6, n_q), dtype=int)
    strain_basis_cumsum = jnp.cumsum(strain_selector)
    for i in range(strain_selector.shape[0]):
        j = int(strain_basis_cumsum[i].item()) - 1
        if strain_selector[i].item() is True:
            strain_basis = strain_basis.at[i, j].set(1)
    return strain_basis


@jit
def configuration_to_constant_strain(
    B_xi: jnp.ndarray, xi0: jnp.ndarray, q: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the (constant) strain for a given configuration q and the strain basis matrix B_xi
    :param B_xi: strain basis of shape (6, n_q)
    :param xi0: rest strain in SE(3) of shape (6,)
    :param q: configuration of the soft segment (n_q,)
    :return:
    """
    xi = jnp.matmul(B_xi, q) + xi0
    return xi


@jit
def constant_strain_expmap(s: jnp.ndarray, xi: jnp.ndarray, eps: float) -> jnp.ndarray:
    """
    Computes the exponential map for constant strain xi in SE(3)
    https://github.com/kucars/Discrete_Cosserat_CB/blob/master/src/piecewise_expmap.m
    :param s: point on the curve [0, L0] of shape (1,)
    :param xi: constant strain in SE(3) of shape (6,)
    :param eps: small number to avoid division by zero
    :return: g: exponential map in se(3) of shape (4x4)
    """
    xi_hat = strain_SE3_to_se3(xi, eps)

    # magnitude of rotational strain [rad / m]
    # equal to sqrt(k^T * k) where k is the angular strain vector
    theta = jnp.sqrt(jnp.dot(xi[:3], xi[:3]))

    g = (
        jnp.identity(4)
        + s * xi_hat
        + ((1 - jnp.cos(s * theta)) / (theta**2 + eps))
        * jnp.linalg.matrix_power(xi_hat, 2)
        + ((s * theta - jnp.sin(s * theta)) / (theta**3 + eps))
        * jnp.linalg.matrix_power(xi_hat, 3)
    )

    return g


@jit
def constant_strain_tangential_expmap(
    s: jnp.ndarray, xi: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the exponential map for constant strain xi in SE(3)
    https://github.com/kucars/Discrete_Cosserat_CB/blob/master/src/piecewise_ADJ.m
    :param s: point on the curve [0, L0] of shape (1,)
    :param xi: constant strain in SE(3) of shape (6,)
    :param eps: small number to avoid division by zero
    :return: T_xi: tangent operator of exponential map for constant strain xi
    """
    ad_xi = strain_adjoint(xi)

    # magnitude of rotational strain [rad / m]
    # equal to sqrt(k^T * k) where k is the angular strain vector
    theta = jnp.sqrt(jnp.dot(xi[:3], xi[:3]))

    T_xi = (
        jnp.identity(6)
        + (
            (4 - 4 * jnp.cos(s * theta) - s * theta * jnp.sin(s * theta))
            / (2 * theta**2 + eps)
        )
        * ad_xi
        + (
            (4 * s * theta - 5 * jnp.sin(s * theta) + s * theta * jnp.cos(s * theta))
            / (2 * theta**3 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 2)
        + (
            (2 - 2 * jnp.cos(s * theta) - s * theta * jnp.sin(s * theta))
            / (2 * theta**4 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 3)
        + (
            (2 * s * theta - 3 * jnp.sin(s * theta) + s * theta * jnp.cos(s * theta))
            / (2 * theta**5 + eps)
        )
        * jnp.linalg.matrix_power(ad_xi, 4)
    )

    return T_xi


@jit
def constant_strain_forward_kinematics(
    B_xi: jnp.ndarray, xi0: jnp.ndarray, s: jnp.ndarray, q: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the forward kinematics of a constant strain curve
    :param B_xi: strain basis of shape (6, n_q)
    :param xi0: rest strain in SE(3) of shape (6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: g: exponential map in se(3) of shape (4, 4)
    """
    xi = configuration_to_constant_strain(B_xi, xi0, q)

    # compute exponential map
    g = constant_strain_expmap(s, xi, eps)

    return g


@jit
def constant_strain_forward_kinematics_quat_SE3(
    B_xi: jnp.ndarray, xi0: jnp.ndarray, s: jnp.ndarray, q: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the forward kinematics of a constant strain curve and outputs the quaternion representation of SE(3)
    :param B_xi: strain basis of shape (6, n_q)
    :param xi0: rest strain in SE(3) of shape (6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: chi: vector in SE(3) of shape (7, ) consisting of the rotational quaternion and the translation vector:
        chi = [quat_x, quat_y, quat_z, quat_w, t_x, t_y, t_z]
    """
    # compute the transformation matrix
    g = constant_strain_forward_kinematics(B_xi, xi0, s, q, eps)

    # compute the quaternion representation of SE(3)
    chi = se3_to_quat_SE3(g, eps)

    return chi


@jit
def constant_strain_geometric_jacobian(
    B_xi: jnp.ndarray, xi0: jnp.ndarray, s: jnp.ndarray, q: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the forward kinematics of a constant strain curve
    ATTENTION: This geometric jacobian is probably not correct (especially the t_z components)
    :param B_xi: strain basis of shape (6, n_q)
    :param xi0: rest strain in SE(3) of shape (6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: J: geometric Jacobian of shape (6, n_q)
    """
    xi = configuration_to_constant_strain(B_xi, xi0, q)

    inv_Adjoint = strain_inv_Adjoint(s, xi, eps)
    T_xi = constant_strain_tangential_expmap(s, xi, eps)

    # geometric Jacobian
    J = jnp.matmul(jnp.matmul(inv_Adjoint, T_xi), B_xi)

    return J


@jit
def constant_strain_autodiff_analytical_quat_jacobian(
    B_xi: jnp.ndarray, xi0: jnp.ndarray, s: jnp.ndarray, q: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the analytical Jacobian for a quaternion pose representation using jax autodiff
    :param B_xi: strain basis of shape (6, n_q)
    :param xi0: rest strain in SE(3) of shape (6,)
    :param s: point on the curve [0, L0] of shape (1,)
    :param q: configuration of the soft segment (n_q,)
    :param eps: small number to avoid division by zero
    :return: J: analytical jacobian of shape (7, n_q)
    """
    # make sure that the Jacobian does not become singular
    q = q + jnp.sign(q + eps) * 1e3 * eps

    chi_fun = lambda _q: constant_strain_forward_kinematics_quat_SE3(
        B_xi, xi0, s, _q, eps
    )

    J = jacfwd(chi_fun)(q)

    return J
