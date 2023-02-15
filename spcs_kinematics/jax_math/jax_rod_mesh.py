from functools import partial
from jax import jit, vmap
import jax.numpy as jnp


@jit
def backbone_radial_transformation(T_s: jnp.ndarray, r: jnp.ndarray, phi: jnp.ndarray) -> jnp.array:
    """
    Computes the transformation of a backbone curve with radial strain
    :param T_s: transformation matrix from the inertial / base frame to a frame s along the backbone of the rod.
        np.ndarray of shape (4, 4)
    :param r: radial offset of the desired point from the backbone curve. jnp.ndarray of shape (1,)
    :param phi: azimuth angle of point. jnp.ndarray of shape (1,)
    :return T_r: transformation matrix from the inertial / base frame to a frame at a radial distance of the rod
        np.ndarray of shape (4, 4)
    """
    T_r = T_s @ jnp.array([
        [jnp.cos(phi), -jnp.sin(phi), 0, r * jnp.cos(phi)],
        [jnp.sin(phi), jnp.cos(phi), 0, r * jnp.sin(phi)],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])

    return T_r


vbackbone_radial_transformation = vmap(backbone_radial_transformation, in_axes=(None, 0, 0), out_axes=-1)


@partial(jit, static_argnames=("r_resolution", "phi_resolution"))
def generate_infinitesimal_cylindrical_mesh_points(
        T_s: jnp.ndarray,
        outside_radius: jnp.ndarray,
        inside_radius: jnp.ndarray,
        r_resolution: int = 10,
        phi_resolution: int = 32,
):
    """
    Generates a mesh of points on a cylinder with a radius of 1.
    :param T_s: transformation matrix from the inertial / base frame to a frame s along the backbone of the rod.
        np.ndarray of shape (4, 4)
    :param outside_radius: outside radius of the outside of the rod. jnp.ndarray of shape (1,)
    :param inside_radius: inside radius of the inside of the rod. jnp.ndarray of shape (1,)
    :param r_resolution: radial resolution (e.g. number of points along the radial direction)
    :param phi_resolution: azimuthal resolution (e.g. number of points along the azimuthal direction)
    :return T_r: transformation matrix from the inertial / base frame to a frame at a radial distance of the rod
        np.ndarray of shape (4, 4)
    """
    # Define grid in polar coordinates
    r = jnp.linspace(inside_radius, outside_radius, r_resolution)
    phi = jnp.linspace(0, (1.0 - 5e-5) * 2 * jnp.pi, phi_resolution)
    r_matrix, phi_matrix = jnp.meshgrid(r, phi)

    rr = r_matrix.ravel()
    phiphi = phi_matrix.ravel()

    # transform into cartesian space
    T_r = vbackbone_radial_transformation(T_s, rr, phiphi)

    return T_r
