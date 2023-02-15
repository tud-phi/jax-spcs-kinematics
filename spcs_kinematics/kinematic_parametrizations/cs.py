from functools import partial
import jax
from jax import numpy as jnp
import progressbar
from typing import Callable, Tuple
import warnings


from .numeric_kinematics import NumericKinematics
import spcs_kinematics.jax_math as jmath


class ConstantStrain(NumericKinematics):
    def __init__(
        self,
        strain_selector: jnp.ndarray = jnp.ones((6,), dtype=bool),
        rest_strain: jnp.ndarray = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # make sure inputs are correct
        assert strain_selector.shape == (6,) and strain_selector.dtype == bool
        assert rest_strain.shape == (6,)

        self.strain_selector = strain_selector
        self.strain_basis = jmath.compute_strain_basis(strain_selector)
        self.rest_strain = rest_strain

        # size of configuration vector
        self.state = jnp.zeros((self.strain_basis.shape[1],))

        vconstant_strain_forward_kinematics = jax.jit(
            jax.vmap(
                jmath.constant_strain_forward_kinematics,
                in_axes=(None, None, 0, None, None),
                out_axes=2,
            )
        )
        self.transformation_fun = lambda _s, _q: vconstant_strain_forward_kinematics(
            self.strain_basis, self.rest_strain, _s, _q, self.eps
        )
        vconstant_strain_forward_kinematics_quat_SE3 = jax.jit(
            jax.vmap(
                jmath.constant_strain_forward_kinematics_quat_SE3,
                in_axes=(None, None, 0, None, None),
                out_axes=1,
            )
        )
        self.pose_fun = lambda _s, _q: vconstant_strain_forward_kinematics_quat_SE3(
            self.strain_basis, self.rest_strain, _s, _q, self.eps
        )

        vconstant_strain_autodiff_analytical_quat_jacobian = jax.jit(
            jax.vmap(
                jmath.constant_strain_autodiff_analytical_quat_jacobian,
                in_axes=(None, None, 0, None, None),
                out_axes=2,
            )
        )
        self.analytical_jacobian_fun = (
            lambda _s, _q: vconstant_strain_autodiff_analytical_quat_jacobian(
                self.strain_basis, self.rest_strain, _s, _q, self.eps
            )
        )
