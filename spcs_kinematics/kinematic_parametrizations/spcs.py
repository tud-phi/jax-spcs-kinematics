import jax
from jax import numpy as jnp


from .numeric_kinematics import NumericKinematics
import spcs_kinematics.jax_math as jmath


class SelectivePiecewiseConstantStrain(NumericKinematics):
    def __init__(
        self,
        l0: jnp.ndarray,
        strain_selector_cs: jnp.ndarray = jnp.zeros((6,), dtype=bool),
        strain_selector_pcs: jnp.ndarray = jnp.ones((6,), dtype=bool),
        rest_strain: jnp.ndarray = None,
        *args,
        **kwargs,
    ):
        """
        Selective Piecewise Constant Strain kinematic parametrization
        Args:
            l0: array of shape (n_S, ) with segment lengths
            strain_selector_cs: array of shape (6, ) with boolean values indicating which components of the
                strain are constant along the entire rod
            strain_selector_pcs: array of shape (6, ) with boolean values indicating which components of the
                strain are piecewise constant along the rod
            rest_strain: array of shape (n_S + 1, 6) with the rest strain of the rod
        """
        super().__init__(*args, **kwargs)
        self.l0 = l0

        # number of segments
        num_segments = l0.shape[0]

        # make sure inputs are correct
        assert strain_selector_cs.shape == (6,) and strain_selector_cs.dtype == bool
        assert strain_selector_pcs.shape == (6,) and strain_selector_pcs.dtype == bool
        self.strain_selector_cs, self.strain_selector_pcs = strain_selector_cs, strain_selector_pcs
        self.strain_basis_cs = jmath.compute_strain_basis(strain_selector_cs)
        self.strain_basis_pcs = jmath.compute_strain_basis(strain_selector_pcs)

        if rest_strain is None:
            rest_strain = jnp.zeros((num_segments + 1, 6))
            # by default, set the axial rest strain across the rod to 1.0
            rest_strain = rest_strain.at[0, -1].set(1.0)
        else:
            assert rest_strain.shape == (num_segments + 1, 6)
        self.rest_strain = rest_strain

        # size of configuration vector
        self.configuration = jnp.zeros((
            1 + self.strain_basis_cs.shape[1] + num_segments * self.strain_basis_pcs.shape[1],
        ))

        spcs_forward_kinematics_vmapped = jax.jit(
            jax.vmap(
                jmath.spcs_forward_kinematics,
                in_axes=(None, None, None, 0, None, None, None),
                out_axes=2,
            )
        )
        self.transformation_fun = lambda _s, _q: spcs_forward_kinematics_vmapped(
            self.strain_basis_cs, self.strain_basis_pcs, self.rest_strain, _s, l0, _q, self.eps
        )
        spcs_forward_kinematics_quat_SE3_vmapped = jax.jit(
            jax.vmap(
                jmath.spcs_forward_kinematics_quat_SE3,
                in_axes=(None, None, None, 0, None, None, None),
                out_axes=1,
            )
        )
        self.pose_fun = lambda _s, _q: spcs_forward_kinematics_quat_SE3_vmapped(
            self.strain_basis_cs, self.strain_basis_pcs, self.rest_strain, _s, l0, _q, self.eps
        )

        spcs_autodiff_analytical_quat_jacobian_vmapped = jax.jit(
            jax.vmap(
                jmath.spcs_autodiff_analytical_quat_jacobian,
                in_axes=(None, None, None, 0, None, None, None),
                out_axes=2,
            )
        )
        self.analytical_jacobian_fun = (
            lambda _s, _q: spcs_autodiff_analytical_quat_jacobian_vmapped(
                self.strain_basis_cs, self.strain_basis_pcs, self.rest_strain, _s, l0, _q, self.eps
            )
        )
