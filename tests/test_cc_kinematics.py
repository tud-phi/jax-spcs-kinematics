from spcs_kinematics.kinematic_parametrizations import ConstantCurvatureClosedForm
import jax.numpy as jnp
from numpy.testing import assert_allclose
import pytest
from scipy.spatial.transform import Rotation


def test_cc_forward_kinematics():
    kinematics = ConstantCurvatureClosedForm(L0=1.0, d=1.0)
    points = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])

    kinematics.state = jnp.array([0.5, 0.0, 0.0])
    transformations = kinematics.forward_kinematics(points)
    euler_angles = Rotation.from_matrix(
        transformations[:3, :3, :].transpose(2, 0, 1)
    ).as_euler("xyz", degrees=False)

    assert_allclose(
        euler_angles[:, 1], points * kinematics.state[0], atol=kinematics.eps
    )

    kinematics.state = jnp.array([0.0, 0.0, 0.5])
    transformations = kinematics.forward_kinematics(points)
    assert_allclose(
        transformations[2, 3, :],
        points + points / kinematics.L0 * kinematics.state[2],
        atol=kinematics.eps,
    )


def test_cc_inverse_kinematics():
    kinematics = ConstantCurvatureClosedForm(L0=1.0, d=1.0)
    points = jnp.array([0.25, 0.5, 0.75, 1.0])
    transformations = jnp.repeat(
        jnp.expand_dims(jnp.identity(4), axis=2), repeats=points.shape[0], axis=2
    )
    transformations = transformations.at[2, 3, :].set(
        jnp.array([0.375, 0.75, 1.125, 1.5])
    )

    state = kinematics.inverse_kinematics(transformations, points)

    # check if there are any numerical issues
    assert_allclose(jnp.isnan(state).any(), jnp.array(False))

    # check the correctness of delta_L
    assert_allclose(
        state[2], jnp.array([0.5, 0.5, 0.5, 0.5]), atol=10 ** 2 * kinematics.eps
    )
