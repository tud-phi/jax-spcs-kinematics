from jax import numpy as jnp

from spcs_kinematics.kinematic_parametrizations import SelectivePiecewiseConstantStrain
import spcs_kinematics.jax_math as jmath
from spcs_kinematics.visualization import (
    plot_rod_shape,
    plot_inverse_kinematics_convergence,
)

jnp.set_printoptions(precision=3, suppress=True)

# step size of gradient descent
GAMMA = 2e-1 * jnp.ones((11,))

if __name__ == "__main__":
    kinematics = SelectivePiecewiseConstantStrain(
        # specify the original length of each segment
        # the rod is composed of 2 segments of length 0.5 m each
        l0=jnp.array([0.5, 0.5]),
        # model twist and elongation to be constant across entire rod
        strain_selector_cs=jnp.array([False, False, True, False, False, True]),
        # model the bending and shear strains to be constant across each segment (i.e. piecewise constant)
        strain_selector_pcs=jnp.array([True, True, False, True, True, False])
    )

    # max value of point coordinate s
    s_max = jnp.sum(kinematics.l0)

    q = jnp.zeros(kinematics.configuration.shape)  # initialize configuration vector to zero
    q = q.at[0].set(20 / 180 * jnp.pi)  # set the twist angle at the base to 20 degrees
    q = q.at[1].set(jnp.pi)  # set the twist strain constant along the rod to pi rad / m
    q = q.at[2].set(0.1)  # set the elongation strain constant along the rod to 0.1 m / m
    q = q.at[3].set(90 / 180 * jnp.pi)  # set the x-bending strain constant along the 1st segment
    q = q.at[-3].set(90 / 180 * jnp.pi)  # set the y-bending strain constant along the 2nd segment

    points = jnp.linspace(start=0.0, stop=s_max, num=9)

    print("Plotting configuration:\n", q)
    T = kinematics.forward_kinematics(points, configuration=q)
    plot_rod_shape(T=T)

    # Run inverse kinematics to estimate the configuration of the rod
    q_init = jnp.zeros_like(q)  # initial guess for the configuration
    print("Running inverse kinematics...")
    q_hat, e_chi, q_its, e_chi_its = kinematics.inverse_kinematics(
        T,
        points,
        num_iterations=1000,
        state_init=q_init,
        translational_error_weight=1e0,
        rotational_error_weight=1e0,
        gamma=GAMMA,
    )
    print("Estimated configuration:\n", q_hat)
    e_quat, e_t = jmath.quat_pose_error_to_rmse(e_chi)
    print(f"RMSE errors: e_quat={e_quat}, e_t={e_t}")

    # use estimated configuration to compute transformation matrices to points
    T_hat = kinematics.forward_kinematics(points, configuration=q_hat)

    # plot the ground-truth and the estimated rod shape
    plot_points = jnp.linspace(start=0.0, stop=1.0, num=20)
    plot_T = kinematics.forward_kinematics(plot_points, q)
    plot_T_hat = kinematics.forward_kinematics(plot_points, q_hat)
    plot_rod_shape(T=plot_T, T_hat=plot_T_hat)

    plot_inverse_kinematics_convergence(q_its[:, 1:-1], e_chi_its)
