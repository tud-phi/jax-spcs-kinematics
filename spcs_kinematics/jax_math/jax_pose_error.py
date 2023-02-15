from jax import jit, vmap
import jax.numpy as jnp
from jax.numpy.linalg import norm
from typing import Tuple

from .jax_rotation import rotmat_to_quat, rotmat_to_euler_xyz
from .jax_quaternion import quaternion_orientation_error


@jit
def compute_pose_error_quaternion_representation(
    T1: jnp.ndarray, T2: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the pose error between two given poses in se(3)
    The return pose error consists of a rotation error in quaternion representation and a translation error:
        pose_error = (quat_error^T, t_error^T)^T
    where t_error is t_2 - t_1. Attention: this pose error needs to be used with the geometric Jacobian.
    :param: T_curr: current pose in se(3) (e.g. transformation matrix) of shape (4, 4)
    :param: T_goal: goal pose in se(3) (e.g. transformation matrix) of shape (4, 4)
    :param: eps: small number to avoid division by zero
    :return: pose_error: pose error in SE(3) consisting of (quat_error^T, t_error^T)^T of shape (7, )
    """
    quat1 = rotmat_to_quat(T1[0:3, 0:3], 1e1 * eps)
    quat2 = rotmat_to_quat(T2[0:3, 0:3], 1e1 * eps)

    # quat_{12} = quat_{1I}^T * quat_{I2} where I is the inertial frame
    quat_error = quaternion_orientation_error(quat1, quat2)

    # translation error
    t_error = T2[0:3, 3] - T1[0:3, 3]

    # construct pose error as (quat_error^T, t_error^T)^T
    pose_error = jnp.concatenate([quat_error, t_error], axis=0)
    return pose_error


# vectorized version of the compute_pose_error_quaternion_representation function
vcompute_pose_error_quaternion_representation = jit(
    vmap(compute_pose_error_quaternion_representation, in_axes=(2, 2, None), out_axes=1)
)


@jit
def compute_pose_error_euler_xyz_representation(
    T1: jnp.ndarray, T2: jnp.ndarray, eps: float
) -> jnp.ndarray:
    """
    Computes the pose error between two given poses in se(3)
    The return pose error consists of a rotation error in Euler XYZ representation and a translation error:
        pose_error = (euler_xyz_error^T, t_error^T)^T
    where t_error is t_2 - t_1. Attention: this pose error needs to be used with the geometric Jacobian.
    :param: T_curr: current pose in se(3) (e.g. transformation matrix) of shape (4, 4)
    :param: T_goal: goal pose in se(3) (e.g. transformation matrix) of shape (4, 4)
    :param: eps: small number to avoid division by zero
    :return: pose_error: pose error in SE(3) consisting of (quat_error^T, t_error^T)^T of shape (7, )
    """
    # relative rotation from T1 to T2
    # R_{12} = R_{1I}^T * R_{I2} where I is the inertial frame
    R12 = T1[0:3, 0:3].T @ T2[0:3, 0:3]
    # compute euler angles
    xyz_euler_angles = rotmat_to_euler_xyz(R12)

    # translation error
    t_error = T2[0:3, 3] - T1[0:3, 3]

    # construct pose error as (euler_xyz_error^T, t_error^T)^T
    pose_error = jnp.concatenate([xyz_euler_angles, t_error], axis=0)

    return pose_error


@jit
def quat_pose_error_to_rmse(e_chi: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the residuals from the pose error in SE(3) in quaternion representation.
    The residuals consist of a rotational and a translational component.
    :param: e_chi: pose error in SE(3) of shape (7, N) where N is the number of points. It consists of:
        e_chi = [e_quat_x, e_quat_y, e_quat_z, e_quat_w, e_t_x, e_t_y, e_t_z] where each of the entries are residuals
    :return: e_quat: rotational RMSE error of shape (, )
    :return: e_t: translational RMSE error of shape (, )
    """
    # compute residuals
    e_quat = jnp.sqrt(jnp.mean(norm(e_chi[0:3], axis=0) ** 2, axis=0))
    e_t = jnp.sqrt(jnp.mean(norm(e_chi[4:7], axis=0) ** 2, axis=0))
    return e_quat, e_t


@jit
def euler_angles_pose_error_to_rmse(
    e_chi: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the residuals from the pose error in SE(3) in euler representation.
    The residuals consist of a rotational and a translational component.
    :param: e_chi: pose error in SE(3) of shape (6, N) where N is the number of points. It consists of:
        e_chi = [e_r_x, e_r_y, e_r_z, e_t_x, e_t_y, e_t_z] where each of the entries are residuals
    :return: e_euler: rotational RMSE error of shape (, )
    :return: e_t: translational RMSE error of shape (, )
    """
    # compute residuals
    e_euler = jnp.sqrt(jnp.mean(e_chi[0:3] ** 2, axis=-1))
    e_t = jnp.sqrt(jnp.mean(norm(e_chi[3:6], axis=0) ** 2, axis=0))
    return e_euler, e_t
