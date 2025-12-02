# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

from neural_wbc.core import ReferenceMotionState, math_utils

import isaaclab.envs.mdp as mdp
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from ..neural_env import NeuralWBCEnv


def randomize_body_com(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the com of the bodies by adding, scaling or setting random values.

    This function allows randomizing the center of mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales or sets the values into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms()

    if not hasattr(env, "default_coms"):
        # Randomize robot base com
        env.default_coms = coms.clone()
        env.base_com_bias = torch.zeros((env.num_envs, 3), dtype=torch.float, device=coms.device)

    # apply randomization on default values
    coms[env_ids[:, None], body_ids] = env.default_coms[env_ids[:, None], body_ids].clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (distribution_params[0].to(coms.device), distribution_params[1].to(coms.device))

    env.base_com_bias[env_ids, :] = dist_fn(
        *distribution_params, (env_ids.shape[0], env.base_com_bias.shape[1]), device=coms.device
    )

    # sample from the given range
    if operation == "add":
        coms[env_ids[:, None], body_ids, :3] += env.base_com_bias[env_ids[:, None], :]
    elif operation == "abs":
        coms[env_ids[:, None], body_ids, :3] = env.base_com_bias[env_ids[:, None], :]
    elif operation == "scale":
        coms[env_ids[:, None], body_ids, :3] *= env.base_com_bias[env_ids[:, None], :]
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_pd_scale(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the scale of the pd gains by adding, scaling or setting random values.

    This function allows randomizing the scale of the pd gain. The function samples random values from the
    given distribution parameters and adds, or sets the values into the simulation based on the operation.

    """
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    kp_scale = env.default_kp_scale.clone()
    kd_scale = env.default_kd_scale.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        kp_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "abs":
        kp_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] = dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    elif operation == "scale":
        kp_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kp_scale.shape[1]), device=kp_scale.device
        )
        kd_scale[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], kd_scale.shape[1]), device=kd_scale.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.kp_scale[env_ids, :] = kp_scale[env_ids, :]
    env.kd_scale[env_ids, :] = kd_scale[env_ids, :]


def randomize_action_noise_range(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the sample range of the added action noise by adding, scaling or setting random values.

    This function allows randomizing the scale of the sample range of the added action noise. The function
    samples random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    rfi_lim = env.default_rfi_lim.clone()

    dist_fn = resolve_dist_fn(distribution)

    # sample from the given range
    if operation == "add":
        rfi_lim[env_ids, :] += dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    elif operation == "abs":
        rfi_lim[env_ids, :] = dist_fn(*distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device)
    elif operation == "scale":
        rfi_lim[env_ids, :] *= dist_fn(
            *distribution_params, (env_ids.shape[0], rfi_lim.shape[1]), device=rfi_lim.device
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.rfi_lim[env_ids, :] = rfi_lim[env_ids, :]


def randomize_motion_ref_xyz(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    distribution_params: tuple[float, float] | tuple[torch.Tensor, torch.Tensor],
    operation: Literal["add", "abs", "scale"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the motion reference x,y,z offset by adding, scaling or setting random values.

    This function allows randomizing the motion reference x,y,z offset. The function samples
    random values from the given distribution parameters and adds, scales or sets the values into the
    simulation based on the operation.

    """

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # apply randomization on default values
    ref_episodic_offset = env.default_ref_episodic_offset.clone()

    dist_fn = resolve_dist_fn(distribution)

    if isinstance(distribution_params[0], torch.Tensor):
        distribution_params = (
            distribution_params[0].to(ref_episodic_offset.device),
            distribution_params[1].to(ref_episodic_offset.device),
        )

    # sample from the given range
    if operation == "add":
        ref_episodic_offset[env_ids, :] += dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    elif operation == "abs":
        ref_episodic_offset[env_ids, :] = dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    elif operation == "scale":
        ref_episodic_offset[env_ids, :] *= dist_fn(
            *distribution_params,
            (env_ids.shape[0], ref_episodic_offset.shape[1]),
            device=ref_episodic_offset.device,
        )
    else:
        raise ValueError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'abs' or 'scale'."
        )
    # set the mass into the physics simulation
    env.ref_episodic_offset[env_ids, :] = ref_episodic_offset[env_ids, :]


def push_by_setting_velocity_with_recovery(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w[:] = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)
    # give pushed robot time to recover
    env.recovery_counters[env_ids] = env.cfg.recovery_count


def reset_robot_state_and_motion(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot and reference motion.

    The full reset percess is:
    1. Reset the robot root state to the origin position of its terrain.
    2. Reset motion reference to random time step.
    3. Moving the motion reference trajectory to the current robot position.
    4. Reset the robot joint to reference motion's joint states
    5. Reset the robot root state to the reference motion's root state.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()
    root_states[:, :3] += env._terrain.env_origins[env_ids]
    state = root_states[:, :7]
    asset.write_root_pose_to_sim(state, env_ids=env_ids)

    mdp.reset_joints_by_scale(env, env_ids, (1.0, 1.0), (0.0, 0.0), asset_cfg)

    # Sample new commands
    env._ref_motion_mgr.reset_motion_start_times(env_ids=env_ids, sample=env.cfg.mode.is_training_mode())

    # Record new start locations on terrain
    env._start_positions_on_terrain[env_ids, ...] = root_states[:, :3]

    ref_motion_state: ReferenceMotionState = env._ref_motion_mgr.get_state_from_motion_lib_cache(
        episode_length_buf=0,
        offset=env._start_positions_on_terrain,
        terrain_heights=env.get_terrain_heights(),
    )

    env._ref_motion_visualizer.visualize(ref_motion_state)

    joint_pos = ref_motion_state.joint_pos[env_ids]
    joint_vel = ref_motion_state.joint_vel[env_ids]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env._joint_ids, env_ids=env_ids)

    root_states = asset.data.default_root_state[env_ids].clone()

    root_states[:, :3] = ref_motion_state.root_pos[env_ids]
    root_states[:, 2] += 0.04  # in case under the terrain

    root_states[:, 3:7] = ref_motion_state.root_rot[env_ids]
    root_states[:, 7:10] = ref_motion_state.root_lin_vel[env_ids]
    root_states[:, 10:13] = ref_motion_state.root_ang_vel[env_ids]

    state = root_states[:, :7]
    velocities = root_states[:, 7:13]
    asset.write_root_pose_to_sim(state, env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def update_curriculum(
    env: NeuralWBCEnv,
    env_ids: torch.Tensor | None,
    penalty_level_down_threshold: float,
    penalty_level_up_threshold: float,
    penalty_level_degree: float,
    min_penalty_scale: float,
    max_penalty_scale: float,
    num_compute_average_epl: float,
):
    """
    Update average episode length and in turn penalty curriculum.

    This function is rewritten from update_average_episode_length of legged_gym.

    When the policy is not able to track the motions, we reduce the penalty to help it explore more actions. When the
    policy is able to track the motions, we increase the penalty to smooth the actions and reduce the maximum action
    it uses.
    """
    N = env.num_envs if env_ids is None else len(env_ids)
    current_average_episode_length = torch.mean(env.episode_length_buf[env_ids], dtype=torch.float)
    env.average_episode_length = env.average_episode_length * (
        1 - N / num_compute_average_epl
    ) + current_average_episode_length * (N / num_compute_average_epl)

    if env.average_episode_length < penalty_level_down_threshold:
        env.penalty_scale *= 1 - penalty_level_degree
    elif env.average_episode_length > penalty_level_up_threshold:
        env.penalty_scale *= 1 + penalty_level_degree
    env.penalty_scale = np.clip(env.penalty_scale, min_penalty_scale, max_penalty_scale)


def cache_body_mass_scale(
    env: NeuralWBCEnv, env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses()
    scale = (masses / asset.data.default_mass).to(env.device)

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if not hasattr(env, "body_mass_scale"):
        # Cache body mass randomization scale for privileged observation.
        env.mass_randomized_body_ids, _ = asset.find_bodies(env.cfg.mass_randomized_body_names, preserve_order=True)
        env.body_mass_scale = torch.ones(
            env.num_envs, len(env.mass_randomized_body_ids), dtype=torch.float, device=env.device
        )

    env.body_mass_scale[env_ids, :] *= scale[env_ids[:, None], env.mass_randomized_body_ids]


def resolve_dist_fn(
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    dist_fn = math_utils.sample_uniform

    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise ValueError(f"Unrecognized distribution {distribution}")

    return dist_fn


# ==================== 新增鲁棒性域随机化函数 ====================


def randomize_joint_armature(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "scale",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """随机化关节惯量（armature），模拟关节磨损和老化。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        armature_distribution_params: 惯量分布参数 (min, max)
        operation: 操作类型 ("add", "scale", "abs")
        distribution: 分布类型
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()
    
    # 解析关节索引
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device="cpu")
    
    # 获取当前惯量 (使用内部属性 _dof_armatures)
    armatures = asset.root_physx_view._dof_armatures.clone()
    
    # 初始化默认惯量
    if not hasattr(env, "default_armatures"):
        env.default_armatures = armatures.clone()
    
    # 基于默认值进行随机化
    if isinstance(joint_ids, torch.Tensor):
        armatures[env_ids[:, None], joint_ids] = \
            env.default_armatures[env_ids[:, None], joint_ids].clone()
    else:
        armatures[env_ids, joint_ids] = \
            env.default_armatures[env_ids, joint_ids].clone()
    
    dist_fn = resolve_dist_fn(distribution)
    
    if isinstance(joint_ids, slice):
        num_joints = armatures.shape[1]
    else:
        num_joints = len(joint_ids)
    
    rand_samples = dist_fn(
        *armature_distribution_params,
        (len(env_ids), num_joints),
        device=armatures.device
    )
    
    if isinstance(joint_ids, torch.Tensor):
        if operation == "add":
            armatures[env_ids[:, None], joint_ids] += rand_samples
        elif operation == "scale":
            armatures[env_ids[:, None], joint_ids] *= rand_samples
        elif operation == "abs":
            armatures[env_ids[:, None], joint_ids] = rand_samples
    else:
        if operation == "add":
            armatures[env_ids, joint_ids] += rand_samples
        elif operation == "scale":
            armatures[env_ids, joint_ids] *= rand_samples
        elif operation == "abs":
            armatures[env_ids, joint_ids] = rand_samples
    
    # 使用 set_dof_armatures 设置惯量
    asset.root_physx_view.set_dof_armatures(armatures, env_ids)


def randomize_action_noise_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    noise_std: float = 0.01,
    noise_type: Literal["gaussian", "uniform"] = "gaussian",
):
    """随机化动作噪声，模拟控制信号不完美（量化误差、通讯抖动）。
    
    此函数设置环境中的动作噪声参数，实际噪声在动作应用时添加。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        noise_std: 噪声标准差
        noise_type: 噪声类型 ("gaussian" 或 "uniform")
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化动作噪声缓冲区
    if not hasattr(env, "action_noise_std"):
        env.action_noise_std = torch.zeros(env.num_envs, device=env.device)
        env.action_noise_type = "gaussian"
    
    # 随机采样噪声标准差
    env.action_noise_std[env_ids] = noise_std * torch.rand(len(env_ids), device=env.device)
    env.action_noise_type = noise_type


def randomize_action_delay_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_delay_steps: int = 5,
):
    """随机化动作延迟，模拟通讯延迟和控制周期不对齐。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        max_delay_steps: 最大延迟步数
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化动作延迟缓冲区
    if not hasattr(env, "action_delay_steps"):
        env.action_delay_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.action_history = None  # 将在首次使用时初始化
    
    # 随机采样延迟步数
    env.action_delay_steps[env_ids] = torch.randint(0, max_delay_steps + 1, (len(env_ids),), device=env.device)


def randomize_joint_encoder_noise_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_noise_std: float = 0.005,
    vel_noise_std: float = 0.01,
    pos_bias_range: tuple[float, float] = (-0.01, 0.01),
    vel_bias_range: tuple[float, float] = (-0.02, 0.02),
):
    """随机化关节编码器噪声，模拟编码器测量误差和零点偏移。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        pos_noise_std: 位置噪声标准差 (rad)
        vel_noise_std: 速度噪声标准差 (rad/s)
        pos_bias_range: 位置偏置范围 (rad)
        vel_bias_range: 速度偏置范围 (rad/s)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化编码器噪声缓冲区
    if not hasattr(env, "encoder_pos_noise_std"):
        env.encoder_pos_noise_std = torch.zeros(env.num_envs, num_joints, device=env.device)
        env.encoder_vel_noise_std = torch.zeros(env.num_envs, num_joints, device=env.device)
        env.encoder_pos_bias = torch.zeros(env.num_envs, num_joints, device=env.device)
        env.encoder_vel_bias = torch.zeros(env.num_envs, num_joints, device=env.device)
    
    # 设置噪声参数
    env.encoder_pos_noise_std[env_ids] = pos_noise_std
    env.encoder_vel_noise_std[env_ids] = vel_noise_std
    
    # 随机采样偏置
    env.encoder_pos_bias[env_ids] = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*pos_bias_range)
    env.encoder_vel_bias[env_ids] = torch.empty(len(env_ids), num_joints, device=env.device).uniform_(*vel_bias_range)


def randomize_imu_noise_and_bias_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ang_vel_noise_std: float = 0.02,
    lin_acc_noise_std: float = 0.05,
    ang_vel_bias_range: tuple[float, float] = (-0.1, 0.1),
    lin_acc_bias_range: tuple[float, float] = (-0.2, 0.2),
    bias_drift_std: float = 0.01,
):
    """随机化IMU噪声和漂移，模拟真实IMU的测量特性。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        ang_vel_noise_std: 角速度噪声标准差 (rad/s)
        lin_acc_noise_std: 线加速度噪声标准差 (m/s^2)
        ang_vel_bias_range: 角速度偏置范围
        lin_acc_bias_range: 线加速度偏置范围
        bias_drift_std: 偏置漂移标准差
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化IMU噪声缓冲区
    if not hasattr(env, "imu_ang_vel_noise_std"):
        env.imu_ang_vel_noise_std = torch.zeros(env.num_envs, device=env.device)
        env.imu_lin_acc_noise_std = torch.zeros(env.num_envs, device=env.device)
        env.imu_ang_vel_bias = torch.zeros(env.num_envs, 3, device=env.device)
        env.imu_lin_acc_bias = torch.zeros(env.num_envs, 3, device=env.device)
        env.imu_bias_drift_std = torch.zeros(env.num_envs, device=env.device)
    
    # 设置噪声参数
    env.imu_ang_vel_noise_std[env_ids] = ang_vel_noise_std
    env.imu_lin_acc_noise_std[env_ids] = lin_acc_noise_std
    env.imu_bias_drift_std[env_ids] = bias_drift_std
    
    # 随机采样偏置
    env.imu_ang_vel_bias[env_ids] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*ang_vel_bias_range)
    env.imu_lin_acc_bias[env_ids] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*lin_acc_bias_range)


def randomize_observation_dropout_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    dropout_prob: float = 0.001,
    dropout_mode: Literal["zero", "hold"] = "hold",
):
    """随机化观测丢包，模拟传感器偶发失效。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        dropout_prob: 每个维度丢包概率
        dropout_mode: 丢包时的处理模式 ("zero" 置零, "hold" 保持上一帧值)
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化观测丢包缓冲区
    if not hasattr(env, "obs_dropout_prob"):
        env.obs_dropout_prob = torch.zeros(env.num_envs, device=env.device)
        env.obs_dropout_mode = "hold"
        env.prev_observations = None  # 将在首次使用时初始化
    
    env.obs_dropout_prob[env_ids] = dropout_prob
    env.obs_dropout_mode = dropout_mode


def randomize_joint_failure_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    failure_prob: float = 0.0001,
    failure_mode: Literal["lock", "weak", "dead"] = "weak",
    weak_factor: float = 0.5,
):
    """随机化关节故障，模拟电机故障（极低概率）。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        failure_prob: 每个关节失效概率
        failure_mode: 故障模式 ("lock" 锁定, "weak" 弱化, "dead" 完全失效)
        weak_factor: 弱化模式下的扭矩衰减因子
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化关节故障缓冲区
    if not hasattr(env, "joint_failure_mask"):
        env.joint_failure_mask = torch.zeros(env.num_envs, num_joints, dtype=torch.bool, device=env.device)
        env.joint_failure_mode = "weak"
        env.joint_weak_factor = torch.ones(env.num_envs, num_joints, device=env.device)
    
    # 随机采样故障关节
    failure_mask = torch.rand(len(env_ids), num_joints, device=env.device) < failure_prob
    env.joint_failure_mask[env_ids] = failure_mask
    env.joint_failure_mode = failure_mode
    
    # 设置弱化因子
    env.joint_weak_factor[env_ids] = torch.where(
        failure_mask,
        torch.full((len(env_ids), num_joints), weak_factor, device=env.device),
        torch.ones(len(env_ids), num_joints, device=env.device)
    )


def randomize_sensor_latency_spike_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    spike_prob: float = 0.001,
    max_latency_steps: int = 10,
):
    """随机化传感器延迟尖峰，模拟偶发的通讯阻塞。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        asset_cfg: 资产配置
        spike_prob: 延迟尖峰发生概率
        max_latency_steps: 最大延迟步数
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化传感器延迟缓冲区
    if not hasattr(env, "sensor_latency_spike_prob"):
        env.sensor_latency_spike_prob = torch.zeros(env.num_envs, device=env.device)
        env.sensor_max_latency_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env.sensor_current_latency = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    
    env.sensor_latency_spike_prob[env_ids] = spike_prob
    env.sensor_max_latency_steps[env_ids] = max_latency_steps


def randomize_gravity_bias_event(
    env: "NeuralWBCEnv",
    env_ids: torch.Tensor | None,
    gravity_bias_range: dict[str, tuple[float, float]],
):
    """随机化重力方向偏置，模拟基座倾斜/坡度。
    
    注意：此函数通过在观测中添加重力偏置来模拟坡度效果，
    而不是真正改变物理引擎中的重力。
    
    Args:
        env: 环境实例
        env_ids: 需要随机化的环境ID
        gravity_bias_range: 重力偏置范围字典，包含 "x", "y", "z" 键
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 初始化重力偏置缓冲区
    if not hasattr(env, "gravity_bias"):
        env.gravity_bias = torch.zeros(env.num_envs, 3, device=env.device)
    
    # 按 xyz 范围采样重力偏置
    for i, key in enumerate(["x", "y", "z"]):
        if key in gravity_bias_range:
            low, high = gravity_bias_range[key]
            env.gravity_bias[env_ids, i] = torch.empty(len(env_ids), device=env.device).uniform_(low, high)
