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


import torch
from dataclasses import MISSING

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .events import (
    cache_body_mass_scale,
    push_by_setting_velocity_with_recovery,
    randomize_action_noise_range,
    randomize_body_com,
    randomize_motion_ref_xyz,
    randomize_pd_scale,
    reset_robot_state_and_motion,
    update_curriculum,
    # 新增鲁棒性域随机化函数
    randomize_joint_armature,
    randomize_action_noise_event,
    randomize_action_delay_event,
    randomize_joint_encoder_noise_event,
    randomize_imu_noise_and_bias_event,
    randomize_observation_dropout_event,
    randomize_joint_failure_event,
    randomize_sensor_latency_spike_event,
    randomize_gravity_bias_event,
)


@configclass
class NeuralWBCEventCfg:
    reset_robot_and_motion = EventTerm(
        func=reset_robot_state_and_motion,
        mode="reset",
    )

    # -- internal states
    update_curriculum = EventTerm(
        func=update_curriculum,
        mode="reset",
        params={
            "penalty_level_down_threshold": 50,
            "penalty_level_up_threshold": 125,  # use 110 for hard rough terrain
            "penalty_level_degree": 0.00001,
            "min_penalty_scale": 0.0,
            "max_penalty_scale": 1.0,
            "num_compute_average_epl": 10000,
        },
    )


@configclass
class NeuralWBCPlayEventCfg(NeuralWBCEventCfg):
    # 播放模式下禁用所有新增的鲁棒性随机化事件
    randomize_joint_friction = None
    action_noise = None
    action_delay = None
    encoder_noise = None
    imu_noise = None
    observation_dropout = None
    joint_failure = None
    sensor_latency_spike = None
    slope_randomization = None

    def __post_init__(self):
        super().__post_init__()

        # Fix penalty scale at 1
        self.update_curriculum.params["min_penalty_scale"] = 1.0


@configclass
class NeuralWBCTrainEventCfg(NeuralWBCEventCfg):
    """
    Configuration for training events includes robot reset and some domain randomization.

    The physics parameters are only randomized during startup by default. If you would like to perform more
    randomization, they can be set to reset mode. However, this will significantly reduce the FPS.
    """

    # interval
    push_robot = EventTerm(
        func=push_by_setting_velocity_with_recovery,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )

    # startup
    # -- robot
    reset_robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    reset_robot_rigid_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_robot_base_com = EventTerm(
        func=randomize_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "distribution_params": (torch.tensor([-0.1, -0.1, -0.1]), torch.tensor([0.1, 0.1, 0.1])),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    # -- internal states
    cache_body_mass_scale = EventTerm(
        func=cache_body_mass_scale,
        mode="startup",
    )

    # reset
    # -- robot
    reset_joint_pd_gain = EventTerm(
        func=randomize_pd_scale,
        mode="reset",
        params={
            "distribution_params": (0.75, 1.25),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    reset_joint_action_noise_range = EventTerm(
        func=randomize_action_noise_range,
        mode="reset",
        params={
            "distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    reset_robot_and_motion = EventTerm(
        func=reset_robot_state_and_motion,
        mode="reset",
    )

    # -- motion reference
    reset_motion_ref_offset = EventTerm(
        func=randomize_motion_ref_xyz,
        mode="reset",
        params={
            "distribution_params": (torch.tensor([-0.02, -0.02, -0.1]), torch.tensor([0.02, 0.02, 0.1])),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    # ==================== 新增鲁棒性随机化（极低频率 corner case） ====================

    # 关节惯量随机化 - 模拟关节磨损和老化
    randomize_joint_friction = EventTerm(
        func=randomize_joint_armature,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.5, 2.0),
            "operation": "scale",
        },
    )

    # 动作噪声 - 模拟控制信号不完美（量化误差、通讯抖动）
    action_noise = EventTerm(
        func=randomize_action_noise_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "noise_std": 0.01,
            "noise_type": "gaussian",
        },
    )

    # 动作延迟 - 模拟通讯延迟和控制周期不对齐
    action_delay = EventTerm(
        func=randomize_action_delay_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_delay_steps": 5,
        },
    )

    # 关节编码器噪声 - 模拟编码器测量误差和零点偏移
    encoder_noise = EventTerm(
        func=randomize_joint_encoder_noise_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pos_noise_std": 0.005,
            "vel_noise_std": 0.01,
            "pos_bias_range": (-0.01, 0.01),
            "vel_bias_range": (-0.02, 0.02),
        },
    )

    # IMU噪声和漂移 - 模拟真实IMU的测量特性
    imu_noise = EventTerm(
        func=randomize_imu_noise_and_bias_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ang_vel_noise_std": 0.02,
            "lin_acc_noise_std": 0.05,
            "ang_vel_bias_range": (-0.1, 0.1),
            "lin_acc_bias_range": (-0.2, 0.2),
            "bias_drift_std": 0.01,
        },
    )

    # 观测丢包 - 模拟传感器偶发失效
    observation_dropout = EventTerm(
        func=randomize_observation_dropout_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "dropout_prob": 0.001,
            "dropout_mode": "hold",
        },
    )

    # 关节故障 - 模拟电机故障（极低概率）
    joint_failure = EventTerm(
        func=randomize_joint_failure_event,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "failure_prob": 0.0001,
            "failure_mode": "weak",
            "weak_factor": 0.5,
        },
    )

    # 传感器延迟尖峰 - 模拟偶发的通讯阻塞
    sensor_latency_spike = EventTerm(
        func=randomize_sensor_latency_spike_event,
        mode="interval",
        interval_range_s=(40.0, 60.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "spike_prob": 0.001,
            "max_latency_steps": 10,
        },
    )

    # 重力方向偏置 - 模拟基座倾斜/坡度
    slope_randomization = EventTerm(
        func=randomize_gravity_bias_event,
        mode="startup",
        params={
            "gravity_bias_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.05, 0.05),
            },
        },
    )
