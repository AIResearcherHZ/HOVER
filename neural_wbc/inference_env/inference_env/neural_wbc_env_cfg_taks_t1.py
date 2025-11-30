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
from dataclasses import dataclass

from inference_env.neural_wbc_env_cfg import NeuralWBCEnvCfg

from neural_wbc.core.mask import calculate_mask_length
from neural_wbc.data import get_data_path


@dataclass
class NeuralWBCEnvCfgTaksT1(NeuralWBCEnvCfg):
    decimation = 4
    dt = 0.005
    max_episode_length_s = 3600
    action_scale = 0.25
    ctrl_delay_step_range = [2, 2]
    default_rfi_lim = 0
    robot = "mujoco_robot"

    # Extended bodies for tracking (hands and head)
    extend_body_parent_names = ["left_wrist_pitch_link", "right_wrist_pitch_link", "neck_pitch_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0.1]])

    tracked_body_names = [
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # Distillation parameters:
    # Taks_T1 has 33 bodies and 32 joints
    single_history_dim = 63
    observation_history_length = 25
    num_bodies = 33  # pelvis + 12 leg links + 3 waist links + 14 arm links + 3 neck links
    num_joints = 32
    mask_length = calculate_mask_length(
        num_bodies=num_bodies + len(extend_body_parent_names),
        num_joints=num_joints,
    )

    control_type = "Pos"
    robot_actuation_type = "Torque"  # Pos or Torque

    # Control parameters - based on actuator force ranges from Taks_T1.xml
    stiffness = {
        # Legs
        "left_hip_pitch": 200.0,
        "left_hip_roll": 150.0,
        "left_hip_yaw": 150.0,
        "left_knee": 200.0,
        "left_ankle_pitch": 40.0,
        "left_ankle_roll": 40.0,
        "right_hip_pitch": 200.0,
        "right_hip_roll": 150.0,
        "right_hip_yaw": 150.0,
        "right_knee": 200.0,
        "right_ankle_pitch": 40.0,
        "right_ankle_roll": 40.0,
        # Waist
        "waist_yaw": 150.0,
        "waist_roll": 150.0,
        "waist_pitch": 150.0,
        # Arms
        "left_shoulder_pitch": 40.0,
        "left_shoulder_roll": 40.0,
        "left_shoulder_yaw": 40.0,
        "left_elbow": 40.0,
        "left_wrist_roll": 20.0,
        "left_wrist_yaw": 20.0,
        "left_wrist_pitch": 20.0,
        "right_shoulder_pitch": 40.0,
        "right_shoulder_roll": 40.0,
        "right_shoulder_yaw": 40.0,
        "right_elbow": 40.0,
        "right_wrist_roll": 20.0,
        "right_wrist_yaw": 20.0,
        "right_wrist_pitch": 20.0,
        # Neck
        "neck_yaw": 10.0,
        "neck_roll": 10.0,
        "neck_pitch": 10.0,
    }

    damping = {
        # Legs
        "left_hip_pitch": 5.0,
        "left_hip_roll": 5.0,
        "left_hip_yaw": 5.0,
        "left_knee": 5.0,
        "left_ankle_pitch": 4.0,
        "left_ankle_roll": 4.0,
        "right_hip_pitch": 5.0,
        "right_hip_roll": 5.0,
        "right_hip_yaw": 5.0,
        "right_knee": 5.0,
        "right_ankle_pitch": 4.0,
        "right_ankle_roll": 4.0,
        # Waist
        "waist_yaw": 5.0,
        "waist_roll": 5.0,
        "waist_pitch": 5.0,
        # Arms
        "left_shoulder_pitch": 10.0,
        "left_shoulder_roll": 10.0,
        "left_shoulder_yaw": 10.0,
        "left_elbow": 10.0,
        "left_wrist_roll": 5.0,
        "left_wrist_yaw": 5.0,
        "left_wrist_pitch": 5.0,
        "right_shoulder_pitch": 10.0,
        "right_shoulder_roll": 10.0,
        "right_shoulder_yaw": 10.0,
        "right_elbow": 10.0,
        "right_wrist_roll": 5.0,
        "right_wrist_yaw": 5.0,
        "right_wrist_pitch": 5.0,
        # Neck
        "neck_yaw": 2.0,
        "neck_roll": 2.0,
        "neck_pitch": 2.0,
    }

    effort_limit = {
        # Legs - from actuatorfrcrange in Taks_T1.xml
        "left_hip_pitch": 120.0,
        "left_hip_roll": 97.0,
        "left_hip_yaw": 97.0,
        "left_knee": 120.0,
        "left_ankle_pitch": 27.0,
        "left_ankle_roll": 27.0,
        "right_hip_pitch": 120.0,
        "right_hip_roll": 97.0,
        "right_hip_yaw": 97.0,
        "right_knee": 120.0,
        "right_ankle_pitch": 27.0,
        "right_ankle_roll": 27.0,
        # Waist
        "waist_yaw": 97.0,
        "waist_roll": 97.0,
        "waist_pitch": 97.0,
        # Arms
        "left_shoulder_pitch": 27.0,
        "left_shoulder_roll": 27.0,
        "left_shoulder_yaw": 27.0,
        "left_elbow": 27.0,
        "left_wrist_roll": 7.0,
        "left_wrist_yaw": 7.0,
        "left_wrist_pitch": 7.0,
        "right_shoulder_pitch": 27.0,
        "right_shoulder_roll": 27.0,
        "right_shoulder_yaw": 27.0,
        "right_elbow": 27.0,
        "right_wrist_roll": 7.0,
        "right_wrist_yaw": 7.0,
        "right_wrist_pitch": 7.0,
        # Neck
        "neck_yaw": 3.0,
        "neck_roll": 3.0,
        "neck_pitch": 3.0,
    }

    position_limit = {
        # Legs - from joint range in Taks_T1.xml
        "left_hip_pitch": [-2.5307, 2.8798],
        "left_hip_roll": [-0.5236, 2.9671],
        "left_hip_yaw": [-2.7576, 2.7576],
        "left_knee": [-0.087267, 2.8798],
        "left_ankle_pitch": [-0.87267, 0.5236],
        "left_ankle_roll": [-0.2618, 0.2618],
        "right_hip_pitch": [-2.5307, 2.8798],
        "right_hip_roll": [-2.9671, 0.5236],
        "right_hip_yaw": [-2.7576, 2.7576],
        "right_knee": [-0.087267, 2.8798],
        "right_ankle_pitch": [-0.87267, 0.5236],
        "right_ankle_roll": [-0.2618, 0.2618],
        # Waist
        "waist_yaw": [-2.618, 2.618],
        "waist_roll": [-0.52, 0.52],
        "waist_pitch": [-0.52, 0.52],
        # Arms
        "left_shoulder_pitch": [-3.0, 2.0],
        "left_shoulder_roll": [-0.2, 2.2515],
        "left_shoulder_yaw": [-2.58, 2.58],
        "left_elbow": [-0.7, 1.57],
        "left_wrist_roll": [-2.67, 2.67],
        "left_wrist_yaw": [-0.9, 0.9],
        "left_wrist_pitch": [-0.9, 0.9],
        "right_shoulder_pitch": [-2.0, 2.0],
        "right_shoulder_roll": [-2.2515, 0.2],
        "right_shoulder_yaw": [-2.58, 2.58],
        "right_elbow": [-0.7, 1.57],
        "right_wrist_roll": [-2.67, 2.67],
        "right_wrist_yaw": [-0.9, 0.9],
        "right_wrist_pitch": [-0.9, 0.9],
        # Neck
        "neck_yaw": [-1.57, 1.57],
        "neck_roll": [-0.873, 0.873],
        "neck_pitch": [-0.873, 0.873],
    }

    robot_init_state = {
        "base_pos": [0.0, 0.0, 0.75],
        "base_quat": [1.0, 0.0, 0.0, 0.0],
        "joint_pos": {
            # Legs - standing pose
            "left_hip_pitch": 0.0,
            "left_hip_roll": 0.0,
            "left_hip_yaw": 0.0,
            "left_knee": 0.0,
            "left_ankle_pitch": 0.0,
            "left_ankle_roll": 0.0,
            "right_hip_pitch": 0.0,
            "right_hip_roll": 0.0,
            "right_hip_yaw": 0.0,
            "right_knee": 0.0,
            "right_ankle_pitch": 0.0,
            "right_ankle_roll": 0.0,
            # Waist
            "waist_yaw": 0.0,
            "waist_roll": 0.0,
            "waist_pitch": 0.0,
            # Arms - relaxed pose
            "left_shoulder_pitch": 0.0,
            "left_shoulder_roll": 0.3,
            "left_shoulder_yaw": 0.0,
            "left_elbow": 0.5,
            "left_wrist_roll": 0.0,
            "left_wrist_yaw": 0.0,
            "left_wrist_pitch": 0.0,
            "right_shoulder_pitch": 0.0,
            "right_shoulder_roll": -0.3,
            "right_shoulder_yaw": 0.0,
            "right_elbow": 0.5,
            "right_wrist_roll": 0.0,
            "right_wrist_yaw": 0.0,
            "right_wrist_pitch": 0.0,
            # Neck
            "neck_yaw": 0.0,
            "neck_roll": 0.0,
            "neck_pitch": 0.0,
        },
        "joint_vel": {},
    }

    # Lower and upper body joint ids in the MJCF model (based on actuator order)
    # Actuator order: left_hip_pitch(0), right_hip_pitch(1), waist_yaw(2), left_hip_roll(3), right_hip_roll(4),
    # waist_roll(5), left_hip_yaw(6), right_hip_yaw(7), waist_pitch(8), left_knee(9), right_knee(10),
    # left_shoulder_pitch(11), neck_yaw(12), right_shoulder_pitch(13), left_ankle_pitch(14), right_ankle_pitch(15),
    # left_shoulder_roll(16), neck_roll(17), right_shoulder_roll(18), left_ankle_roll(19), right_ankle_roll(20),
    # left_shoulder_yaw(21), neck_pitch(22), right_shoulder_yaw(23), left_elbow(24), right_elbow(25),
    # left_wrist_roll(26), right_wrist_roll(27), left_wrist_yaw(28), right_wrist_yaw(29),
    # left_wrist_pitch(30), right_wrist_pitch(31)
    lower_body_joint_ids = [0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 19, 20]  # hips, knees, ankles
    upper_body_joint_ids = [2, 5, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # waist, arms, neck

    def __post_init__(self):
        self.reference_motion_cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        self.reference_motion_cfg.skeleton_path = get_data_path("motion_lib/Taks_T1/Taks_T1.xml")
