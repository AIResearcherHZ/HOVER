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


from __future__ import annotations

import torch

from neural_wbc.core.modes import NeuralWBCModes
from neural_wbc.data import get_data_path

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from .events import NeuralWBCPlayEventCfg, NeuralWBCTrainEventCfg
from .neural_wbc_env_cfg import NeuralWBCEnvCfg
from .terrain import HARD_ROUGH_TERRAINS_CFG, flat_terrain

DISTILL_MASK_MODES_ALL = {
    "exbody": {
        "upper_body": ["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        "lower_body": ["root.*"],
    },
    "humanplus": {
        "upper_body": ["waist_.*_joint", ".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"],
        "lower_body": [".*_hip_.*_joint", ".*_knee_joint", ".*_ankle_.*_joint", "root.*"],
    },
    "h2o": {
        "upper_body": [
            ".*_shoulder_.*_link",
            ".*_elbow_link",
            ".*_wrist_.*_link",
            ".*_hand_link",
        ],
        "lower_body": [".*_ankle_roll_link"],
    },
    "omnih2o": {
        "upper_body": [".*_hand_link", ".*head_link"],
    },
    "twist2": {
        "upper_body": [".*_wrist_pitch_link", "neck_pitch_link"],
        "lower_body": [".*_ankle_roll_link"],
    },
}


# Note: You need to create a TAKS_T1_CFG similar to H1_CFG in isaaclab_assets
# For now, we define the robot configuration inline
@configclass
class NeuralWBCEnvCfgTaksT1(NeuralWBCEnvCfg):
    # General parameters:
    action_space = 32
    observation_space = 1313  # Adjusted for 32 DOF robot
    state_space = 1390

    # Distillation parameters:
    single_history_dim = 63
    observation_history_length = 25

    # Mask setup for an OH2O specialist policy as default:
    distill_mask_sparsity_randomization_enabled = False
    distill_mask_modes = {"twist2": DISTILL_MASK_MODES_ALL["twist2"]}

    # Robot geometry / actuation parameters:
    actuators = {
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch", ".*_hip_roll", ".*_hip_yaw", ".*_knee",
            ],
            effort_limit={
                ".*_hip_pitch": 120.0,
                ".*_hip_roll": 97.0,
                ".*_hip_yaw": 97.0,
                ".*_knee": 120.0,
            },
            velocity_limit={
                ".*_hip_pitch": 23.0,
                ".*_hip_roll": 23.0,
                ".*_hip_yaw": 23.0,
                ".*_knee": 14.0,
            },
            stiffness=0,
            damping=0,
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ankle_pitch", ".*_ankle_roll"],
            effort_limit=27.0,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_yaw", "waist_roll", "waist_pitch"],
            effort_limit=97.0,
            velocity_limit=23.0,
            stiffness=0,
            damping=0,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit=9.0,
            velocity_limit=9.0,
            stiffness=0,
            damping=0,
        ),
        "wrists": IdealPDActuatorCfg(
            joint_names_expr=[".*_wrist_roll", ".*_wrist_yaw", ".*_wrist_pitch"],
            effort_limit=3.0,
            velocity_limit=20.0,
            stiffness=0,
            damping=0,
        ),
        "neck": IdealPDActuatorCfg(
            joint_names_expr=["neck_yaw", "neck_roll", "neck_pitch"],
            effort_limit=0.8,
            velocity_limit=5.0,
            stiffness=0,
            damping=0,
        ),
    }

    # Note: You need to define TAKS_T1_CFG in isaaclab_assets or use a USD file
    # robot: ArticulationCfg = TAKS_T1_CFG.replace(prim_path="/World/envs/env_.*/Robot", actuators=actuators)

    body_names = [
        "pelvis",
        # Left leg
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        # Right leg
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        # Waist
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        # Left arm
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        # Right arm
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        # Neck
        "neck_yaw_link",
        "neck_roll_link",
        "neck_pitch_link",
    ]

    # Joint names by the order in the MJCF model (actuator order).
    joint_names = [
        "left_hip_pitch",
        "right_hip_pitch",
        "waist_yaw",
        "left_hip_roll",
        "right_hip_roll",
        "waist_roll",
        "left_hip_yaw",
        "right_hip_yaw",
        "waist_pitch",
        "left_knee",
        "right_knee",
        "left_shoulder_pitch",
        "neck_yaw",
        "right_shoulder_pitch",
        "left_ankle_pitch",
        "right_ankle_pitch",
        "left_shoulder_roll",
        "neck_roll",
        "right_shoulder_roll",
        "left_ankle_roll",
        "right_ankle_roll",
        "left_shoulder_yaw",
        "neck_pitch",
        "right_shoulder_yaw",
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_yaw",
        "right_wrist_yaw",
        "left_wrist_pitch",
        "right_wrist_pitch",
    ]

    # Lower and upper body joint ids in the MJCF model.
    lower_body_joint_ids = [0, 1, 3, 4, 6, 7, 9, 10, 14, 15, 19, 20]  # hips, knees, ankles
    upper_body_joint_ids = [2, 5, 8, 11, 12, 13, 16, 17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # waist, arms, neck

    base_name = "torso_link"
    root_id = body_names.index(base_name)

    feet_name = ".*_ankle_roll_link"

    extend_body_parent_names = ["left_wrist_pitch_link", "right_wrist_pitch_link", "neck_pitch_link"]
    extend_body_names = ["left_hand_link", "right_hand_link", "head_link"]
    extend_body_pos = torch.tensor([[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0.1]])

    # These are the bodies that are tracked by the teacher.
    tracked_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "left_knee_link",
        "left_ankle_pitch_link",
        "left_ankle_roll_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "right_knee_link",
        "right_ankle_pitch_link",
        "right_ankle_roll_link",
        "waist_yaw_link",
        "waist_roll_link",
        "torso_link",
        "left_shoulder_pitch_link",
        "left_shoulder_roll_link",
        "left_shoulder_yaw_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "left_wrist_yaw_link",
        "left_wrist_pitch_link",
        "right_shoulder_pitch_link",
        "right_shoulder_roll_link",
        "right_shoulder_yaw_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "right_wrist_yaw_link",
        "right_wrist_pitch_link",
        "neck_yaw_link",
        "neck_roll_link",
        "neck_pitch_link",
        "left_hand_link",
        "right_hand_link",
        "head_link",
    ]

    # control parameters
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

    mass_randomized_body_names = [
        "pelvis",
        "left_hip_pitch_link",
        "left_hip_roll_link",
        "left_hip_yaw_link",
        "right_hip_pitch_link",
        "right_hip_roll_link",
        "right_hip_yaw_link",
        "torso_link",
    ]

    undesired_contact_body_names = [
        "pelvis",
        ".*_yaw_link",
        ".*_roll_link",
        ".*_pitch_link",
        ".*_knee_link",
    ]

    # Add a height scanner to the torso to detect the height of the terrain mesh
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    def __post_init__(self):
        super().__post_init__()

        self.reference_motion_manager.motion_path = get_data_path("motions/stable_punch.pkl")
        self.reference_motion_manager.skeleton_path = get_data_path("motion_lib/Taks_T1/Taks_T1.xml")

        if self.terrain.terrain_generator == HARD_ROUGH_TERRAINS_CFG:
            self.events.update_curriculum.params["penalty_level_up_threshold"] = 125

        if self.mode == NeuralWBCModes.TRAIN:
            self.episode_length_s = 20.0
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
        elif self.mode == NeuralWBCModes.DISTILL:
            self.max_ref_motion_dist = 0.5
            self.events = NeuralWBCTrainEventCfg()
            self.events.reset_robot_rigid_body_mass.params["asset_cfg"].body_names = self.mass_randomized_body_names
            self.events.reset_robot_base_com.params["asset_cfg"].body_names = "torso_link"
            self.add_policy_obs_noise = False
            self.reset_mask = True
            num_regions = len(self.distill_mask_modes)
            if num_regions == 1:
                region_modes = list(self.distill_mask_modes.values())[0]
                if len(region_modes) == 1:
                    self.reset_mask = False
        elif self.mode == NeuralWBCModes.TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        elif self.mode == NeuralWBCModes.DISTILL_TEST:
            self.terrain = flat_terrain
            self.events = NeuralWBCPlayEventCfg()
            self.distill_teleop_selected_keypoints_names = []
            self.ctrl_delay_step_range = (2, 2)
            self.max_ref_motion_dist = 0.5
            self.default_rfi_lim = 0.0
            self.add_policy_obs_noise = False
            self.resample_motions = False
            self.distill_mask_sparsity_randomization_enabled = False
            self.distill_mask_modes = {"omnih2o": DISTILL_MASK_MODES_ALL["omnih2o"]}
        else:
            raise ValueError(f"Unsupported mode {self.mode}")
