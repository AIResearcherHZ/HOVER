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
import unittest

from mujoco_wrapper.mujoco_simulator import WBCMujoco
from mujoco_wrapper.mujoco_utils import get_entity_id, get_entity_name

from neural_wbc.data import get_data_path


class TestMujocoBodyStateTaksT1(unittest.TestCase):
    def setUp(self):
        self.num_instances = 1
        self.device = torch.device("cpu")
        self.robot = WBCMujoco(model_path=get_data_path("mujoco/models/Taks_T1/Taks_T1.xml"), device=self.device)

    def test_body_state_data(self):
        num_instances = self.num_instances
        num_controls = self.robot.model.nu
        # Confirm sizes are correct - Taks_T1 has 32 actuators
        self.assertEqual(num_controls, 32)
        self.assertEqual(self.robot.joint_positions.shape, torch.Size((num_instances, num_controls)))
        self.assertEqual(self.robot.joint_velocities.shape, torch.Size((num_instances, num_controls)))

        # Only consider the robot bodies.
        num_bodies = len(self.robot.body_names)
        self.assertEqual(self.robot.body_positions.shape, torch.Size((num_instances, num_bodies, 3)))  # [x,y,z]
        self.assertEqual(self.robot.body_rotations.shape, torch.Size((num_instances, num_bodies, 4)))  # [w,x,y,z]

        lin_vel, ang_vel = self.robot.body_velocities
        self.assertEqual(lin_vel.shape, torch.Size((num_instances, num_bodies, 3)))
        self.assertEqual(ang_vel.shape, torch.Size((num_instances, num_bodies, 3)))


class TestMujocoUtilsTaksT1(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.robot = WBCMujoco(model_path=get_data_path("mujoco/models/Taks_T1/Taks_T1.xml"), device=self.device)

        self.sample_taks_t1_body_ids = {
            "pelvis": 0,
            "left_hip_pitch_link": 1,
            "left_hip_roll_link": 2,
            "left_hip_yaw_link": 3,
            "left_knee_link": 4,
            "left_ankle_pitch_link": 5,
            "left_ankle_roll_link": 6,
        }

    def test_free_joint_check(self):
        self.assertEqual(self.robot.has_free_joint, True)

    def test_get_entity_name(self):
        # body
        test_body_id = 0
        output_body_name = get_entity_name(self.robot.model, "body", test_body_id + 1)
        self.assertEqual(output_body_name, "pelvis")

        # Missing element
        self.assertRaises(IndexError, lambda: get_entity_name(self.robot.model, "body", -1))

    def test_get_entity_id(self):
        # body
        test_body_name = "pelvis"
        expected_body_id = 1  # NOTE: We mirror the transformation in WBCMujoco here
        output_body_id = get_entity_id(self.robot.model, "body", test_body_name)
        self.assertEqual(output_body_id, expected_body_id)

        # Missing element
        self.assertEqual(get_entity_id(self.robot.model, "body", "random_name"), -1)
        self.assertEqual(get_entity_id(self.robot.model, "joint", "random_name"), -1)

    def test_joint_count(self):
        # Taks_T1 has 32 actuated joints
        self.assertEqual(self.robot.model.nu, 32)

    def test_body_count(self):
        # Taks_T1 has 33 bodies (excluding world)
        num_bodies = len(self.robot.body_names)
        self.assertEqual(num_bodies, 33)


if __name__ == "__main__":
    unittest.main()
