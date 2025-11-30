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

from neural_wbc.data import get_data_path


class TestMujocoSimulatorTaksT1(unittest.TestCase):
    """Test class for MuJoCo simulator with Taks_T1 robot."""

    def setUp(self):
        self.num_instances = 2
        self.device = torch.device("cpu")
        self.model_path = get_data_path("mujoco/models/Taks_T1/scene_Taks_T1.xml")
        self.robot = WBCMujoco(
            model_path=self.model_path,
            num_instances=self.num_instances,
            device=self.device,
        )

    def test_initialization(self):
        """Test that the simulator initializes correctly."""
        self.assertIsNotNone(self.robot)
        self.assertEqual(self.robot.num_instances, self.num_instances)

    def test_joint_count(self):
        """Test that Taks_T1 has 32 actuated joints."""
        self.assertEqual(self.robot.model.nu, 32)

    def test_body_count(self):
        """Test that Taks_T1 has 33 bodies."""
        num_bodies = len(self.robot.body_names)
        self.assertEqual(num_bodies, 33)

    def test_step(self):
        """Test stepping the simulation."""
        # Get initial state
        initial_pos = self.robot.joint_positions.clone()

        # Step with zero action
        action = torch.zeros(self.num_instances, 32)
        self.robot.step(action)

        # State should change after stepping
        self.assertIsNotNone(self.robot.joint_positions)

    def test_reset(self):
        """Test resetting the simulation."""
        # Create initial qpos and qvel
        qpos = torch.zeros(self.num_instances, 7 + 32)  # 7 for free joint + 32 joints
        qpos[:, 2] = 0.75  # Set initial height
        qpos[:, 3] = 1.0  # Set quaternion w component
        qvel = torch.zeros(self.num_instances, 6 + 32)  # 6 for free joint vel + 32 joint vels

        self.robot.reset(qpos=qpos, qvel=qvel)

        # Check that reset was successful
        self.assertIsNotNone(self.robot.joint_positions)

    def test_body_positions(self):
        """Test getting body positions."""
        body_positions = self.robot.body_positions
        num_bodies = len(self.robot.body_names)
        self.assertEqual(body_positions.shape, torch.Size((self.num_instances, num_bodies, 3)))

    def test_body_rotations(self):
        """Test getting body rotations."""
        body_rotations = self.robot.body_rotations
        num_bodies = len(self.robot.body_names)
        self.assertEqual(body_rotations.shape, torch.Size((self.num_instances, num_bodies, 4)))

    def test_get_body_ids(self):
        """Test getting body IDs."""
        body_names = ["pelvis", "torso_link", "left_ankle_roll_link"]
        body_ids = self.robot.get_body_ids(body_names)
        self.assertEqual(len(body_ids), len(body_names))
        for name in body_names:
            self.assertIn(name, body_ids)

    def test_get_joint_ids(self):
        """Test getting joint IDs."""
        joint_names = ["left_hip_pitch_joint", "right_knee_joint", "waist_yaw_joint"]
        joint_ids = self.robot.get_joint_ids(joint_names)
        self.assertEqual(len(joint_ids), len(joint_names))
        for name in joint_names:
            self.assertIn(name, joint_ids)


if __name__ == "__main__":
    unittest.main()
