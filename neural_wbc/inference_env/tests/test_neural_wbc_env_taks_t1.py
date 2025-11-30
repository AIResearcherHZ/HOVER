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

from inference_env.neural_wbc_env import NeuralWBCEnv
from inference_env.neural_wbc_env_cfg_taks_t1 import NeuralWBCEnvCfgTaksT1

from neural_wbc.data import get_data_path


class TestNeuralWBCEnvTaksT1(unittest.TestCase):
    def setUp(self):
        # create environment configuration
        self.env_cfg = NeuralWBCEnvCfgTaksT1(model_xml_path=get_data_path("mujoco/models/Taks_T1/Taks_T1.xml"))

        # setup RL environment
        self.device = torch.device("cpu")
        self.env = NeuralWBCEnv(cfg=self.env_cfg, device=self.device)

    def test_reset(self):
        obs, _ = self.env.reset()
        self.assertIsNotNone(obs)

    def test_step(self):
        obs, _ = self.env.reset()
        action = torch.zeros(1, self.env_cfg.num_joints)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsNotNone(obs)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(terminated)
        self.assertIsNotNone(truncated)
        self.assertIsNotNone(info)

    def test_observation_shape(self):
        obs, _ = self.env.reset()
        # Check that observation is a tensor
        self.assertIsInstance(obs, torch.Tensor)

    def test_action_shape(self):
        obs, _ = self.env.reset()
        # Taks_T1 has 32 joints
        action = torch.zeros(1, 32)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.assertIsNotNone(obs)

    def test_robot_properties(self):
        # Test that robot has correct number of joints
        self.assertEqual(self.env_cfg.num_joints, 32)
        self.assertEqual(self.env_cfg.num_bodies, 33)

    def test_joint_limits(self):
        # Test that all joints have position limits defined
        for joint_name in self.env_cfg.stiffness.keys():
            self.assertIn(joint_name, self.env_cfg.position_limit)
            self.assertIn(joint_name, self.env_cfg.damping)
            self.assertIn(joint_name, self.env_cfg.effort_limit)


if __name__ == "__main__":
    unittest.main()
