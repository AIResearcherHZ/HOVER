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

from neural_wbc.core.modes import NeuralWBCModes

from .test_main import APP_IS_READY

if APP_IS_READY:
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env import NeuralWBCEnv
    from neural_wbc.isaac_lab_wrapper.neural_wbc_env_cfg_taks_t1 import NeuralWBCEnvCfgTaksT1


class TestNeuralWBCEnvTaksT1(unittest.TestCase):
    """Test class for NeuralWBC environment with Taks_T1 robot."""

    @classmethod
    def setUpClass(cls):
        if not APP_IS_READY:
            raise unittest.SkipTest("Isaac Lab is not available")

    def test_step(self):
        # create environment configuration
        env_cfg = NeuralWBCEnvCfgTaksT1()
        env_cfg.scene.num_envs = 100
        env_cfg.scene.env_spacing = 20
        env_cfg.mode = NeuralWBCModes.DISTILL  # Set modes to distill so that all observations are tested.

        # setup RL environment
        env = NeuralWBCEnv(cfg=env_cfg)

        # reset environment
        obs, info = env.reset()
        self.assertIsNotNone(obs)
        self.assertIsNotNone(info)

        # step environment
        action = torch.zeros(env_cfg.scene.num_envs, env_cfg.action_space)
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsNotNone(obs)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(terminated)
        self.assertIsNotNone(truncated)
        self.assertIsNotNone(info)

        # close environment
        env.close()

    def test_action_space(self):
        env_cfg = NeuralWBCEnvCfgTaksT1()
        # Taks_T1 has 32 DOF
        self.assertEqual(env_cfg.action_space, 32)

    def test_joint_count(self):
        env_cfg = NeuralWBCEnvCfgTaksT1()
        self.assertEqual(len(env_cfg.joint_names), 32)

    def test_body_count(self):
        env_cfg = NeuralWBCEnvCfgTaksT1()
        self.assertEqual(len(env_cfg.body_names), 33)


if __name__ == "__main__":
    unittest.main()
