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

"""Test evaluator with Taks_T1 robot model."""

import torch
import unittest
from unittest.mock import Mock

from neural_wbc.core import EnvironmentWrapper
from neural_wbc.core.evaluator import Evaluator
from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg
from neural_wbc.data import get_data_path


class TestEvaluatorWithReferenceManagerTaksT1(unittest.TestCase):
    """Test Evaluator with Taks_T1 reference motion manager."""

    num_envs = 1
    device = torch.device("cpu")

    def setUp(self):
        env_wrapper = Mock(spec=EnvironmentWrapper)
        env_wrapper.num_envs = self.num_envs
        env_wrapper.device = self.device

        env_wrapper.reference_motion_manager = self._create_reference_motion_manager()
        self.evaluator = Evaluator(env_wrapper)

    def _create_reference_motion_manager(self):
        cfg = ReferenceMotionManagerCfg()
        cfg.motion_path = get_data_path("motions/stable_punch.pkl")
        cfg.skeleton_path = get_data_path("motion_lib/Taks_T1/Taks_T1.xml")

        return ReferenceMotionManager(
            cfg=cfg,
            device=self.device,
            num_envs=self.num_envs,
            random_sample=False,
            extend_head=False,
            dt=0.005,
        )

    def _create_dones_and_info(self, ref_motion_state):
        # Construct a test data that fails at first step.
        dones = torch.tensor([True])
        info = {
            "data": {
                "state": {},
                "ground_truth": {},
                "upper_joint_ids": range(0, 20),  # Taks_T1 has more upper body joints
                "lower_joint_ids": range(20, 32),  # Taks_T1 has 32 joints total
            },
            "termination_conditions": {"gravity": torch.tensor([[True]])},
        }
        # Only create mask for the bodies as that's the only thing this test cares for.
        num_bodies = ref_motion_state.body_pos_extend.shape[1]
        info["data"]["mask"] = torch.zeros((self.num_envs, num_bodies), dtype=torch.bool)
        info["data"]["mask"][:, -3:] = True

        # Copy reference motion state data to both ground truth and state dictionaries in info
        for key in ["body_pos_extend", "joint_pos", "root_pos", "root_lin_vel", "root_rot"]:
            data = getattr(ref_motion_state, key).detach().clone()
            info_key = key.replace("_extend", "")  # handle body_pos_extend special case
            info["data"]["ground_truth"][info_key] = data
            info["data"]["state"][info_key] = data
        return dones, info

    def test_forward_motion_without_offset(self):
        # The internal motion manager cache will always reset if the offset is None.
        # Load the initial frame of the first motion
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        ref_motion_state1 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=None
        )

        dones, info = self._create_dones_and_info(ref_motion_state=ref_motion_state1)
        reset_env = self.evaluator.collect(dones=dones, info=info)
        # Confirm reset is true and
        self.assertTrue(reset_env)
        self.assertFalse(self.evaluator.is_evaluation_complete())

        # Forward the motion lib.
        self.evaluator.forward_motion_samples()
        ref_motion_state2 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf,
            offset=None,
        )

        # Confirm that the motion is updated properly
        self.assertFalse(torch.equal(ref_motion_state1.joint_pos, ref_motion_state2.joint_pos))

    def test_forward_motion_with_offset(self):
        # Load the initial frame of the first motion
        episode_length_buf = torch.Tensor([0] * self.num_envs, device=self.device).long()
        start_positions_on_terrain = torch.zeros([self.num_envs, 3], device=self.device)
        ref_motion_state1 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )

        dones, info = self._create_dones_and_info(ref_motion_state=ref_motion_state1)
        reset_env = self.evaluator.collect(dones=dones, info=info)
        # Confirm reset is true and the evaluation is not complete.
        self.assertTrue(reset_env)
        self.assertFalse(self.evaluator.is_evaluation_complete())

        # Forward the motion lib.
        self.evaluator.forward_motion_samples()
        ref_motion_state2 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )

        # Confirm that the motion is updated. When we forward motion samples, we move on to the next motion.
        self.assertFalse(torch.equal(ref_motion_state1.joint_pos, ref_motion_state2.joint_pos))

        # Set the episode length properly with make the motion cache updated properly.
        episode_length_buf += 1
        ref_motion_state3 = self.evaluator._ref_motion_mgr.get_state_from_motion_lib_cache(
            episode_length_buf=episode_length_buf, offset=start_positions_on_terrain
        )
        self.assertFalse(torch.equal(ref_motion_state2.joint_pos, ref_motion_state3.joint_pos))

    def test_reference_motion_manager_initialization(self):
        """Test that the reference motion manager is properly initialized with Taks_T1 skeleton."""
        self.assertIsNotNone(self.evaluator._ref_motion_mgr)
        self.assertEqual(self.evaluator._num_envs, self.num_envs)


if __name__ == "__main__":
    unittest.main()
