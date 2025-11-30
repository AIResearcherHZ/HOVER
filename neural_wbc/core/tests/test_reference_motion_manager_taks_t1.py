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

from neural_wbc.core.reference_motion import ReferenceMotionManager, ReferenceMotionManagerCfg
from neural_wbc.data import get_data_path


class TestReferenceMotionManagerTaksT1(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.num_envs = 2

        cfg = ReferenceMotionManagerCfg()
        cfg.skeleton_path = get_data_path("motion_lib/Taks_T1/Taks_T1.xml")
        cfg.motion_path = get_data_path("motions/stable_punch.pkl")

        self.motion_manager = ReferenceMotionManager(
            cfg=cfg,
            num_envs=self.num_envs,
            device=self.device,
        )

    def test_initialization(self):
        """Test that the motion manager initializes correctly."""
        self.assertIsNotNone(self.motion_manager)

    def test_sample_motions(self):
        """Test sampling motions."""
        self.motion_manager.sample_motions()
        # Should not raise any errors

    def test_get_motion_state(self):
        """Test getting motion state."""
        self.motion_manager.sample_motions()
        motion_state = self.motion_manager.get_motion_state(time_offset=0.0)
        self.assertIsNotNone(motion_state)


if __name__ == "__main__":
    unittest.main()
