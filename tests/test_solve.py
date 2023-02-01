# Copyright 2023 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch
from data import Titanic
from correlation import correlation_feature_selection_cqm
import solve


class TestSolve(unittest.TestCase):
    @patch("solve.LeapHybridCQMSampler")
    def test_solve_feature_selection_cqm(self, mock_sampler):
        sampler = mock_sampler.return_value
        test_data = Titanic()
        test_cqm = correlation_feature_selection_cqm(
            test_data.X, test_data.y, 0.5, 5
        )

        solve.solve_feature_selection_cqm(
            test_cqm, time_limit=5
        )

        mock_sampler.assert_called_once()
        sampler.sample_cqm.assert_called_with(
            test_cqm, time_limit=5, label="Feature Selection"
        )
        