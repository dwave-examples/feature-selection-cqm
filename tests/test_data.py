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

import numpy as np
import unittest
from unittest.mock import patch
import data


class TestData(unittest.TestCase):
    @patch("data.solve_feature_selection_cqm")
    @patch("data.correlation_feature_selection_cqm")
    @patch("data.beta_to_alpha")
    def test_titanic_class(self, mock_beta, mock_corr, mock_solve):
        titanic = data.Titanic()
        relevance = titanic.get_relevance()

        self.assertEqual(len(relevance), titanic.n)
        for value in relevance:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1.0)

        redundancy = titanic.calc_redundancy()
        self.assertEqual(redundancy.shape, (titanic.n, titanic.n))

        from_get_redundancy = titanic.get_redundancy()
        self.assertTrue(np.array_equal(redundancy, from_get_redundancy))

        titanic.solve_cqm(k=3, beta=0.5, time_limit=5)
        mock_beta.assert_called_with(0.5, 3)
        mock_corr.assert_called_with(
            titanic.X, titanic.y, mock_beta(), 3
        )
        mock_solve.assert_called_with(mock_corr(), time_limit=5)
    
        score_by_feature_indices = titanic.score_indices_cv(
            list(range(np.size(titanic.X, 1)))
        )
        self.assertLessEqual(score_by_feature_indices, 1.0)
        self.assertGreaterEqual(score_by_feature_indices, 0)

        baseline_score = titanic.score_baseline_cv()
        self.assertLessEqual(baseline_score, 1.0)
        self.assertGreaterEqual(baseline_score, 0)
    
    @patch("data.solve_feature_selection_cqm")
    @patch("data.correlation_feature_selection_cqm")
    @patch("data.beta_to_alpha")
    def test_scene_class(self, mock_beta, mock_corr, mock_solve):
        scene = data.Scene()
        relevance = scene.get_relevance()

        self.assertEqual(len(relevance), scene.n)
        for value in relevance:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 1.0)

        redundancy = scene.calc_redundancy()
        self.assertEqual(redundancy.shape, (scene.n, scene.n))

        from_get_redundancy = scene.get_redundancy()
        self.assertTrue(np.array_equal(redundancy, from_get_redundancy))

        scene.solve_cqm(k=3, beta=0.5, time_limit=5)
        mock_beta.assert_called_with(0.5, 3)
        mock_corr.assert_called_with(
            scene.X, scene.y, mock_beta(), 3
        )
        mock_solve.assert_called_with(mock_corr(), time_limit=5)
    
        score_by_feature_indices = scene.score_indices_cv(
            list(range(np.size(scene.X, 1)))
        )
        self.assertLessEqual(score_by_feature_indices, 1.0)
        self.assertGreaterEqual(score_by_feature_indices, 0)

        baseline_score = scene.score_baseline_cv()
        self.assertLessEqual(baseline_score, 1.0)
        self.assertGreaterEqual(baseline_score, 0)

    def test_dataset(self):
        titanic = data.DataSet("titanic")
        scene = data.DataSet("scene")

        self.assertIsInstance(titanic, data.Titanic)
        self.assertIsInstance(scene, data.Scene)
        