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
from correlation import (beta_to_alpha,
                         correlation_feature_selection_bqm,
                         correlation_feature_selection_cqm,
                         make_cqm)
from data import Titanic


class TestCorrelation(unittest.TestCase):
    # Simple test dataset
    def setUp(self):
        self.titanic = Titanic()
    
    def test_beta_to_alpha(self):
        self.assertEqual(beta_to_alpha(1,5), 1)
        self.assertEqual(beta_to_alpha(0.5, 3), 0.8)

    def test_correlation_feature_selection_bqm(self):
        bqm_pearson = correlation_feature_selection_bqm(
            self.titanic.X, self.titanic.y, 0.5, "pearson"
        )

        # Expecting 14 variables for this dataset
        self.assertEqual(bqm_pearson.num_variables, 14)

        # Expecting 91 interactions, (14*14 - 14)/2 
        self.assertEqual(bqm_pearson.num_interactions, 91)

        bqm_spearman = correlation_feature_selection_bqm(
            self.titanic.X, self.titanic.y, 0.5, "spearman"
        )
        self.assertEqual(bqm_spearman.num_variables, 14)
        self.assertEqual(bqm_spearman.num_interactions, 91)

        with self.assertRaises(ValueError):
            correlation_feature_selection_bqm(
                self.titanic.X, self.titanic.y, 0.5, "unknown"
            )

    def test_correlation_feature_selection_cqm(self):
        cqm_pearson = correlation_feature_selection_cqm(
            self.titanic.X, self.titanic.y, 0.5, 5, "pearson"
        )

        self.assertEqual(len(cqm_pearson.variables), 14)
        self.assertEqual(len(cqm_pearson.constraints), 1)

        cqm_spearman = correlation_feature_selection_cqm(
            self.titanic.X, self.titanic.y, 0.5, 5, "spearman"
        )

        self.assertEqual(len(cqm_spearman.variables), 14)
        self.assertEqual(len(cqm_spearman.constraints), 1)

        with self.assertRaises(ValueError):
            correlation_feature_selection_cqm(
                self.titanic.X, self.titanic.y, 0.5, 5, "unknown"
            )
    
    def test_make_cqm(self):
        test_bqm = correlation_feature_selection_bqm(
            self.titanic.X, self.titanic.y, 0.5, "pearson"
        )

        test_cqm = make_cqm(test_bqm, k=5)

        self.assertEqual(len(test_cqm.variables), 14)
        self.assertEqual(len(test_cqm.constraints), 1)
        