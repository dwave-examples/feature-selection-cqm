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

"""Utility functions for retrieving the solution to a feature selection problem that is encoded as a binary quadratic model."""

import numpy as np

from dwave.system import LeapHybridCQMSampler



def solve_feature_selection_cqm(cqm, sampler=None, return_indices=True, return_energy=False,
                                **sampler_args):
    """Solve feature selection CQM and retrieve best feature set.

    Args:
        cqm (ConstrainedQuadraticModel)
        sampler (Sampler)
        return_indices (bool):
            If True, return array of selected feature indices.
        return_energy (bool):
            If True, also return energy of the solution.

    Returns:
        features (array):
            If return_indices is True, the array contains the indices
            of the selected features, otherwise it is a mask array.
        energy (float):
            Energy value is returned when return_energy is True.
    """
    if sampler is None:
        sampler = LeapHybridCQMSampler()

    sampler_args.setdefault('label', 'Feature Selection')
    sampleset = sampler.sample_cqm(cqm, **sampler_args)
    return _postprocess_cqm_results(cqm, sampleset, return_indices, return_energy)


def _postprocess_cqm_results(cqm, sampleset, return_indices, return_energy):
    feasible = sampleset.filter(lambda s: s.is_feasible)
    if feasible:
        best = feasible.first
    else:
        assert len(cqm.constraints) == 1
        # Sort on violation, then energy
        best = sorted(sampleset.data(), key=lambda x: (list(cqm.violations(x.sample).values())[0],
                                                     x.energy))[0]

    assert list(best.sample.keys()) == sorted(best.sample.keys())
    is_selected = np.array([bool(val) for val in best.sample.values()])

    if return_indices:
        features = np.array([i for i, val in enumerate(is_selected) if val])
    else:
        features = is_selected

    if return_energy:
        return features, best.energy
    else:
        return features
