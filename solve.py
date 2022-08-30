# COPYRIGHT 2022 D-WAVE SYSTEMS INC.

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
    results = sampler.sample_cqm(cqm, **sampler_args)
    return postprocess_cqm_results(cqm, results, return_indices, return_energy)


def postprocess_cqm_results(cqm, results, return_indices, return_energy):
    feasible = results.filter(lambda s: s.is_feasible)
    if feasible:
        best = feasible.first
    else:
        assert len(cqm.constraints) == 1
        # Sort on violation, then energy
        best = sorted(results.data(), key=lambda x: (list(cqm.violations(x.sample).values())[0],
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
