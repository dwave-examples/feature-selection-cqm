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

"""Implementation of feature selection based on maximization of relevance and minimization of redundancy.

Reference:

Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
Selection in Credit Scoring and Classification Using a Quantum Annealer."
1QBit; White Paper.
<https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer/>.
"""

import numpy as np
from scipy.stats import spearmanr

import dimod


def correlation_feature_selection_bqm(X, y, alpha, correlation_type='pearson'):
    """Build BQM for feature selection based on maximizing influence and independence as
    measured by correlation.

    Based on the formulation given in Milne et al., 2017.

    Args:
        X (array):
            2D array of feature vectors (numerical).
        y (array):
            1D array of class labels (numerical).
        alpha (float):
            Hyperparameter between 0 and 1 that controls the relative weight of
            the relevance and redundancy terms.  `alpha=1` places all weight on
            relevance and selects all features, whereas `alpha=0` places all
            weight on redundancy and selects no features.
        correlation_type (str):
            Type of correlation coefficient to use.  Valid values are
            "spearman" and "pearson".

    Returns:
        bqm (BinaryQuadraticModel)
    """
    if correlation_type == 'spearman':
        correlation_func = lambda data: spearmanr(data)[0]
    elif correlation_type == 'pearson':
        correlation_func = lambda data: np.corrcoef(data, rowvar=False)
    else:
        raise ValueError("unknown correlation type")

    correlation_matrix = abs(correlation_func(np.hstack((X, y[:, np.newaxis]))))
    # Note: the full symmetric matrix (with both upper- and lower-diagonal
    # entries for each correlation coefficient) is retained for consistency with
    # the original formulation from Milne et al.
    Q = correlation_matrix[:-1,:-1] * (1 - alpha)
    Rxy = correlation_matrix[:,-1]
    np.fill_diagonal(Q, -Rxy * alpha)

    bqm = dimod.BinaryQuadraticModel(Q, 'BINARY')

    return bqm


def make_cqm(bqm, k):
    """Create a CQM by adding a k-of-n cardinality constraint to a BQM.

    Args:
        bqm (BinaryQuadraticModel)
        k (int): Specifies the cardinality constraint such that $\Sum x_i = k$.

    Returns:
        ConstrainedQuadraticModel
    """
    cqm = dimod.ConstrainedQuadraticModel()
    cqm.set_objective(bqm)
    cqm.add_constraint_from_iterable(((i, 1) for i in range(len(cqm.variables))), '==', rhs=k)
    return cqm


def correlation_feature_selection_cqm(X, y, alpha, k, correlation_type='pearson'):
    """Construct a CQM for feature selection using relevance and redundancy.

    Args:
        X (array): 2d array of feature data with features as columns.
        y (array): 1d array of target data.
        alpha (float): hyperparameter controlling relevance and redundancy tradeoff.
        k (int): Number of features to include in solution.
        correlation_type (str): 'pearson' or 'spearman'.

    Returns:
        ConstrainedQuadraticModel
    """
    bqm = correlation_feature_selection_bqm(X, y, alpha, correlation_type=correlation_type)
    return make_cqm(bqm, k)


def beta_to_alpha(beta, k):
    """Convert adjusted coefficient weighting parameter.

    The original formulation from Milne et al. expresses the relative weighting
    of linear (relevance) and quadratic (redundancy) coefficients through a
    parameter alpha.  However, this parameter does not adjust for the fact that
    the total number of quadratic terms depends on the number of features in the
    solution.

    We define a new parameter, beta, which adjusts for the relative number of
    linear and quadratic terms.  Let the objective function be expressed as:

    $$\beta \sum linear + (1 - \beta) / (2 (k - 1)) \sum quad$$

    where $k$ is the number of features in the solution, and the $2*(k-1)$
    factor corrects for the relative number of terms in each sum (the factor of
    2 is because the Milne et al. formulation counts each unique redundancy term
    twice).

    The conversion between alpha and beta is obtained by equating the ratio of
    linear to quadratic weighting coefficients in each formulation.

    Args:
        beta (float): Weighting parameter between 0 and 1.
        k (int): Number of features in the solution.

    Returns:
        alpha (float)
    """
    if beta == 1:
        return 1
    return 2 * beta * (k-1) / (1 - beta + 2*beta*(k-1))
