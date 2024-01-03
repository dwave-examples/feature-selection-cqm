[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/feature-selection-cqm?quickstart=1)

# Feature Selection for CQM

This demo showcases feature selection using the constrained quadratic model
(CQM) solver via 
[D-Wave's scikit-learn plug-in](https://github.com/dwavesystems/dwave-scikit-learn-plugin).
The demo can be used with two different data sets:

- `titanic`: This is a well-known data set based on passenger survival from the
  Titanic.  It includes 14 features and illustrates how feature redundancy
  impacts the solution.
- `scene`: This is a larger data set with 299 features.  It is associated with
  recognizing scenes based on feature data contained in images.  For additional
  information, see:
  [OpenML](https://www.openml.org/search?type=data&sort=runs&id=312&status=active).
  The features have generic labels such as "attr1" and are associated with image
  characteristics such as the mean or variance of different color channels
  within regions of the image.  As with the Titanic data, this dataset also
  illustrates the impact of feature redundancy.

---
**Note:** This example solves a CQM on a Leap&trade; quantum-classical 
[hybrid solver](https://docs.ocean.dwavesys.com/en/stable/concepts/hybrid.html). 
The [MIQUBO Method of Feature Selection](https://github.com/dwave-examples/mutual-information-feature-selection) 
example solves this same problem using a
[binary quadratic model (BQM)](https://docs.ocean.dwavesys.com/en/stable/concepts/bqm.html)
directly on a D-Wave quantum computer's quantum processing unit (QPU).

---

## Installation

You can run this example without installation in cloud-based IDEs that support 
the [Development Containers specification](https://containers.dev/supporting)
(aka "devcontainers").

For development environments that do not support ``devcontainers``, install 
requirements:

    pip install -r requirements.txt

If you are cloning the repo to your local system, working in a 
[virtual environment](https://docs.python.org/3/library/venv.html) is 
recommended.

## Usage

Run `python app.py` and open http://127.0.0.1:8050/ in your browser.  A
dropdown menu is provided to choose the dataset.

To visualize feature redundancy, first activate the "Show redundancy" check box.
Then hover the mouse over any of the bars.  The colors of all bars will be
dynamically updated to show the similarity (redundancy) against the feature that
is currently under the mouse.

Click on the `Solve` button to run the feature selection with the given settings
(each data set is initialized with reasonable default settings for the number of
features and redundancy penalty).  Solutions typically take 1-3 seconds.  Once
complete, the bar chart will update to reflect the selected features, and the
bar graph for accuracy scores will also be updated.

## Problem Description

The objective for this feature selection application aims to choose features that
optimize relationships between individual features in the dataset as well as between
the features and the target variable. The goal is to choose features that will help 
the machine learning model learn by promoting diversity between features and strong
relationships to the target variable. Correlation between features generates the 
redundancy metric, which is minimized between each pair of features to promote 
diversity. Correlation between features and the target is maximized to promote a 
strong relationship. 

The constraints here focus on choosing the exact number of defined features.

## Model Overview

In this example we use the Titanic and Scene datasets to generate a Constrained Quadratic Model. 
The datasets are assumed to be clean, meaning there are no missing entries or repeated features. 
The features of the dataset are used to build a correlation matrix which compares the features to 
each other as well as a correlation matrix that compares the features to the target variable. Those 
correlation matrices are used to build the objective function. 

---
**Note:** Although a model overview is provided here, all of the code to build
the feature selection model is contained within 
[D-Wave's scikit-learn plug-in](https://github.com/dwavesystems/dwave-scikit-learn-plugin).
---

### Parameters

These are the parameters of the problem:

- `num_features`: the number of selected features
- `redund_val`: used to determine factor applied to redundancy terms

### Variables
- `x_i`: binary variable that shows if feature `i` is selected

### Expressions
 Expressions are linear or quadratic combinations of variables used for easier representations of the models. 
- None

### Objective
The objective function has two terms. The first term corresponds to minimizing the correlation between 
chosen features in the dataset controlled by the redundancy parameter. The second term corresponds to 
maximizing the correlation between the features and the target variable. 
### Constraints
A single constraint is used to enforce the number of features selected by the model matches the `num_features`
parameter.
## Code Overview

Given a selected value for the `num_features` and `redund_val` sliders, the code proceeds as follows:

* The selected dataset and parameters are passed to D-Wave's feature selection scikit-learn plugin
* The resulting selected features are returned from the plugin
* A random forest classifier model is trained on the selected features and accuracy score is calculated
* The display image is updated to reflect the selected features and the classifier accuracy 

## References

Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
Selection in Credit Scoring and Classification Using a Quantum Annealer."
1QBit; White Paper.
https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer/
