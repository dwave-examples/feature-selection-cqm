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


---
**Note:** Although a model overview is provided here, all of the code to build
the feature selection model is contained within 
[D-Wave's scikit-learn plug-in](https://github.com/dwavesystems/dwave-scikit-learn-plugin).
---

### Parameters

### Variables

### Expressions

### Objective

### Constraints

## Code Overview

A general overview of how the code works.

We prefer descriptions in bite-sized bullet points:

* Here's an example bullet point

## Code Specifics

Notable parts of the code implementation.

This is the place to:

* Highlight a part of the code implementation
* Talk about unusual or potentially difficult parts of the code
* Explain a code decision
* Explain how parameters were tuned

Note: there is no need to repeat everything that is already well-documented in
the code.

## References

Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
Selection in Credit Scoring and Classification Using a Quantum Annealer."
1QBit; White Paper.
https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer/
