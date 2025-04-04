[![Open in GitHub Codespaces](
  https://img.shields.io/badge/Open%20in%20GitHub%20Codespaces-333?logo=github)](
  https://codespaces.new/dwave-examples/feature-selection-cqm?quickstart=1)

# Feature Selection for CQM

This demo showcases feature selection using the constrained quadratic model
(CQM) solver via
[D-Wave's scikit-learn plug-in](https://github.com/dwavesystems/dwave-scikit-learn-plugin).
The demo can be used with two different datasets:

- `titanic`: This is a well-known dataset based on passenger survival from the
  Titanic.  It includes 14 features and illustrates how feature redundancy
  impacts the solution.
- `scene`: This is a larger dataset with 299 features.  It is associated with
  recognizing scenes based on feature data contained in images.  For additional
  information, see:
  [OpenML](https://www.openml.org/search?type=data&sort=runs&id=312&status=active).
  The features have generic labels such as "attr1" and are associated with image
  characteristics such as the mean or variance of different color channels
  within regions of the image.  As with the Titanic data, this dataset also
  illustrates the impact of feature redundancy.

---
**Note:** This example solves a CQM on a Leap&trade; quantum-classical
[hybrid solver](https://docs.dwavequantum.com/en/latest/concepts/hybrid.html).
The [MIQUBO Method of Feature Selection](https://github.com/dwave-examples/mutual-information-feature-selection)
example solves this same problem using a
[binary quadratic model (BQM)](https://docs.dwavequantum.com/en/latest/concepts/models.html#binary-quadratic-models)
directly on a D-Wave quantum computer's quantum processing unit (QPU).

---

## Installation

You can run this example without installation in cloud-based IDEs that support
the
[Development Containers specification](https://containers.dev/supporting) (aka
"devcontainers") such as GitHub Codespaces.

For development environments that do not support `devcontainers`, install
requirements:

```bash
pip install -r requirements.txt
```

If you are cloning the repo to your local system, working in a
[virtual environment](https://docs.python.org/3/library/venv.html) is
recommended.

## Usage

Your development environment should be configured to access the
[Leap&trade; quantum cloud service](https://docs.dwavequantum.com/en/latest/ocean/sapi_access_basic.html).
You can see information about supported IDEs and authorizing access to your Leap
account
[here](https://docs.dwavequantum.com/en/latest/leap_sapi/dev_env.html).

Run the following terminal command to start the Dash application:

```bash
python app.py
```

Access the user interface with your browser at http://127.0.0.1:8050/.
A dropdown menu is provided to choose the dataset.

To visualize feature redundancy, first activate the "Show redundancy" check box.
Then hover the mouse over any of the bars.  The colors of all bars will be
dynamically updated to show the similarity (redundancy) against the feature that
is currently under the mouse.

Click on the `Solve` button to run the feature selection with the given settings
(each dataset is initialized with reasonable default settings for the number of
features and redundancy penalty).  Solutions typically take 1-3 seconds.  Once
complete, the bar chart will update to reflect the selected features, and the
bar graph for accuracy scores will also be updated.

## References

Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
Selection in Credit Scoring and Classification Using a Quantum Annealer."
1QBit; White Paper.
https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer/
