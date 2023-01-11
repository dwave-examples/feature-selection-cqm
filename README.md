# Feature Selection Demo

This demo showcases feature selection using the constrained quadratic model
(CQM) solver.  The demo can be used with two different data sets:

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

## References

Milne, Andrew, Maxwell Rounds, and Phil Goddard. 2017. "Optimal Feature
Selection in Credit Scoring and Classification Using a Quantum Annealer."
1QBit; White Paper.
https://1qbit.com/whitepaper/optimal-feature-selection-in-credit-scoring-classification-using-quantum-annealer/
