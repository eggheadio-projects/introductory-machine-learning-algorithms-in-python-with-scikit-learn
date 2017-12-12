## Use Logistic Regression To Estimate Discrete Values (Classification) with Python and Scikit-learn

Despite its often confusing name, logistic regression is a **linear** model that is used for **classification**, or estimating discrete values.

We'll use an inbuilt scikit-learn dataset of iris data to classify irises into three categories. We'll also look at metrics and tools to evaluate our classification models, including the accuracy score, classification report, and confusion matrix.

Precision, recall, f1-score, and support are used in the classification report to give a general indicator of how well the model did. More about these variables can be found at http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html.

We’ll see how to implement logistic regression to estimate discrete values. In this case, we’ll classify images of irises.

**Precision**: the amount of true positives over the amount of true and false positives. What is the probability of a sample actually being positive if it is labeled as positive?

**Recall**: the amount of true positives of the amount of true positives plus the false negatives. What is the probability that the model will accurately pick up on a true positive?

**F1-score**: a combination of precision and recall.

**Support**: the number of samples of each class in the dataset.

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html