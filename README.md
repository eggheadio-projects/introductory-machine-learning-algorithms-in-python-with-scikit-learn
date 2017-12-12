## About this course:
The focus of this course is on implementation and a high-level understanding of these algorithms. We will not be going too much into the math behind them, but instead learn what each algorithm is good for, and how to train them. We'll also learn about a few metrics for evaluating models.

#### Evaluating models:
In this course, we'll look at a few ways to evaluate our models, for both classification and regression models. We'll touch on mean squared error and coefficient of determination (for regression), and accuracy score, logarithmic loss, confusion matrices, and classification reports (for classification). More in-depth ways to evaluate models can be found at http://scikit-learn.org/stable/modules/model_evaluation.html.

#### Disclaimer:
We are working with toy datasets here, so it is important not to read too much into the numbers and accuracy results from these lessons. Real datasets will be much larger. 


### Installation:

Scikit-learn and matplotlib need to be installed via your installer of choice (generally 'pip install scikit-learn' and 'pip install matplotlib' work). For lesson  5, pandas_ml will also need to be installed for those who want to visualize their confusion matrix. For lesson 6, graphviz needs to be installed.


### Datasets: 

Read more about scikit-learn's datasets at http://scikit-learn.org/stable/datasets/

To load your own, review this page to make sure you format your dataset correctly for the scikit-learn library: http://scikit-learn.org/stable/datasets/index.html#external-datasets


### Vocabulary:

#### Classification: a supervised learning problem where the aim is to classify data into discrete categories.

#### Regression: a supervised learning problem that aims to predict a number along a continuous spectrum, such as estimating the price of a house.

Classification and regression both fall under the category of supervised learning.

#### Supervised Learning: using algorithms on data that has target labels.

#### Unsupervised Learning: using algorithms on data that does not have labels.

You should understand basic machine learning vocabulary (including 'classification', 'regression', 'supervised learning', and 'unsupervised learning') for this course. Scikit-learn has a vocabulary section at http://scikit-learn.org/stable/tutorial/basic/tutorial.html.


### Common Errors:

#### "RuntimeError: Python is not installed as a framework":

In terminal, run: 

python -m site 

to find where your site packages are. Open the matplotlibrc file at PATH_TO_SITE_PACKAGES/site-packages/matplotlib/mpl-data/matplotlibrc. 

Change the backend to an interactive backend. I am using this line:

backend      : TkAgg 