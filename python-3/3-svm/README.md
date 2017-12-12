## Use Support Vector Machines To Find More Complex Decision Boundaries with Scikit-learn’s SVM module in Python

We’ll continue with the iris dataset to implement support vector machines, which can be used to find more complex boundaries for classification or regression problems. 

More about Support Vector Machines can be found at http://scikit-learn.org/stable/modules/svm.html.

There are several types of kernel function that can be used with SVMs. Scikit-learn supports these kernels:

-linear
-polynomial ('poly')
-rbf (radial basis function) 
-sigmoid 

Custom kernels are also supported. Rbf is the default kernel type.

A good overview of kernels can be found at https://www.kdnuggets.com/2016/06/select-support-vector-machine-kernels.html, or at the scikit-learn page at http://scikit-learn.org/stable/modules/svm.html#svm-kernels.