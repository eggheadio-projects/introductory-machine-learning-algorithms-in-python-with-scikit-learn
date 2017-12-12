import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = datasets.load_boston()

X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=16)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

#r squared, coefficient of determination
print model.score(X_test, y_test)

#mean squared error
print metrics.mean_squared_error(y_test, predictions)
