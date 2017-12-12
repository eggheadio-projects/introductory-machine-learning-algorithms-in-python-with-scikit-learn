from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

iris = datasets.load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

model = SVC(kernel='poly')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.score(X_test, y_test))

print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))