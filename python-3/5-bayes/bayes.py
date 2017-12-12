from sklearn import datasets
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix

newsgroups_train = datasets.fetch_20newsgroups(subset='train')
newsgroups_test = datasets.fetch_20newsgroups(subset='test')

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(newsgroups_train.data)
X_test = vectorizer.transform(newsgroups_test.data)

y_train = newsgroups_train.target
y_test = newsgroups_test.target

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# print model.score(X_test, y_test)
# print metrics.classification_report(y_test, predictions)

labels = list(newsgroups_train.target_names)
cm = ConfusionMatrix(y_test, predictions, labels)
cm.plot()
plt.show()
