from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

iris = datasets.load_iris()

X = iris.data[:, 1:3]

model = KMeans(n_clusters=3, random_state=0)
model.fit(X)

centroids = model.cluster_centers_

plt.scatter(centroids[:, 0], centroids[:, 1], marker='^', s=170, zorder=10, c='m')
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.xlabel("Sepal width")
plt.ylabel("Petal length")
plt.show()