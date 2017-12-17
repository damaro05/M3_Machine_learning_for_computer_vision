import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from logistic_regression import LogisticRegression


def sigmoid(X):
    '''
    Computes the Sigmoid function of the input argument X.
    '''
    return 1.0 / (1 + np.exp(-X))

lr=LogisticRegression()
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
lr.fit(X,y)
H=lr.predict(X)
print("Training Accuracy : "+str(float(np.sum(H == y)) / y.shape[0]))
#Plot data
plt.scatter(X[y==1, 0], X[y==1, 1], marker='o', c='b') #positive samples
plt.scatter(X[y==0, 0], X[y==0, 1], marker='x', c='r') #negative samples

#Plot Decision Boundary
u = np.linspace(-2, 2, 50)
v = np.linspace(-2, 2, 50)
z = np.zeros(shape=(len(u), len(v)))
for i in range(len(u)):
    for j in range(len(v)):
        z[i, j] = sigmoid(np.dot(np.array([1,u[i],v[j]]),lr.theta))

z = z.T

cs = plt.contour(u, v, z, levels=[0.5])
plt.show()