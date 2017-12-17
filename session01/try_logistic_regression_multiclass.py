from logistic_regression import LogisticRegression
import numpy as np
from sklearn import svm, datasets

# import some data to play with
iris = datasets.load_iris()

# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
lr=LogisticRegression(method='OneVsAll')
lr.fit(X,y)
H=lr.predict(X)

print("Training Accuracy : "+str(float(np.sum(H == y)) / y.shape[0]))

