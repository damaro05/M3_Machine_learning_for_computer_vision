import numpy as np
from scipy.misc import comb

class LogisticRegression():
    def __init__(self,alpha=0.1, max_iterations=2500,method='OneVsOne'):
        self.alpha = alpha
        self.max_iterations = max_iterations
        self.method=method

    def _sigmoid(self,X):
        '''
        Computes the Sigmoid function of the input argument X.
        '''
        return 1.0 / (1 + np.exp(-X))

    def _classify_vector_prob(self,X,i):
        '''
        Evaluate the Logistic Regression model h(x) with theta parameters,
        and returns probability of x.
        '''
        return self._sigmoid(sum(np.dot(X[np.newaxis], self.theta[:,i])))


    def fit(self,X, y):
        x = np.ones(shape=(X.shape[0], 1))
        x = np.append(x, X, axis=1)

        m,n = x.shape  # number of features

        # y must be a column vector
        y = y.reshape(m, 1)

        self.labels = np.unique(y)
        nlabels = len(self.labels)
        if self.method=='OneVsAll':
            # initialize the parameters
            self.theta = np.ones(shape=(n, nlabels))

            for y1 in range(nlabels):
                cury = np.squeeze(y == self.labels[y1])

                # Repeat until convergence (or max_iterations)
                for iteration in range(self.max_iterations):
                    h = self._sigmoid(np.dot(x, self.theta[:, y1]))
                    error = (h - cury)
                    gradient = np.dot(x.T, error) / m
                    self.theta[:, y1] = self.theta[:, y1] - self.alpha * gradient
        else:
            # initialize the parameters
            self.theta = np.ones(shape=(n, comb(nlabels,2,exact=True)))

            i=0
            for y1 in range(nlabels):
                for y2 in range(y1+1,nlabels):
                    indexes = np.squeeze(np.logical_or(y==self.labels[y1],y==self.labels[y2]))
                    curx = x[indexes,:]
                    cury = np.squeeze(y[indexes]==self.labels[y1])

                    curm = curx.shape[0]  # number of samples

                    # Repeat until convergence (or max_iterations)
                    for iteration in range(self.max_iterations):
                        h = self._sigmoid(np.dot(curx, self.theta[:,i]))
                        error = (h - cury)
                        gradient = np.dot(curx.T, error) / curm
                        self.theta[:,i] = self.theta[:,i] - self.alpha * gradient
                    i=i+1
        return self



    def predict(self,X):
        x = np.ones(shape=(X.shape[0], 1))
        x = np.append(x, X, axis=1)

        #nlr=self.theta.shape[1]
        nlabels = len(self.labels)
        if self.method=='OneVsAll':
            probabilities = np.zeros((x.shape[0], nlabels))
            for i in range(x.shape[0]):
                for y1 in range(nlabels):
                    probabilities[i, y1] = probabilities[i, y1] + self._classify_vector_prob(x[i, :], y1)
        else:
            probabilities = np.zeros((x.shape[0],nlabels))
            for i in range(x.shape[0]):
                j=0
                for y1 in range(nlabels):
                    for y2 in range(y1 + 1, nlabels):
                        probabilities[i,y1] = probabilities[i,y1] + self._classify_vector_prob(x[i, :],j)
                        probabilities[i,y2] = probabilities[i,y2] + 1 - self._classify_vector_prob(x[i, :],j)
                        j=j+1

        return np.argmax(probabilities,axis=1)