import numpy as np
from scipy.stats import multivariate_normal as mvn

# ----------------------------NB------------------------------
# ALL CODE BELOW ASSUMES THAT THE CLASSES ARE 0, 1, 2,..., K-1
# --------------------------IMPORTANT-------------------------

class GaussianGenerativeModel:
    def __init__(self, is_shared_covariance=False):
        self.is_shared_covariance = is_shared_covariance
        self.pi = None
        self.mu = None
        self.Sigma = None
        self.labels = None
        self.gaussians = None

    def fit(self, X, y):
        self.labels, counts = np.unique(y, return_counts=True)
        self.pi = counts / len(y)
        self.mu = np.array([np.mean(X[np.where(y == i)], axis=0) for i in self.labels])
        if self.is_shared_covariance:
            mus = np.array([self.mu[i] for i in y])
            # diff is m x n where n is length of a single mu and m is number of data points
            # and we want to end up with n x n; note that m = len(y)
            diff = X - mus
            self.Sigma = diff.T @ diff / len(y)
            self.gaussians = [mvn(mean=self.mu[i], cov=self.Sigma) for i in self.labels]
        else:
            def sigma(label):
                X_in_class = X[np.where(y == label)]
                # diff is m x n where n is length of a single mu and m is number of data points in class i
                # and we want to end up with n x n; note that m = len(X_in_class)
                diff = X_in_class - self.mu[label:label+1]
                return diff.T @ diff / len(X_in_class)
            self.Sigma = [sigma(i) for i in self.labels]
            self.gaussians = [mvn(mean=self.mu[i], cov=self.Sigma[i]) for i in self.labels]

    def predict(self, X_pred):
        # p(y|x) is proportional to p(x|y) * p(y)
        def pred(x):
            return np.argmax([self.gaussians[i].pdf(x) * self.pi[i] for i in self.labels])
        return np.array([pred(x) for x in X_pred])

    def negative_log_likelihood(self, X, y):
        def log_likelihood(x, label):
            return self.gaussians[label].logpdf(x) + np.log(self.pi[label])
        return -np.sum([log_likelihood(X[i], y[i]) for i in range(len(y))])