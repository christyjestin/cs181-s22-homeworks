import numpy as np
import matplotlib.pyplot as plt

# ----------------------------NB------------------------------
# ALL CODE BELOW ASSUMES THAT THE CLASSES ARE 0, 1, 2,..., K-1
# --------------------------IMPORTANT-------------------------

ITERATION_BLOCK = 100
MAX_ITERATIONS = 200000

class LogisticRegression:
    def __init__(self, eta, lam):
        self.eta = eta
        self.lam = lam
        self.W = None
        self.losses = None

    def __add_ones(self, X):
        return np.hstack((np.ones((len(X), 1)), X))

    def __y_hat(self, X):
        y_hat = np.exp(np.dot(X, self.W))
        return y_hat / np.linalg.norm(y_hat, ord=1, axis=1, keepdims=True)

    def __step(self, X, y):
        y_hat = self.__y_hat(X)
        def column_gradient(k):
            return np.sum(X * y_hat[:, k:k+1], axis=0) - np.sum(X[np.where(y == k)], axis=0)
        gradient = np.hstack([np.expand_dims(column_gradient(k), axis=1) for k in self.labels])
        regularizer = 2 * self.lam * self.W
        self.W -= (gradient + regularizer) * self.eta

    def __compute_loss(self, X, y):
        return -np.sum([np.log(self.__y_hat(X))[i, v] for i, v in enumerate(y)])

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.W = np.random.rand(X.shape[1] + 1, len(self.labels))
        self.losses = []
        for i in range(MAX_ITERATIONS):
            self.__step(self.__add_ones(X), y)
            if i % ITERATION_BLOCK == 0:
                self.losses.append(self.__compute_loss(self.__add_ones(X), y))


    def predict(self, X_pred):
        return np.argmax(np.dot(self.__add_ones(X_pred), self.W), axis=1)

    def visualize_loss(self, output_file, show_charts=False):
        x = np.arange(0, len(self.losses) * ITERATION_BLOCK, ITERATION_BLOCK) + 1
        y = np.array(self.losses)
        plt.figure()
        plt.title('Loss over Iterations')
        plt.xlabel('Number of Iterations')
        plt.ylabel('Negative Log-Likelihood Loss')
        plt.plot(x, y)
        # Saving the image to a file, and showing it as well
        plt.savefig(output_file + '.png')
        if show_charts:
            plt.show()