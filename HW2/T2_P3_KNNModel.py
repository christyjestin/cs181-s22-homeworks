import numpy as np

class KNNModel:
    def __init__(self, k):
        self.X = None
        self.y = None
        self.K = k

    def predict(self, X_pred):
        def pred(x):
            diff = x - self.X
            diff[:, 0] = diff[:, 0] / 3
            k_closest_labels = self.y[np.argsort(np.linalg.norm(diff, axis=1))][:self.K]
            labels, counts = np.unique(k_closest_labels, return_counts=True)
            return labels[np.argmax(counts)]
        return np.array([pred(x) for x in X_pred])

    def fit(self, X, y):
        self.X = X
        self.y = y