import numpy as np
import matplotlib.pyplot as plt

# set up data
data = [(0., 0.),
        (1., 0.5),
        (2., 1),
        (3., 2),
        (4., 1),
        (6., 1.5),
        (8., 0.5)]

x_train = np.array([d[0] for d in data])
y_train = np.array([d[1] for d in data])
x_test = np.arange(0, 12, 0.1)

def predict_knn(k):
    def helper(x):
        return np.mean(y_train[np.argsort(np.abs(x_train - x))][:k])
    return np.array([helper(x) for x in x_test])


def plot_knn_preds(k):
    plt.xlim([0, 12])
    plt.ylim([0,3])

    y_test = predict_knn(k)

    plt.scatter(x_train, y_train, label = "training data", color = 'black')
    plt.plot(x_test, y_test, label = "predictions using k = " + str(k))

    plt.legend()
    plt.title("KNN Predictions with k = " + str(k))
    plt.show()

for k in (1, 3, len(x_train)-1):
    plot_knn_preds(k)