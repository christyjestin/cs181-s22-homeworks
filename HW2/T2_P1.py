import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c
import matplotlib.patches as mpatches

def basis_generator(x, highest_power):
    return np.stack([x ** i for i in range(highest_power + 1)], axis=1)

def basis1(x):
    return basis_generator(x, 1)

def basis2(x):
    return basis_generator(x, 2)

def basis3(x):
    return basis_generator(x, 5)

class LogisticRegressor:
    def __init__(self, eta, runs):
        self.eta = eta
        self.runs = runs

    def __step(self, x, y):
        z = np.dot(x, self.W)
        gradient = -x * self.__sigmoid(z) * (y * (np.exp(-z) + 1) - 1)
        self.W -= self.eta * np.mean(gradient, axis=0).reshape(self.W.shape)

    def fit(self, x, y, w_init=None):
        if w_init is not None:
            self.W = w_init
        else:
            self.W = np.random.rand(x.shape[1], 1)
        for _ in range(self.runs):
            self.__step(x, y)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        return self.__sigmoid(np.dot(x, self.W))

# Function to visualize prediction lines
# Takes as input last_x, last_y, [list of models], basis function, title
# last_x and last_y should specifically be the dataset that the last model
# in [list of models] was trained on
def visualize_prediction_lines(last_x, last_y, models, basis, title):
    # Plot setup
    green = mpatches.Patch(color='green', label='Ground truth model')
    black = mpatches.Patch(color='black', label='Mean of learned models')
    purple = mpatches.Patch(color='purple', label='Model learned from displayed dataset')
    plt.legend(handles=[green, black, purple], loc='upper right')
    plt.title(title)
    plt.xlabel('X Value')
    plt.ylabel('Y Label')
    plt.axis([-3, 3, -0.1, 1.1]) # Plot ranges

    # Plot dataset that last model in models (models[-1]) was trained on
    cmap = c.ListedColormap(['r', 'b'])
    plt.scatter(last_x, last_y, c=last_y, cmap=cmap, linewidths=1, edgecolors='black')

    # Plot models
    X_pred = np.linspace(-3, 3, 1000)
    X_pred_transformed = basis(X_pred)

    ## Ground truth model
    plt.plot(X_pred, np.sin(1.2*X_pred) * 0.4 + 0.5, 'g', linewidth=5)

    ## Individual learned logistic regressor models
    Y_hats = []
    for i in range(len(models)):
        model = models[i]
        Y_hat = model.predict(X_pred_transformed)
        Y_hats.append(Y_hat)
        if i < len(models) - 1:
            plt.plot(X_pred, Y_hat, linewidth=.3)
        else:
            plt.plot(X_pred, Y_hat, 'purple', linewidth=3)

    # Mean / expectation of learned models over all datasets
    plt.plot(X_pred, np.mean(Y_hats, axis=0), 'k', linewidth=5)

    plt.savefig('plots/' + title + '.png')
    plt.show()

# Function to generate datasets from underlying distribution
def generate_data(dataset_size):
    x, y = [], []
    for _ in range(dataset_size):
        x_i = 6 * np.random.random() - 3
        p_i = np.sin(1.2*x_i) * 0.4 + 0.5
        y_i = np.random.binomial(1, p_i)
        x.append(x_i)
        y.append(y_i)
    return np.array(x), np.array(y).reshape(-1, 1)

if __name__ == "__main__":
    
    # DO NOT CHANGE THE SEED!
    np.random.seed(1738)
    eta = 0.001
    runs = 10000
    N = 30

    basis_titles = ["Basis 1", "Basis 2", "Basis 3"]
    bases = [basis1, basis2, basis3]
    for i in range(3):
        all_models = []
        for _ in range(10):
            x, y = generate_data(N)
            x_transformed = bases[i](x)
            model = LogisticRegressor(eta=eta, runs=runs)
            model.fit(x_transformed, y)
            all_models.append(model)
        # Here x and y contain last dataset:
        visualize_prediction_lines(x, y, all_models, bases[i], basis_titles[i])
