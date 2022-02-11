import numpy as np

data = [(0., 0.),
        (1., 0.5),
        (2., 1.),
        (3., 2.),
        (4., 1.),
        (6., 1.5),
        (8., 0.5)]

def predict(x, y, tau, inputs, ignore_same = True):
    output = np.ones(inputs.shape)
    for i in range(output.shape[0]):
        similarity = (x - np.repeat(inputs[i], x.shape[0])) ** 2
        kernel = np.exp(-similarity / tau)
        # we don't want to use y_i to predict yhat_i, so we'll ignore y_i if x* == x_i
        if ignore_same:
            kernel[similarity == 0] = 0
        output[i] = kernel @ y
    return output

def compute_loss(tau):
    x = np.array([x for x,_ in data])
    y = np.array([y for _,y in data])
    error = predict(x, y, tau, x) - y
    return np.sum(error ** 2)

if __name__ == "__main__":
    for tau in (0.01, 2, 100):
        print("Loss for tau = " + str(tau) + ": " + str(compute_loss(tau)))