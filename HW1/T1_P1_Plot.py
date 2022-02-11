import numpy as np
import matplotlib.pyplot as plt
from T1_P1 import data, predict


x_data = np.array([x for x,_ in data])
y_data = np.array([y for _,y in data])
x_vals = np.arange(0, 12.1, 0.1)
y_vals = dict()
for tau in [0.01, 2, 100]:
    y_vals[tau] = predict(x_data, y_data, tau, x_vals, ignore_same = False)


plt.xlim([0, 12])
plt.scatter(x_data, y_data, label="kernel data", color = 'green')
plt.plot(x_vals, y_vals[0.01], label = "tau = 0.01", color = 'black')
plt.plot(x_vals, y_vals[2], label = "tau = 2", color = 'blue')
plt.plot(x_vals, y_vals[100], label = "tau = 100", color = 'red')
plt.legend()
plt.title("Kernel Predictions with Various Lengthscales ")
plt.show()