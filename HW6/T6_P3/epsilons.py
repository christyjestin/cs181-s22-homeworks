from stub import Learner, run_games
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10

epsilons = np.array([0, 0.00001, 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.01, 0.05])
n = epsilons.shape[0]
max_vals = np.zeros(n)
mean_vals = np.zeros(n)
for i in range(n):
    trial_maxes = np.zeros(NUM_TRIALS)
    trial_means = np.zeros(NUM_TRIALS)
    for j in range(NUM_TRIALS):
        agent = Learner(epsilon = epsilons[i])
        hist = []
        run_games(agent, hist, 100, 100)
        hist = np.array(hist)
        trial_maxes[j] = np.max(hist[50:])
        trial_means[j] = np.mean(hist[50:])
    max_vals[i] = np.mean(trial_maxes)
    mean_vals[i] = np.mean(trial_means)

plt.plot(epsilons, mean_vals, label='Mean Scores')
plt.ylabel('Score')
plt.xlabel('epsilon')
plt.title('Model Performance with Different Exploration Rates')
plt.legend()
plt.savefig('../plots/epsilons_mean_score.png')
plt.clf()
plt.plot(epsilons, max_vals, label='Max Score')
plt.ylabel('Score')
plt.xlabel('epsilon')
plt.title('Model Performance with Different Exploration Rates')
plt.legend()
plt.savefig('../plots/epsilons_max_score.png')