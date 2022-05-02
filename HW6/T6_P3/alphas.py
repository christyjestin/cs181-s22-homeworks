from stub import Learner, run_games
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10

alphas = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.25, 0.3, 0.5, 0.7])
n = alphas.shape[0]
max_vals = np.zeros(n)
mean_vals = np.zeros(n)
for i in range(n):
    trial_maxes = np.zeros(NUM_TRIALS)
    trial_means = np.zeros(NUM_TRIALS)
    for j in range(NUM_TRIALS):
        agent = Learner(alpha = alphas[i])
        hist = []
        run_games(agent, hist, 100, 100)
        hist = np.array(hist)
        trial_maxes[j] = np.max(hist[50:])
        trial_means[j] = np.mean(hist[50:])
    max_vals[i] = np.mean(trial_maxes)
    mean_vals[i] = np.mean(trial_means)

plt.plot(alphas, mean_vals, label='Mean Scores')
plt.ylabel('Score')
plt.xlabel('alpha')
plt.title('Model Performance with Different Learning Rates')
plt.legend()
plt.savefig('../plots/alphas_mean_score.png')
plt.clf()
plt.plot(alphas, max_vals, label='Max Score')
plt.ylabel('Score')
plt.xlabel('alpha')
plt.title('Model Performance with Different Learning Rates')
plt.legend()
plt.savefig('../plots/alphas_max_score.png')