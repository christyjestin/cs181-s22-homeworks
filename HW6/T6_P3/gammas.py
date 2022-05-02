from stub import Learner, run_games
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 10

gammas = np.arange(10) * 0.03 + 0.7
n = gammas.shape[0]
max_vals = np.zeros(n)
mean_vals = np.zeros(n)
for i in range(n):
    trial_maxes = np.zeros(NUM_TRIALS)
    trial_means = np.zeros(NUM_TRIALS)
    for j in range(NUM_TRIALS):
        agent = Learner(gamma = gammas[i])
        hist = []
        run_games(agent, hist, 100, 100)
        hist = np.array(hist)
        trial_maxes[j] = np.max(hist[50:])
        trial_means[j] = np.mean(hist[50:])
    max_vals[i] = np.mean(trial_maxes)
    mean_vals[i] = np.mean(trial_means)

plt.plot(gammas, mean_vals, label='Mean Scores')
plt.ylabel('Score')
plt.xlabel('gamma')
plt.title('Model Performance with Different Discount Factors')
plt.legend()
plt.savefig('../plots/gammas_mean_score.png')
plt.clf()
plt.plot(gammas, max_vals, label='Max Score')
plt.ylabel('Score')
plt.xlabel('gamma')
plt.title('Model Performance with Different Discount Factors')
plt.legend()
plt.savefig('../plots/gammas_max_score.png')