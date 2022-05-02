from epsilon_decay_stub import Learner, run_games
import matplotlib.pyplot as plt
import numpy as np

NUM_TRIALS = 1

starting_vals = [0.001, 0.01]
#, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]
decay_rates = np.arange(5) / 5
n = decay_rates.shape[0]
max_vals = [np.zeros(n)] * len(starting_vals)
mean_vals = [np.zeros(n)] * len(starting_vals)
for svi in range(len(starting_vals)):
    for i in range(n):
        print(i)
        trial_maxes = np.zeros(NUM_TRIALS)
        trial_means = np.zeros(NUM_TRIALS)
        for j in range(NUM_TRIALS):
            agent = Learner(decay_rate = decay_rates[i], starting_epsilon = starting_vals[svi])
            hist = []
            run_games(agent, hist, 100, 1)
            hist = np.array(hist)
            trial_maxes[j] = np.max(hist[50:])
            trial_means[j] = np.mean(hist[50:])
        max_vals[svi][i] = np.mean(trial_maxes)
        mean_vals[svi][i] = np.mean(trial_means)

plt.plot(decay_rates, mean_vals[0], label='starting val=' + str(starting_vals[0]))
plt.plot(decay_rates, mean_vals[1], label='starting val=' + str(starting_vals[1]))
plt.ylabel('Score')
plt.xlabel('Decay Rate')
plt.title('Model Performance with Different Epsilon Decay Rates (Mean Scores)')
plt.legend()
plt.savefig('../plots/epsilon_decays_mean_score1.png')
plt.clf()
plt.plot(decay_rates, max_vals[0], label='starting val=' + str(starting_vals[0]))
plt.plot(decay_rates, max_vals[1], label='starting val=' + str(starting_vals[1]))
plt.ylabel('Score')
plt.xlabel('Decay Rate')
plt.title('Model Performance with Different Epsilon Decay Rates (Max Scores)')
plt.legend()
plt.savefig('../plots/epsilon_decays_max_score1.png')