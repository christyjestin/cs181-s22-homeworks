import numpy as np
import numpy.random as npr
import pygame as pg

# uncomment this for animation
# from SwingyMonkey import SwingyMonkey

# uncomment this for no animation
from SwingyMonkeyNoAnimation import SwingyMonkey


X_BINSIZE = 200
Y_BINSIZE = 100
X_SCREEN = 1400
Y_SCREEN = 900


class Learner(object):
    """
    This agent jumps randomly.
    """

    def __init__(self, alpha=0.08, gamma=0.88, starting_epsilon=0.8, decay_rate=0.01):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

        # Q learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.curr_epsilon = starting_epsilon
        self.decay_rate = decay_rate
        # We initialize our Q-value grid that has an entry for each action and state.
        # (action, rel_x, rel_y)
        self.Q = np.zeros((2, X_SCREEN // X_BINSIZE, Y_SCREEN // Y_BINSIZE))

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None

    def discretize_state(self, state):
        """
        Discretize the position space to produce binned features.
        rel_x = the binned relative horizontal distance between the monkey and the tree
        rel_y = the binned relative vertical distance between the monkey and the tree
        """

        rel_x = int((state["tree"]["dist"]) // X_BINSIZE)
        rel_y = int((state["tree"]["top"] - state["monkey"]["top"]) // Y_BINSIZE)
        return (rel_x, rel_y)

    def action_callback(self, s):
        """
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        """

        if self.last_state == None:
            self.last_state = s
            self.last_action = int(npr.rand() < 0.5)
            return self.last_action

        a = self.last_action
        r = self.last_reward
        old_x, old_y = self.discretize_state(self.last_state)
        curr_x, curr_y = self.discretize_state(s)
        q_val = self.Q[a, old_x, old_y]
        best_q = np.max(self.Q[:, curr_x, curr_y])
        self.Q[a, old_x, old_y] = (1 - self.alpha) * q_val + self.alpha * (r + self.gamma * best_q)

        rand_move = int(npr.rand() < 0.5)
        new_action = np.argmax(self.Q[:, curr_x, curr_y]) if npr.rand() > self.curr_epsilon else rand_move
        self.last_action = new_action
        self.last_state = s
        self.curr_epsilon *= self.decay_rate
        return self.last_action

    def reward_callback(self, reward):
        """This gets called so you can see what reward you get."""

        self.last_reward = reward


def run_games(learner, hist, iters=100, t_len=100):
    """
    Driver function to simulate learning by having the agent play a sequence of games.
    """
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,  # Don't play sounds.
                             text="Epoch %d" % (ii),  # Display the epoch on screen.
                             tick_length=t_len,  # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':
    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games. You can update t_len to be smaller to run it faster.
    run_games(agent, hist, 100, 100)
    print(hist)
    hist = np.array(hist)
    print('mean: ', np.mean(hist))
    print('max: ', np.max(hist))

    # Save history
    np.save('hist', hist)