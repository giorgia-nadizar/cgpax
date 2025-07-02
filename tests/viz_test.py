import random

import gym
import numpy as np

if __name__ == '__main__':

    env = gym.make("Acrobot-v1")
    observation = env.reset()
    cum_reward = 0
    for _ in range(500):
        # _, _, alpha, omega = observation
        # action = 0 if (-alpha >= 0.1 * omega) else 1
        action = 2 if observation[5] > 0 else 0
        observation, reward, _, _ = env.step(action)
        cum_reward += reward

    print(cum_reward)
    env.close()
