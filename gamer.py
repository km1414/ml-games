import numpy as np
import gym
import gym.spaces
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import time


# Use CPU for processing
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class Gamer:

    """
    This is the class for running the games, tracking statistics,
    saving performance plots and saving game play videos.

    Args:
        game_name (str): env name from Atari gym (Breakout-v0, Pong-v0, ...)
        statistics_update_frequency (int): when to plot and record video.
    """

    def __init__(self, game_name, statistics_update_frequency=100):
        self.game_name = game_name
        self.statistics_update_frequency = statistics_update_frequency
        self.env = gym.make(self.game_name)
        self.reward_history = []
        self.mean_reward_history = []
        self.start_time = time.asctime().replace(' ', '_').replace(':', '-')
        self.game_id = 1

    def connect_model(self, model):
        """
        Connects Gamer object and Model object in order to share env info.
        Also adds game recording feature to self.env.

        Args:
            model (object): initialized decision model.
        """
        self.model = model
        self.model.get_env_features(self)
        self.model_name = model.__class__.__name__
        self.env = gym.wrappers.Monitor(self.env, 'videos/{0}_{1}_{2}'. \
                               format(self.model_name, self.game_name, self.start_time),
                               video_callable=lambda episode_id: episode_id % self.statistics_update_frequency == 0,
                               force=True, uid='{0}_{1}'.format(self.model_name, self.game_name))

    def track_statistics(self, reward):
        """
        Tracks game history and updates with new step data.
        Also prints results to the screen and plots history to image.

        Args:
            reward (int/float): reward value after action done.
        """
        self.reward_history.append(reward)
        if self.game_id == 1:
            mean_reward = reward
            self.mean_reward_history.append(mean_reward)
        else:
            mean_reward = np.around(self.mean_reward_history[-1] * 0.99 + reward * 0.01, decimals=3)
            self.mean_reward_history.append(mean_reward)

        if self.game_id % self.statistics_update_frequency == 0:
            self.plot_statistics()
            print('Statistics saved to /images')
            print('Game recording saved to /videos')

        print('Game: {0}, Number: {1}, Reward: {2}, Average reward: {3}, Model: {4}, Started: {5}'. \
              format(self.game_name, self.game_id, reward, mean_reward, self.model_name, self.start_time))

        self.game_id += 1

    def plot_statistics(self):
        """
        Plot to game history to folder images/.
        """
        if not os.path.exists('images'):
            os.mkdir('images')
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(self.reward_history, color='b')
        plt.title('{0} - {1}'.format(self.model_name, self.game_name))
        plt.legend(['Reward'], loc='upper left', facecolor='w')
        plt.subplot(212)
        plt.plot(self.mean_reward_history, color='g')
        plt.legend(['Average reward'], loc='upper left', facecolor='w')
        fig.savefig('images/{0}_{1}_{2}_history.png'.\
                    format(self.model_name, self.game_name, self.start_time), dpi=400)
        plt.close()

    def play_game(self):
        """
        Run the env and reset it when game finished.
        """
        observation = self.env.reset()
        done = False
        reward_per_game = 0

        while not done:
            action = self.model.predict_action(observation)
            observation, reward, done, info = self.env.step(action)
            self.model.get_step_results(observation, reward, done, info)
            reward_per_game += reward

        self.track_statistics(reward_per_game)

