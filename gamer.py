import gym
import gym.spaces
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os
import time

# Use CPU for processing
os.environ['CUDA_VISIBLE_DEVICES'] = ''


class Gamer:

    def __init__(self, game_name, print_frequency=1, plot_frequency=50, video_frequency=100):
        self.game_name = game_name
        self.print_frequency = print_frequency
        self.plot_frequency = plot_frequency
        self.video_frequency = video_frequency
        self.env = gym.make(self.game_name)
        self.reward_history = []
        self.mean_reward_history = []
        self.start_time = time.asctime().replace(' ', '_').replace(':', '-')
        self.game_id = 1

    def connect_model(self, model):
        model.n_actions = self.env.action_space.n
        self.model = model
        self.model_name = model.__class__.__name__
        self.env = gym.wrappers.Monitor(self.env, 'video/{0}_{1}_{2}'. \
                               format(self.model_name, self.game_name, self.start_time),
                               video_callable=lambda episode_id: episode_id % self.video_frequency == 0,
                               force=True, uid='{0}_{1}'.format(self.model_name, self.game_name))

    def save_statistics(self, reward):
        self.reward_history.append(reward)
        if self.game_id == 1:
            mean_reward = reward
            self.mean_reward_history.append(mean_reward)
        else:
            mean_reward = self.mean_reward_history[-1] * 0.99 + reward * 0.01
            self.mean_reward_history.append(mean_reward)
        
        if self.game_id % self.print_frequency == 0:
            print('Game: {0}, Number: {1}, Reward: {2}, Average reward: {3}, Model: {4}'.\
                  format(self.game_name, self.game_id, reward, mean_reward, self.model_name))
        if self.game_id % self.plot_frequency == 0:
            self.plot_statistics()

        self.game_id += 1

    def plot_statistics(self):
        if not os.path.exists('images'):
            os.mkdir('images')
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(self.reward_history)
        plt.title('{0} - {1}'.format(self.model_name, self.game_name))
        plt.subplot(212)
        plt.plot(self.mean_reward_history)
        fig.savefig('images/{0}_{1}_{2}_history.png'.format(self.model_name, self.game_name, self.start_time))
        plt.close()

    def play_game(self):
        observation = self.env.reset()
        done = False
        reward_per_game = 0

        while not done:
            action = self.model.predict_action(observation)
            observation, reward, done, info = self.env.step(action)
            self.model.get_step_results(observation, reward, done, info)
            reward_per_game += reward

        self.save_statistics(reward_per_game)

