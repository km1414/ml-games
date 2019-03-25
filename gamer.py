import gym
import gym.spaces
from models import RandomModel, PolicyGradientsModel
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os


os.environ['CUDA_VISIBLE_DEVICES'] = ''




class Gamer:

    def __init__(self, game_name, video_frequency=None):
        self.game_name = game_name
        self.video_frequency = video_frequency
        self.env = gym.make(self.game_name)
        self.reward_history = []
        self.mean_reward_history = []
        self.game_id = 1

    def connect_model(self, model):
        model.n_actions = self.env.action_space.n
        self.model = model
        self.model_name = model.__class__.__name__
        if self.video_frequency is not None:
            self.env = gym.wrappers.Monitor(self.env, 'video/{0}_{1}'.format(self.model_name, self.game_name),
                               video_callable=lambda episode_id: episode_id % self.video_frequency == 0,
                               force=True, uid='{0}_{1}'.format(self.model_name, self.game_name))

    def save_statistics(self, reward, print_frequency=1, plot_frequency=50):
        self.reward_history.append(reward)
        if self.game_id == 1:
            mean_reward = reward
            self.mean_reward_history.append(mean_reward)
        else:
            mean_reward = self.mean_reward_history[-1] * 0.99 + reward * 0.01
            self.mean_reward_history.append(mean_reward)
        self.game_id += 1
        
        if self.game_id % print_frequency == 0:
            print('Game number: {0}, Game reward: {1}, Average reward: {2}, Model: {3}'.\
                  format(self.game_id, reward, mean_reward, self.model_name))
        if self.game_id % plot_frequency == 0:
            self.save_plot()

    def save_plot(self):
        if not os.path.exists('images'):
            os.mkdir('images')
        fig = plt.figure()
        plt.subplot(211)
        plt.plot(self.reward_history)
        plt.title('{0} - {1}'.format(self.model_name, self.game_name))
        plt.subplot(212)
        plt.plot(self.mean_reward_history)
        fig.savefig('images/{0}_{1}_history.png'.format(self.model_name, self.game_name))
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






# Test
controller = Gamer('Breakout-v0')
model = RandomModel()
controller.connect_model(model)
for i in range(10000):
    controller.play_game()



controller = Gamer('Breakout-v0', video_frequency=100)
model = PolicyGradientsModel()
controller.connect_model(model)
for i in range(10000):
    controller.play_game()







