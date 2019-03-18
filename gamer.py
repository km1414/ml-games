import gym
import gym.spaces
from models import RandomModel



class Gamer:

    def __init__(self, game_name):
        self.env = gym.make(game_name)
        self.n_actions = self.env.action_space.n
        self.reward_history = []

    def get_action(self, observation, solver):
            return solver.predict_action(observation)

    def play_game(self, solver):
        observation_before = self.env.reset()
        done = False
        reward_sum = 0
        output = []

        while not done:
            action = self.get_action(observation_before, solver)
            observation_after, reward, done, info =  self.env.step(action)
            reward_sum += reward
            output.append((observation_before, reward, reward_sum, done))
            observation_before = observation_after

        return output

    def play_multiple_games(self, n_games, solver):
        games_data = [self.play_game(solver) for x in n_games * [0]]
        return [item for sublist in games_data for item in sublist]




# Test
g = Gamer('Breakout-v0')
r = RandomModel(n_actions=g.n_actions)

for i in range(10):
    print('Running iter:', i)
    games_data = g.play_multiple_games(n_games=10, solver=r)
    r.train_model(games_data)



