import numpy as np


class RandomModel:

    def __init__(self, n_actions):
        self.n_actions = n_actions

    def predict_action(self, observation):
        return np.random.randint(self.n_actions)

    def train_model(self,data):
        pass