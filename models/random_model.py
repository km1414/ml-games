import numpy as np


class RandomModel:

    """
    Baseline random model.
    Predict random actions all the time.
    """

    def __init__(self):
        self.n_actions = None

    def predict_action(self, observation):
        if self.n_actions is None:
            print('Model not connected!')
        action = np.random.randint(self.n_actions)
        return action

    def get_step_results(self, observation, reward, done, info):
        pass


