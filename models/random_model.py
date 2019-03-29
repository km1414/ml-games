import numpy as np


class RandomModel:

    """
    Baseline random model.
    Predict random actions all the time.
    """

    def __init__(self):
        pass


    def get_env_features(self, gamer_obj):
        """
        Get mandatory game features from initialized Gamer object.
        Method is called by Gamer.

        Args:
            gamer_obj (object): initialized Gamer object
        """
        self.n_actions = gamer_obj.env.action_space.n


    def predict_action(self, observation):
        """
        Get observation from env and return action to Gamer.

        Args:
            observation (array): current env state (3D)
        """
        if self.n_actions is None:
            print('Model not connected!')
        action = np.random.randint(self.n_actions)
        return action


    def get_step_results(self, observation, reward, done, info):
        """
        Get results after predicted action done.

        Args:
            observation (array): current env state (3D array)
            reward (int/float): reward value from step
            done (bool): if game finished
            info (dict): additional game info
        """
        pass


