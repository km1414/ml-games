import tensorflow as tf
import numpy as np
import scipy


class PolicyGradientsModel:
    """
    Policy Gradients model.
    https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146

    Resize observations to (observation_size, 1) arrays.
    """
    def __init__(self):
        self.observation_size = 4096
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.train_frequency = 1
        self.previous_observation = None
        self.difference_observation = None
        self.n_actions = None
        self.reset_games_data()
        self.game_id = 1
        self.step_id = 1

    def build_neural_network(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])
        weights = tf.Variable(tf.random_uniform([self.observation_size, self.n_actions],
                                                minval = -0.1, maxval = 0.1))
        predictions = tf.nn.softmax(tf.matmul(x, weights))


        discounted_rewards = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        mean, variance = tf.nn.moments(discounted_rewards, [0], shift=None)
        gradient_loss = (discounted_rewards - mean) / tf.sqrt(variance + 1e-6)

        cost = tf.nn.l2_loss(y - predictions)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        grads = optimizer.compute_gradients(cost, var_list=tf.trainable_variables(), grad_loss=gradient_loss)
        train_step = optimizer.apply_gradients(grads)
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        self.x = x
        self.y = y
        self.predictions = predictions
        self.discounted_rewards = discounted_rewards
        self.train_step = train_step
        self.sess = sess

    def reset_games_data(self):
        self.games_data = {
                            'observation': [],
                            'action': [],
                            'reward': [],
                            'done': [],
                            'info': [],
                          }

    def get_step_results(self, observation, reward, done, info):
        self.games_data['observation'].append(self.processed_observation)
        self.games_data['action'].append(self.label)
        self.games_data['reward'].append(reward)
        self.games_data['done'].append(done)
        self.games_data['info'].append(info)

        if done:
            if self.game_id % self.train_frequency == 0:
                self.train_model()
            self.game_id += 1
            self.step_id = 1

    def predict_action(self, observation):

        if self.n_actions is None:
            print('Model not connected!')

        if self.game_id == 1 and self.step_id == 1:
            self.build_neural_network()

        # getting action from NN
        self.processed_observation = self.augment_observation(observation)
        self.processed_observation = self.processed_observation
        feed = {self.x: np.reshape(self.processed_observation, (1, -1))}
        pred = self.sess.run(self.predictions, feed)[0, :]
        self.action = np.random.choice(self.n_actions, p=pred)
        self.label = np.zeros_like(pred)
        self.label[self.action] = 1

        self.step_id += 1

        return self.action

    def train_model(self):
        feed = {self.x: np.vstack(self.games_data['observation']),
                self.discounted_rewards: np.vstack(self.discount_rewards(self.games_data['reward'])),
                self.y: np.vstack(self.games_data['action'])}
        _ = self.sess.run(self.train_step, feed)
        self.reset_games_data()

    # merge observation with previous observations
    def augment_observation(self, observation):

        # Resize observation
        observation = observation.reshape(1, -1)
        observation = scipy.misc.imresize(observation, size=(1, self.observation_size))
        observation = (observation / 255).ravel()

        # Augment observation
        if self.previous_observation is None: 
            self.previous_observation = observation
        diff = np.zeros_like(observation)
        diff[observation != self.previous_observation] = 1
        if self.difference_observation is None: 
            self.difference_observation = np.zeros_like(diff)
        output = self.difference_observation * 0.8 + diff
        output[output > 1] = 1
        self.previous_observation = observation
        self.difference_observation = output
        return output
    
    # discount rewards
    def discount_rewards(self, x):
        x = np.array(x).astype(float)
        for i in reversed(range(len(x) - 1)):
            if x[i] == 0: x[i] = x[i + 1] * self.gamma
        x = x.reshape((len(x), 1))
        return x