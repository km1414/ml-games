import numpy as np
import tensorflow as tf
import scipy


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


class PolicyGradientsModel:
    """
    Policy Gradients model.
    https://medium.com/@jonathan_hui/rl-policy-gradients-explained-9b13b688b146

    Resize observations to (observation_size, 1) arrays.
    """
    def __init__(self):
        self.observation_size = 1024
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.train_frequency = 1
        self.prev_x = None
        self.prev_diff_x = None
        self.game_id = 1
        self.step_id = 1
        self.reset_games_data()
        self.n_actions = None

    def build_neural_network(self):
        x = tf.placeholder(dtype=tf.float32, shape=[None, self.observation_size])
        y = tf.placeholder(dtype=tf.float32, shape=[None, self.n_actions])
        weights = tf.Variable(tf.random_normal([self.observation_size, self.n_actions]))
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
                print('Training model.')
                self.train_model()
            self.game_id += 1
            self.step_id = 1

    def predict_action(self, observation):

        if self.n_actions is None:
            print('Model not connected!')

        if self.game_id == 1 and self.step_id == 1:
            self.build_neural_network()

        # getting action from NN
        self.x_cur, self.prev_x, self.prev_diff_x = self.prepare_obs(observation, self.prev_x, self.prev_diff_x)
        self.processed_observation = self.x_cur
        feed = {self.x: np.reshape(self.x_cur, (1, -1))}
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

    # resize observation to 2D and smaller size
    def resize_obs(self, image):
        image = image.reshape(1, -1)
        image = scipy.misc.imresize(image, size=(1, self.observation_size))
        image = (image / 255).ravel()
        return image

    # merge observation with previous observations
    def prepare_obs(self, observation, prev_x, prev_diff_x):
        cur_x = self.resize_obs(observation)
        if prev_x is None: prev_x = cur_x
        diff_x = np.zeros_like(cur_x)
        diff_x[cur_x != prev_x] = 1
        if prev_diff_x is None: prev_diff_x = np.zeros_like(diff_x)
        x = prev_diff_x * 0.8 + diff_x
        x[x > 1] = 1
        prev_x = cur_x
        prev_diff_x = x
        return x, prev_x, prev_diff_x

    # discount rewards
    def discount_rewards(self, x):
        x = np.array(x).astype(float)
        for i in reversed(range(len(x) - 1)):
            if x[i] == 0: x[i] = x[i + 1] * self.gamma
        x = x.reshape((len(x), 1))
        return x