import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gym

class Agent():
    def __init__(self, env):
        #if discrete
        self.env = env
        self.isdiscrete = isinstance(env.action_space, gym.spaces.Discrete)
        if self.isdiscrete:
            self.action_space_dim = env.action_space.n
        else:
            self.action_space_dim = env.action_space.shape[0]

        self.optimizer = tf.keras.optimizers.Adam()

        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential()
        model.add(layers.Dense(400, activation = 'relu'))
        model.add(layers.Dense(300, activation = 'relu'))
        model.add(layers.Dense(self.action_space_dim))
        return model

    def loss(self, obss, acs):
        return tf.keras.losses.MSE(self.model(obss), acs)

    @tf.function
    def train(self, obss, acs):
        with tf.GradientTape() as gt:
            loss = self.loss(obss,acs)
        grads = gt.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def sample_trajectories(self, model, env, steps = 5000):
        initial = True
        obs = env.reset()

        for step in range(steps):

            ac = model(np.expand_dims(obs, axis=0))
            pair = np.concatenate([obs, ac[0]], axis=0)
            pair = np.expand_dims(pair, axis=0)
            if initial:
                initial = False
                result = pair
            else:
                result = np.vstack([result, pair])

            n_obs, rew, done, info = env.step(ac[0])

            if done:
                obs = env.reset()
            else:
                obs = n_obs

        return result
