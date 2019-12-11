import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from multiagent.util.dqn_utils import huber_loss, ReplayBuffer

import numpy as np


class DQN_Model(Model):
    def __init__(self):
        """Fully connected Neural Network that learns to approximate a q function"""
        super(DQN_Model, self).__init__()
        self.d1 = Dense(1024, activation = "relu")
        self.d2 = Dense(1024, activation = "relu")
        self.final = Dense(9, activation = "softmax")
    
    def call(self, x):
        """Forward pass through neural network"""
        x = self.d1(x)
        x = self.d2(x)
        x = self.final(x)
        return x
    
class DQN_Agent():
    def __init__(self, params, agent_params):
        """Agent that uses a neural network to approximate a q function"""
        self.q_func = DQN_Model()
        self.t = 0

        self.optimizer = params["optimizer"]
        self.loss = params["loss"]
        self.env = params["env"]
        self.ac_dim = self.env.get_ac_dim()

        self.gamma = agent_params["gamma"]
        self.epsilon = agent_params["epsilon"]
        self.batch_size = agent_params["batch_size"]
        self.replay_buffer = ReplayBuffer(agent_params["replay_buffer_size"])

        self.train_loss_metric = agent_params["train_loss_metric"]

    def step(self, mode="train"):
        """The agent takes one step through env. When in train mode, will use epsilon-greedy and increment self.t"""

        # Get obs
        obs = self.env.get_env_state()

        # epsilon-greedy
        if mode == "train" and np.random.random() < self.epsilon.value(self.t):
            acs = self.env.get_random_expert_action()
            self.t += 1
        else:
            q_vals = self.q_func.call(obs)
            acs = tf.one_hot(tf.math.argmax(q_vals, axis=1), self.ac_dim)

        # Execute action
        next_obs, rew, done = self.env.step(acs)
        acs = tf.reshape(acs, (1, self.ac_dim))
        if mode == "train":
            self.t += 1
        
        return obs, acs, rew, next_obs, done

    @tf.function
    def train(self, batch_size = None):
        """Randomly samples 'batch_size' rollouts from self.replay_buffer and updates q_func via gradient descent"""
        if batch_size is None:
            batch_size = self.batch_size

        obs, acs, rew, next_obs, done = self.replay_buffer.sample_random_data(batch_size)

        with tf.GradientTape() as tape:
            pred_q_val = tf.reduce_sum(self.q_func.call(obs) * acs, axis = 1)
            next_q_val = tf.math.reduce_max(self.q_func.call(next_obs), axis = 1)
            target_q_val = rew + (tf.ones(done.shape) - done) * (self.gamma * next_q_val)

            total_loss = tf.math.reduce_mean(self.loss(target_q_val, pred_q_val))

        gradients = tape.gradient(total_loss, self.q_func.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_func.trainable_variables))

        self.train_loss_metric(total_loss)
        return total_loss

    def add_to_replay_buffer(self, rollouts):
        """Adds rollouts to self.replay_buffer"""
        self.replay_buffer.add_rollouts(rollouts)
    
    def can_sample_replay_buffer(self, batch_size = None):
        """Returns true if self.replay_buffer contains at least batch_size rollouts"""
        if batch_size is None:
            batch_size = self.batch_size
        return self.replay_buffer.can_sample(batch_size)
