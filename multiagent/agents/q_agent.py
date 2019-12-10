import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

from multiagent.util.dqn_utils import huber_loss, ReplayBuffer

import numpy as np


class DQN_Model(Model):
    def __init__(self):
        super(DQN_Agent, self).__init__()
        self.d1 = Dense(1024, activation = "relu")
        self.d2 = Dense(1024, activation = "relu")
        self.final = Dense(9, activation = "softmax")
    
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.final(x)
        return x
    
class DQN_Agent():
    def __init__(self, params, agent_params):
        self.q_func = DQN_Model()
        self.t = 0

        self.optimizer = params["optimizer"]
        self.loss = params["loss"]
        self.env = params["env"]

        self.learning_freq = agent_params["learning_freq"]
        self.gamma = agent_params["gamma"]
        self.epsilon = agent_params["epsilon"]
        self.batch_size = agent_params["batch_size"]
        self.replay_buffer = ReplayBuffer(agent_params["replay_buffer_size"])

        self.train_loss_metric = agent_params["train_loss_metric"]

    @tf.function
    def step(self, mode="train"):
        """Epsilon-greedy step that gets added to replay buffer"""
        if mode == "train" and np.random.random() < self.epsilon(self.t):
            acn = self.env.get_random_expert_action()
            self.t += 1
        else:
            # Get obs and calculate q_vals
            obs = self.env.get_obs()
            q_vals = self.q_func.call(obs)

            # Select an action (based on arg_max)
            action = tf.math.arg_max(q_vals, axis=1)
            
            # Execute action
            next_obs, rew, done = self.env.step(acn)
            
            if mode == "train":
                self.t += 1
        
        return obs, acs, rew, next_obs, done

    @tf.function
    def train(batch_size = self.batch_size)
        obs, acs, rew, next_obs, done = self.replay_buffer.sample_random_data(batch_size)

        with tf.GradientTape() as tape:
            pred_q_val = tf.reduce_sum(self.q_func.call(obs) * tf.one_hot(acs, self.ac_dim), axis = 1)
            next_q_val = tf.math.reduce_max(self.q_func.call(next_obs), axis = 1)
            target_q_val = rew + (tf.ones(done.shape) - done) * (self.gamma * next_q_val)

            total_loss = tf.math.reduce_mean(self.loss(target_q_val, pred_q_val))

        gradients = tape.gradient(total_loss, self.q_func.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_func.trainable_variables))

        self.train_loss_metric(total_loss)
        return total_loss

    
