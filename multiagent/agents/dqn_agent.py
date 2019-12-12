import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from multiagent.util.dqn_utils import huber_loss, ReplayBuffer


def get_q_network(loss = "huber_loss", optimizer = None, dropout = 0.2):
    if optimizer is None:
        optimizer = Adam(lr=1e-6)

    model = Sequential()
    model.add(Dense(1024, input_shape=(4096 + 90,)))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(Dropout(dropout))
    model.add(Dense(9))
    
    model.compile(loss = loss, optimizer = optimizer)
    return model

class DQN_Agent():
    def __init__(self, params, agent_params):
        """Agent that uses a neural network to approximate a q function"""
        self.optimizer = params["optimizer"]
        self.loss = params["loss"]
        self.env = params["env"]
        self.ac_dim = self.env.get_ac_dim()

        self.gamma = agent_params["gamma"]
        self.epsilon = agent_params["epsilon"]
        self.batch_size = agent_params["batch_size"]
        self.replay_buffer = ReplayBuffer(agent_params["replay_buffer_size"])
        self.dropout = agent_params["dropout"]

        self.q_func = get_q_network(self.loss, self.optimizer, self.dropout)
        self.t = 0

    def get_action(self, obs, mode="train"):
        """ Returns action """
        if mode == "train" and np.random.random() < self.epsilon.value(self.t):
            acs = self.env.get_random_expert_action()
        else:
            q_vals = self._get_q_vals(obs)
            acs = tf.one_hot(tf.math.argmax(q_vals), self.ac_dim)
        return acs

    def _get_q_vals(self, obs):
        if len(obs.shape) == 1:
            return self.q_func.predict(tf.reshape(obs, (1, len(obs))))[0]
        return self.q_func.predict(obs)

    def step(self, mode="train"):
        """The agent takes one step through env. When in train mode, will use epsilon-greedy and increment self.t"""
        if mode == "train":
            self.t += 1

        # Get observation
        obs = self.env.get_env_state()
        # Get action
        acs = self.get_action(obs, mode = mode)
        # Execute action
        next_obs, rew, done = self.env.step(acs)
        
        return obs, acs, rew, next_obs, done

    def train(self, batch_size = None):
        """Updates q_func via gradient descent using sampled batch from replay_buffer"""

        if batch_size is None:
            batch_size = self.batch_size

        obs, acs, rew, next_obs, done = self.sample_replay_buffer(batch_size)

        pred_q_vals = self.q_func.predict(obs)
        target_q_vals = self._get_target_q_vals(pred_q_vals, acs, rew, next_obs, done, batch_size)

        self.q_func.fit(obs, target_q_vals, batch_size = batch_size, epochs = 1, verbose = 0)

        loss = self.loss(pred_q_vals, target_q_vals)
        return loss

    def _get_target_q_vals(self, pred_q_vals, acs, rew, next_obs, done, batch_size):
        """Returns target q_vals to train on"""
        q_vals = pred_q_vals.copy()
        update_targets = self._get_update_vals(rew, next_obs, done)
        idx_to_update = self._action_to_idx(acs)
        for i in np.arange(batch_size):
            q_vals[i][idx_to_update[i]] = update_targets[i]
        return q_vals

    def _get_update_vals(self, rew, next_obs, done):
        """
        Gets the target q_val from next state and reward of the specific action preceding next_obs
        
        Q_value target = reward + (1-done) * gamma * q value of next state
        """
        next_q_val = tf.math.reduce_max(self._get_q_vals(next_obs), axis = 1)
        done_mask = tf.reshape(tf.ones(done.shape) - done, next_q_val.shape)
        rew = tf.reshape(rew, next_q_val.shape)
        return rew + (done_mask * self.gamma * next_q_val)

    def _action_to_idx(self, acs):
        """Turns 9-dim action vector into scalar idx of action"""
        if len(acs.shape) == 1:
            return tf.math.argmax(acs)
        return tf.math.argmax(acs, axis = 1)

    def sample_replay_buffer(self, batch_size=None):
        """Randomly samples 'batch_size' rollouts from self.replay_buffer"""
        return self.replay_buffer.sample_random_data(batch_size)

    def add_to_replay_buffer(self, rollouts):
        """Adds rollouts to self.replay_buffer"""
        self.replay_buffer.add_rollouts(rollouts)
    
    def can_sample_replay_buffer(self, batch_size = None):
        """Returns true if self.replay_buffer contains at least batch_size rollouts"""
        if batch_size is None:
            batch_size = self.batch_size
        return self.replay_buffer.can_sample(batch_size)
