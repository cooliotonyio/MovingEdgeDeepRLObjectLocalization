import numpy as np
import tensorflow as tf

def huber_loss(x, delta=1.0):
    # https://en.wikipedia.org/wiki/Huber_loss
    return tf.where(
        tf.abs(x) < delta,
        tf.square(x) * 0.5,
        delta * (tf.abs(x) - 0.5 * delta)
    )

class ReplayBuffer(object):
    def __init__(self, size):
        """Replay buffer for agents"""
        self.size = size
        self.num_in_buffer = 0

        self.obs = None
        self.acs = None
        self.rew = None
        self.next_obs = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if buffer contains at least batch_size rollouts"""
        return batch_size < self.num_in_buffer

    def add_rollouts(self, rollouts):
        """Add rollouts to memory buffer"""
        obs = np.concatenate([r["obs"] for r in rollouts])
        acs = np.concatenate([r["acs"] for r in rollouts])
        rew = [r["rew"] for r in rollouts]
        next_obs = np.concatenate([r["next_obs"] for r in rollouts])
        done = [r["done"] for r in rollouts]

        if self.obs is None:
            self.obs = obs[-self.size:]
            self.acs = acs[-self.size:]
            self.rew = rew[-self.size:]
            self.next_obs = next_obs[-self.size:]
            self.done = done[-self.size:]
        else:
            self.obs = np.concatenate([self.obs, obs])[-self.size:]
            self.acs = np.concatenate([self.acs, acs])[-self.size:]
            self.rew = (self.rew + rew)[-self.size:]
            self.next_obs = np.concatenate([self.next_obs, next_obs])[-self.size:]
            self.done = (self.done + done)[-self.size:]

        assert self.obs.shape[0] == self.acs.shape[0] == len(self.rew) == self.next_obs.shape[0] == len(self.done)

        self.num_in_buffer = self.obs.shape[0]
    
    def sample_random_data(self, batch_size):
        """Samples batch_size rollouts randomly from buffer"""
        assert self.can_sample(batch_size)
        rand_indices = np.random.permutation(self.num_in_buffer)[:batch_size]
        return self.obs[rand_indices], self.acs[rand_indices], self.rew[rand_indices], self.next_obs[rand_indices], self.done[rand_indices]

    def reset(self):
        """Clears out buffer"""
        self.num_in_buffer = 0
        self.obs = None
        self.acs = None
        self.rew = None
        self.next_obs = None
        self.done = None

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """Returns epsilon value based on t"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)