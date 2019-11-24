import tensorflow as tf

class ArgMaxPolicy(object):
    # TODO: Adapt this class
    def __init__(self, critic):
        self.critic = critic

        self.action = tf.argmax(self.critic.q_t_values, axis=1)

    def get_action(self, obs):

        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        return self.critic.sess.run(self.action, feed_dict={self.critic.obs_t_ph: observation})