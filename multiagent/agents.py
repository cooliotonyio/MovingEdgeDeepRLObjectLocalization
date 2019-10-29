import numpy as np
import tensorflow as tf

class DefaultAgent():
    def __init__(self):
        #TODO
        self.replay_buffer = None

    def train(self):
        raise NotImplementedError()

    def add_to_replay_buffer(self, paths):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError
