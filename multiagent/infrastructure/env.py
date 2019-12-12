import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from multiagent.util.bbox import get_iou, draw_bbox

class ObjectLocalizationEnv():
    def __init__(
        self, 
        model, 
        model_input_dim, 
        transformation_factor = 0.2, 
        trigger_threshold = 0.6, 
        trigger_reward = 3,
        history_len = 10,
        feature_dim = 4096):
        """
        Environment in which the agent acts in
        """

        self.model = model
        self.model_input_dim = model_input_dim
        self.transformation_factor = transformation_factor
        self.trigger_threshold = trigger_threshold
        self.trigger_reward = trigger_reward
        self.history_len = history_len
        self.feature_dim = feature_dim

        self.actions = np.identity(9)


    def get_reward(self, old_bbox, action, new_bbox):
        """Returns reward, {-1,+1} for non-trigger actions, {-trigger_reward, +trigger_reward} for trigger action"""
        return tf.reshape(tf.cast(self._reward(old_bbox, action, new_bbox), tf.float32), (1,))
        
    def _reward(self, old_bbox, action, new_bbox):
        # non-trigger action reward
        if not action[8]:
            if get_iou(new_bbox, self.target_bbox) > get_iou(old_bbox,self.target_bbox):
                return 1
            return -1
        
        # trigger action reward
        if get_iou(new_bbox, self.target_bbox) > self.trigger_threshold:
            return self.trigger_reward
        return -self.trigger_reward
    
    def step(self, action):
        """
        Takes one step through the environment. Returns next_state, reward, and if the episode has terminated

        Actions: [
            0:right, 
            1:left, 
            2:up, 
            3:down, 
            4:bigger, 
            5:smaller, 
            6:fatter, 
            7:taller, 
            8:trigger]
        """
        if len(action.shape) == 2:
            action = action[0]

        old_bbox = self.obs_bbox.copy()
        self.obs_bbox, done = self._step(action, self.obs_bbox)

        self.history = tf.concat([
            self.history[1:], 
            tf.reshape(tf.cast(action, tf.float32), (1, len(self.actions)))], 
            axis=0)

        self._get_obs_feature()
        self._get_history_vector()

        obs = self.get_env_state()
        rew = self.get_reward(old_bbox, action, self.obs_bbox)
        
        return obs, rew, done
        
    def reset(self, target_bbox = None, image = None):
        """Resets the env. Resets bbox to entire image."""
        if target_bbox is not None:
            self.target_bbox = target_bbox
        if image is not None:
            self.image = image
            
        self.history = tf.zeros([self.history_len, len(self.actions)], dtype=tf.dtypes.float32)
        _, self.max_h, self.max_w, _ = self.image.shape
        self.obs_bbox = [0, 0, self.max_w, self.max_h] 
        self.obs_feature = self._get_obs_feature()
        self.history_vector = self._get_history_vector()

    def get_env_state(self):
        """The extracted feature vector of the bbox concatenated with history vector of actions"""
        return tf.concat((self.obs_feature, self.history_vector), axis=0)
    
    def show(self):
        """Draws both target bbox and current bbox in the image"""
        image = Image.fromarray(self.image.numpy()[0])
        # Draw current bbox in red
        image = draw_bbox(image, self.obs_bbox, fill="red")
        # Draw target bbox in green
        image = draw_bbox(image, self.target_bbox, fill="green")
        return image
    
    def get_ac_dim(self):
        """Returns the dimension of the action space (9)"""
        return len(self.actions)
    
    def _get_obs_feature(self):
        """Extracts feature vector from current bbox"""
        image = tf.image.crop_to_bounding_box(self.image, self.obs_bbox[1], self.obs_bbox[0], self.obs_bbox[3], self.obs_bbox[2])
        image = preprocess_input(tf.image.resize(image, self.model_input_dim), mode="tf")
        self.obs_feature = tf.reshape(self.model(image), (self.feature_dim, ))
        return self.obs_feature

    def _get_history_vector(self):
        self.history_vector = tf.reshape(self.history, (self.history_len * len(self.actions),))
        return self.history_vector

    def _positive_actions_idx(self):
        """Indexes of all positive actions"""
        positive_actions = []
        for i in range(len(self.actions)):
            new_bbox, _ = self._step(self.actions[i], self.obs_bbox)
            if self._reward(self.obs_bbox, self.actions[i], new_bbox) > 0:
                positive_actions.append(i)
        return positive_actions


    def get_random_expert_action(self):
        """Returns random positive-reward action, else random action if none are postive"""
        positive_actions_idx = self._positive_actions_idx()
        if positive_actions_idx:
            action_idx = np.random.choice(positive_actions_idx)
        else:
            action_idx = np.random.randint(len(self.actions))
        return self.actions[action_idx]

    def _step(self, action, bbox):
        bbox = bbox.copy()

        a_w = bbox[2] * self.transformation_factor
        a_h = bbox[3] * self.transformation_factor

        if a_w < 1:
            a_w = 1
        if a_h < 1:
            a_h = 1
        
        done = tf.zeros((1,))
        old_bbox = bbox.copy()

        if action[0]:
            bbox[0] = bbox[0] + a_w
        elif action[1]:
            bbox[0] = bbox[0] - a_w
        elif action[2]:
            bbox[1] = bbox[1] - a_h
        elif action[3]:
            bbox[1] = bbox[1] + a_h
        elif action[4]:
            bbox[0] = bbox[0] - a_w
            bbox[1] = bbox[1] - a_h
            bbox[2] = bbox[2] + 2 * a_w
            bbox[3] = bbox[3] + 2 * a_h
        elif action[5]:
            bbox[0] = bbox[0] + a_w
            bbox[1] = bbox[1] + a_h
            bbox[2] = bbox[2] - 2 * a_w
            bbox[3] = bbox[3] - 2 * a_h
        elif action[6]:
            bbox[1] = bbox[1] + a_h
            bbox[3] = bbox[3] - 2 * a_h
        elif action[7]:
            bbox[0] = bbox[0] + a_w
            bbox[2] = bbox[2] - 2 * a_w
        elif action[8]:
            done = tf.ones((1,))
        
        bbox = [int(i) for i in np.rint(bbox)]
            
        # Ensure obs_bbox is within bounds
        if bbox[0] < 0:
            bbox[0] = 0
        if bbox[1] < 0:
            bbox[1] = 0
        if bbox[2] < 1:
            bbox[2] = 1
        if bbox[3] < 1:
            bbox[3] = 1
        if bbox[2] > self.max_w:
            bbox[2] = self.max_w
        if bbox[3] > self.max_h:
            bbox[3] = self.max_h
        if bbox[0] + bbox[2] > self.max_w:
            bbox[0] = self.max_w - bbox[2]
        if bbox[1] + bbox[3] > self.max_h:
            bbox[1] = self.max_h - bbox[3]

        return bbox, done