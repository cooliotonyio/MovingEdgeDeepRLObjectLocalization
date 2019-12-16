import numpy as np
import tensorflow as tf
import time

from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input
from multiagent.util.bbox import get_iou, draw_bbox, draw_cross, center_distance
from copy import deepcopy


class RotationEnum:
    START = -1
    TL = 0
    TR = 1
    BL = 2
    BR = 3

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


        self.target_bboxs = []
        self.orig_target_bboxs = None
        self.found_bboxs = []
        self.image = None
        self.orig_image = None

        self.target_bbox_ind_to_pop = None

        self.next_reset_location = RotationEnum.START

    def get_reward(self, old_bbox, action, new_bbox):
        """Returns reward, {-1,+1} for non-trigger actions, {-trigger_reward, +trigger_reward} for trigger action"""
        return tf.reshape(tf.cast(self._reward(old_bbox, action, new_bbox), tf.float32), (1,))
        

    def _get_max_iou(self, bbox):
        max_bbox_index = None
        max_iou = -np.inf

        for ind, target_bbox in enumerate(self.target_bboxs):
            iou = get_iou(bbox, target_bbox)
            if iou > max_iou:
                max_bbox_index = ind
                max_iou = iou
        
        return max_iou, max_bbox_index

    def _reward(self, old_bbox, action, new_bbox):
        """Reward function"""
        # non-trigger action reward
        if action[8] == 0:
            new_max_iou, _ = self._get_max_iou(new_bbox)
            old_max_iou, _ = self._get_max_iou(old_bbox)
            if new_max_iou > old_max_iou:
                return 1
            return -1
        
        # trigger action reward
        new_iou, max_bbox_index = self._get_max_iou(new_bbox)
        if new_iou > self.trigger_threshold:
            ret = self.trigger_reward
        else:
            ret = -self.trigger_reward

        self.target_bbox_ind_to_pop = max_bbox_index
        return ret
    
    #TODO: Limit environment to 40 steps, then reset to next location, with a maximum of 200 steps. 
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
        self.obs_bbox = self._transform(action, self.obs_bbox)

        self.history = tf.concat([
            self.history[1:], 
            tf.reshape(tf.cast(action, tf.float32), (1, len(self.actions)))], 
            axis=0)

        self._get_obs_feature()
        self._get_history_vector()

        obs = self.get_env_state()
        rew = self.get_reward(old_bbox, action, self.obs_bbox)

        done = tf.zeros((1,))
        
        if action[8] == 1:
            if self.orig_target_bboxs is not None: #In training mode
                if not self.target_bbox_ind_to_pop: #Draw over target bbox, not found bbox
                    _, max_bbox_index = self._get_max_iou(self.obs_bbox)
                    self.target_bbox_ind_to_pop = max_bbox_index

                bbox_to_draw = self.target_bboxs.pop(self.target_bbox_ind_to_pop) #TODO: Find faster way to store and remove bboxs?
                self._draw_cross_on_env(bbox_to_draw)

                self.target_bbox_ind_to_pop = None
                if len(self.target_bboxs) == 0:
                    done = tf.ones((1,))
            else: #Draw over found bbox in test mode
                self._draw_cross_on_env(self.obs_bbox)
            self.found_bboxs.append(self.obs_bbox)
            self.reset()
        
        return obs, rew, done

    def training_reset(self):
        self.target_bboxs = [deepcopy(bbox) for bbox in self.orig_target_bboxs]
        self.image = tf.identity(self.orig_image)
        self.found_bboxs = []
        self.next_reset_location = RotationEnum.START
        self.reset() #use regular reset afterwards

    #TODO: Allow for multiple target_bboxs's to be used, instead of only one.
    def reset(self, target_bboxs = None, image = None):
        """
        
        Resets the env. Resets bbox to entire image.
        
        If not in training mode:

            Reset resizes obs_bbox to 75% of the original size (not full image)
            placed in one of the following locations, in order: TL, TR, BL, BR

        """
        #TODO: If target_bboxs is None:
        if target_bboxs is not None:
            if type(target_bboxs) == list:
                self.target_bboxs = target_bboxs
            else:
                self.target_bboxs = [target_bboxs]
            self.orig_target_bboxs = [deepcopy(bbox) for bbox in self.target_bboxs]

        if image is not None:
            self.image = image
            self.orig_image = tf.identity(image)
            self.found_bboxs = []
            
        self.history = tf.zeros([self.history_len, len(self.actions)], dtype=tf.dtypes.float32)
        _, self.max_h, self.max_w, _ = self.image.shape


        if self.next_reset_location == RotationEnum.START:
            self.obs_bbox = [0, 0, self.max_w, self.max_h] 
        else:
            new_maxh = self.max_h * 0.866
            new_maxw = self.max_w * 0.866
            if self.next_reset_location == RotationEnum.TL:
                self.obs_bbox = [0, 0, new_maxw, new_maxh]
            elif self.next_reset_location == RotationEnum.TR:
                self.obs_bbox = [self.max_w - new_maxw, 0, new_maxw, new_maxh]
            elif self.next_reset_location == RotationEnum.BL:
                self.obs_bbox = [0, self.max_h - new_maxh, new_maxw, new_maxh]
            elif self.next_reset_location == RotationEnum.BR:
                self.obs_bbox = [self.max_w - new_maxw, self.max_h - new_maxh, new_maxw, new_maxh]
        
        self.obs_bbox = self._sanitize_bbox(self.obs_bbox)
        self.next_reset_location = (self.next_reset_location + 1) % 4

        self.obs_feature = self._get_obs_feature()
        self.history_vector = self._get_history_vector()

    def get_env_state(self):
        """The extracted feature vector of the bbox concatenated with history vector of actions"""
        return tf.concat((self.obs_feature, self.history_vector), axis=0)
    
    def show(self):
        """Draws both target bbox and current bbox in the image"""
        image = Image.fromarray(self.image.numpy()[0])
        # Draw current bbox in red
        if self.found_bboxs:
            for bbox in self.found_bboxs:
                image = draw_bbox(image, bbox, fill="yellow")
        image = draw_bbox(image, self.obs_bbox, fill="red")
        # Draw target bbox in green
        if self.orig_target_bboxs:
            for bbox in self.orig_target_bboxs:
                image = draw_bbox(image, bbox, fill="green")
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
            new_bbox = self._transform(self.actions[i], self.obs_bbox)
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

    def _draw_cross_on_env(self, bbox):
        pil_image = Image.fromarray(self.image.numpy()[0])
        pil_image_w_cross = draw_cross(pil_image, bbox)
        self.image = tf.expand_dims(np.array(pil_image_w_cross), 0)

    def _sanitize_bbox(self, bbox):
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

        return bbox

    def _transform(self, action, bbox):
        """Returns a copy of bbox after transformation of action. Note: this function is non-mutating"""
        bbox = bbox.copy()

        a_w = bbox[2] * self.transformation_factor
        a_h = bbox[3] * self.transformation_factor

        if a_w < 1:
            a_w = 1
        if a_h < 1:
            a_h = 1

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

        bbox = self._sanitize_bbox(bbox)
        return bbox

class TimedObjectLocalizationEnv(ObjectLocalizationEnv):
    """
    Test results:
    {   
        '_reward': 0.5307285785675049,
        '_transform': 33.052706241607666,
        '_get_obs_feature': 523.5038330554962,
        '_get_history_vector': 0.22332167625427246
    }

    """
    def __init__(self, *args, **kwargs):
        super(TimedObjectLocalizationEnv, self).__init__(*args, **kwargs)
        self.time = {
            "_reward": 0,
            "_transform": 0,
            "_get_obs_feature": 0,
            "_get_history_vector": 0,
        }

    def _transform(self, *args, **kwargs):
        start_time = time.time()
        return_val = super(TimedObjectLocalizationEnv, self)._transform(*args, **kwargs)
        self.time["_transform"] += time.time() - start_time
        return return_val

    def _reward(self, *args, **kwargs):
        start_time = time.time()
        return_val = super(TimedObjectLocalizationEnv, self)._reward(*args, **kwargs)
        self.time["_reward"] += time.time() - start_time
        return return_val

    def _get_obs_feature(self, *args, **kwargs):
        start_time = time.time()
        return_val = super(TimedObjectLocalizationEnv, self)._get_obs_feature(*args, **kwargs)
        self.time["_get_obs_feature"] += time.time() - start_time
        return return_val

    def _get_history_vector(self, *args, **kwargs):
        start_time = time.time()
        return_val = super(TimedObjectLocalizationEnv, self)._get_history_vector(*args, **kwargs)
        self.time["_get_history_vector"] += time.time() - start_time
        return return_val

class ObjectLocalizationEnvBetterReward(ObjectLocalizationEnv):

    def _reward(self, old_bbox, action, new_bbox):
        """Reward Function"""
        # Non-trigger reward
        if action[8] == 0:
            new_max_iou, new_max_bbox_index = self._get_max_iou(new_bbox)
            old_max_iou, old_max_bbox_index = self._get_max_iou(old_bbox)
            if new_max_iou > old_max_iou:
                return 1
            elif (  new_max_iou == old_max_iou and 
                    new_max_bbox_index == old_max_bbox_index and
                    self._is_closer(old_bbox, new_bbox, self.target_bboxs[new_max_bbox_index])):
                return 1
            return -1
        
        # Trigger reward
        new_iou, max_bbox_index = self._get_max_iou(new_bbox)
        if new_iou > self.trigger_threshold:
            ret = self.trigger_reward
        else:
            ret = -self.trigger_reward

        self.target_bbox_ind_to_pop = max_bbox_index
        return ret

    def _is_closer(self, old_bbox, new_bbox, target_bbox):
        """Returns if new_bbox is closer to target_bbox than old_bbox)"""
        old_distance = center_distance(old_bbox, target_bbox)
        new_distance = center_distance(new_bbox, target_bbox)
        return new_distance < old_distance
