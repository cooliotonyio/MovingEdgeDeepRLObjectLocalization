from tensorflow.keras.applications.vgg16 import preprocess_input
from multiagent.util.bbox import get_iou

class ObjectLocalizationEnv():
    def __init__(
        self, 
        model, 
        model_input_dim, 
        transformation_factor = 0.2, 
        trigger_threshold = 0.6, 
        trigger_reward = 3,
        history_len = 10):

        self.model = model
        self.model_input_dim = model_input_dim
        self.actions = np.arange(9)
        self.transformation_factor = 0.2
        self.trigger_threshold = trigger_threshold
        self.trigger_reward = trigger_reward
        self.history_len = history_len
        
    def get_reward(self, old_bbox, action, new_bbox):
        """Returns reward, {-1,+1} for non-trigger actions, {-trigger_reward, +trigger_reward} for trigger action"""
        # non-trigger action reward
        if not action[8]:
            return int(get_iou(self.new_bbox, self.target_bbox) > get_iou(self.old_bbox,self.target_bbox))
        
        # trigger action reward
        assert old_bbox == new_bbox
        if get_iou > self.trigger_threshold:
            return self.trigger_reward
        return -self.trigger_reward
    
    def step(self, action):
        '''
        Actions: [right, left, up, down, bigger, smaller, fatter, taller, trigger]
        '''
        a_w = int(self.obs_bbox[2] * self.transformation_factor)
        a_h = int(self.obs_bbox[3] * self.transformation_factor)
        
        done = 0
        old_bbox = self.obs_bbox.copy()

        if action[0]:
            self.obs_bbox[0] = self.obs_bbox[0] + a_w
        elif action[1]:
            self.obs_bbox[0] = self.obs_bbox[0] - a_w
        elif action[2]:
            self.obs_bbox[1] = self.obs_bbox[1] + a_h
        elif action[3]:
            self.obs_bbox[1] = self.obs_bbox[1] - a_h
        elif action[4]:
            self.obs_bbox[0] = self.obs_bbox[0] - a_w
            self.obs_bbox[1] = self.obs_bbox[1] - a_h
            self.obs_bbox[2] = self.obs_bbox[2] + 2 * a_w
            self.obs_bbox[3] = self.obs_bbox[3] + 2 * a_h
        elif action[5]:
            self.obs_bbox[0] = self.obs_bbox[0] + a_w
            self.obs_bbox[1] = self.obs_bbox[1] + a_h
            self.obs_bbox[2] = self.obs_bbox[2] - 2 * a_w
            self.obs_bbox[3] = self.obs_bbox[3] - 2 * a_h
        elif action[6]:
            self.obs_bbox[1] = self.obs_bbox[1] + a_h
            self.obs_bbox[3] = self.obs_bbox[3] - 2 * a_h
        elif action[7]:
            self.obs_bbox[0] = self.obs_bbox[0] + a_w
            self.obs_bbox[2] = self.obs_bbox[2] - 2 * a_w
        elif action[8]:
            done = 1
            
        # Ensure obs_bbox is within bounds
        if self.obs_bbox[0] < 0:
            self.obs_bbox[0] = 0
        if self.obs_bbox[1] < 0:
            self.obs_bbox[1] = 0
        if self.obs_bbox[0] + self.obs_bbox[2] > self.max_w:
            self.obs_bbox[2] = self.max_w - self.obs_bbox[0]
        if self.obs_bbox[1] + self.obs_bbox[3] > self.max_h:
            self.obs_bbox[3] = self.max_h - self.obs_bbox[1]
            
        self.history = tf.concat(self.history[1:], self.action)
        self._get_obs_feature()

        obs = self.get_env_state()
        rew = self.get_reward(old_bbox, action, self.obs_bbox)
        
        return obs, rew, done
        
    def reset(self, target_bbox, image):
        self.target_bbox = target_bbox
        self.history = np.zeros([self.history_len, len(actions)])
        self.image = image
        _, self.max_h, self.max_w, _ = self.image.shape
        self.obs_bbox = [0, 0, self.max_w, self.max_h] 
        self.obs_feature = self._get_obs_feature()
        
    def get_env_state(self):
        return tf.concat([self.obs_feature, np.flatten(self.history)], axis = 1)
    
    def show(self):
        image = Image.fromarray(self.image.numpy()[0])
        # Draw current bbox in red
        image = draw_bbox(image, self.obs_bbox, fill="red")
        # Draw target bbox in green
        image = draw_bbox(image, self.target_bbox, fill="green")
        return image
    
    def get_ac_dim(self):
        return len(self.actions)
    
    def _get_obs_feature(self):
        image = tf.image.crop_to_bounding_box(self.image, self.obs_bbox[1], self.obs_bbox[0], self.obs_bbox[3], self.obs_bbox[2])
        image = preprocess_input(tf.image.resize(image, self.model_input_dim), mode="tf")
        self.obs_feature = self.model.predict(image)
        return self.obs_feature

    def get_random_expert_action():
        """Returns random positive-reward action"""
        
        #TODO: make this return random POSITIVE action instead of just a random action
        a = np.random.randint(9)
        action = np.zeroes(len(self.actions))
        np.put(action,a,1)
        return action