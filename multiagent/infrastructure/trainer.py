import time

import numpy as np
import tensorflow as tf

# Pseudocode
'''
init replay memory of agents
init q-networks with rando weights
for episode in 1,m:
    initialize initial state
    for t in 1,T:
        select a_t via epsilon-greedy
        get tranisition (s,a,r,s)
        store transition in replay buffers
        sample minibatch of transitions
        gradient step
        copy params to virtual agents
        for j!=i:
            do something idk
        
'''

class RL_Trainer(object):

    def __init__(self, params = None):
        #############
        ## INIT
        #############
        self.agent = None
        self.params = params

        seed = self.params["seed"]
        tf.random.set_seed(seed)
        np.random.seed(seed)

        #############
        ## ENVIRONMENT
        #############
        self.env = self.params["env"]
        ob_dim = 50 #TODO
        ac_dim = self.env.get_ac_dim()
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim


        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])



    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """
        self.start_time = time.time()
        self.total_envsteps = 0

        #TODO: implement algorithm

        for iter in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Run agent
            self.agent.step_env()
            self.total_envsteps += 1

            # Train agent (using sampled data from replay buffer)
            losses = self.train_agent()

            # log/save
            if self.params['scalar_log_freq'] == -1:
                self.logmetrics = False
            elif itr % self.params['scalar_log_freq'] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            if self.logvideo or self.logmetrics:
                # perform logging
                print('\nBeginning logging procedure...')
                if isinstance(self.agent, DQNAgent):
                    self.perform_dqn_logging()
                else:
                    self.perform_logging(itr, paths, eval_policy, train_video_paths, all_losses)


                # save policy
                if self.params['save_params']:
                    print('\nSaving agent\'s actor...')
                    if 'actor' in dir(self.agent):
                        self.agent.actor.save(self.params['logdir'] + '/policy_itr_'+str(itr))
                    if 'critic' in dir(self.agent):
                        self.agent.critic.save(self.params['logdir'] + '/critic_itr_'+str(itr))

    ####################################
    ####################################

    def collect_training_trajectories(self, itr):
        # collect data to be used for training
        print("\nCollecting data to be used for training...")
        paths, envsteps_this_batch = sample_trajectories(self.env, collect_policy, batch_size, self.params['ep_len'])

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            train_video_paths = sample_n_trajectories(self.env, collect_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        pass