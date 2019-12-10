import time

import numpy as np
import tensorflow as tf

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

        """
        self.start_time = time.time()
        self.total_envsteps = 0
        #TODO: implement algorithm

        for iter in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # Run agent
            self.agent.train_step()
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

    def train_agent():
