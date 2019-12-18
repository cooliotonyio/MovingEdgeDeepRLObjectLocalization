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
        self.max_path_length = self.params["max_path_length"]

        self.learning_freq = self.params['learning_freq']
        self.save_freq = self.params['save_freq']
        self.model_name = self.params['model_name']

        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.params, self.params['agent_params'])

    def run_training_loop(self, n_iter):
        """
        Returns (returns, losses) of training n_iter times

        :param n_iter:  number of iterations

        # Pseudocode
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
                

        """
        self.start_time = time.time()
        self.total_envsteps = 0
        
        returns = []
        losses = []

        for i in range(n_iter):
            print("********** Iteration %i ************"%i)

            self.env.training_reset()
            rollouts = []
            done = False

            for step in range(self.max_path_length):
                # Run agent
                obs, acs, rew, next_obs, done = self.agent.step(mode="train")
                rollouts.append({
                    "obs": obs,
                    "acs": acs,
                    "rew": rew,
                    "next_obs": next_obs,
                    "done": done
                })
                self.total_envsteps += 1
                if done:
                    break

            self.agent.add_to_replay_buffer(rollouts)
            
            rews = [r["rew"].numpy()[0] for r in rollouts]
            print("PATH:   \t",[np.argmax(r["acs"]) for r in rollouts])
            print("REWARDS:\t", rews)
            print("RETURN: \t", np.sum(rews))
            print("TIME:  \t", time.time() - self.start_time)
            print("STEPS: \t", self.total_envsteps)

            returns.append(np.sum(rews))
            # Train agent (using sampled data from replay buffer)
            if i % self.learning_freq == 0 and self.agent.can_sample_replay_buffer():
                loss = self.agent.train()
                losses.append(loss)
                print("LOSS:  \t", loss.numpy())
        
            if self.save_freq and i % self.save_freq == 0 and i:
                print("SAVING MODEL")
                self.agent.q_func.save("models/{}_iter{}".format(self.model_name, i))

        return returns, losses

