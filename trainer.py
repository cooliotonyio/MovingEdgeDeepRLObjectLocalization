class RL_Trainer(object):

    def __init__(self, params = None):
        # TODO
        self.agent = None

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

    def collect_training_trajectories(self, itr):
        # collect data to be used for training

    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        for train_step in range(self.params['num_agent_train_steps_per_iter']):

            # TODO: sample some data from the data buffer
            # TODO: use the sampled data for training
            loss = self.agent.train(None)
        
        return loss