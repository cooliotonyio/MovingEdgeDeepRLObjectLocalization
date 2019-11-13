import numpy as np
###################################################
# Everything below is stolen from cs285 hw 
###################################################

def lrelu(x, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    # initialize env for the beginning of a new rollout
    ob = env.reset()

    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    while True:
        # use the most recent ob to decide what to do
        obs.append(ob)
        ac = policy.get_action(ob)
        ac = ac[0]
        acs.append(ac)

        # take that action and record results
        ob, rew, done, _ = env.step(ac)

        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(rew)

        # End the rollout if the rollout ended 
        # Note that the rollout can end due to done, or due to max_path_length
        rollout_done = done or steps >= max_path_length
        terminals.append(rollout_done)
        
        if rollout_done: 
            break

    return Path(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render = render, render_mode = render_mode)
        timesteps_this_batch += get_pathlength(path)
        paths.append(path)
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    paths = []
    for i in range(ntraj):
        paths.append(sample_trajectory(env, policy, max_path_length, render = render, render_mode = render_mode))
    return paths

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

def get_pathlength(path):
    return len(path["reward"])