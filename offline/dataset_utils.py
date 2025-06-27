import collections
from typing import Optional
# import ipdb;ipdb.set_trace()
import d4rl
import gym
import numpy as np
from tqdm import tqdm
import copy
import h5py
import matplotlib.pyplot as plt
Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])

MixedBatch = collections.namedtuple(
    'MixedBatch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations','is_expert'])

def split_into_trajectories_with_rewards(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = []
    traj = []
    for i in tqdm(range(len(observations))):
        traj.append(rewards[i])
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append(traj)
            traj = []
    return trajs

def split_into_trajectories(observations, actions, rewards, masks, dones_float,
                            next_observations):
    trajs = [[]]

    for i in tqdm(range(len(observations))):
        trajs[-1].append((observations[i], actions[i], rewards[i], masks[i],
                          dones_float[i], next_observations[i]))
        if dones_float[i] == 1.0 and i + 1 < len(observations):
            trajs.append([])

    return trajs


def merge_trajectories(trajs):
    observations = []
    actions = []
    rewards = []
    masks = []
    dones_float = []
    next_observations = []

    for traj in trajs:
        for (obs, act, rew, mask, done, next_obs) in traj:
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            masks.append(mask)
            dones_float.append(done)
            next_observations.append(next_obs)

    return np.stack(observations), np.stack(actions), np.stack(
        rewards), np.stack(masks), np.stack(dones_float), np.stack(
            next_observations)


class Dataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return Batch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx])



class MixedDataset(object):
    def __init__(self, observations: np.ndarray, actions: np.ndarray,
                 rewards: np.ndarray, masks: np.ndarray,
                 dones_float: np.ndarray, next_observations: np.ndarray, is_expert: np.ndarray,
                 size: int):
        self.observations = observations
        self.actions = actions
        self.rewards = rewards
        self.masks = masks
        self.dones_float = dones_float
        self.next_observations = next_observations
        self.is_expert = is_expert
        self.size = size

    def sample(self, batch_size: int) -> Batch:
        indx = np.random.randint(self.size, size=batch_size)
        return MixedBatch(observations=self.observations[indx],
                     actions=self.actions[indx],
                     rewards=self.rewards[indx],
                     masks=self.masks[indx],
                     next_observations=self.next_observations[indx],
                     is_expert=self.is_expert[indx])


class D4RLDataset(Dataset):
    def __init__(self,
                 env: gym.Env,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):
        if transitions is not None and "antmaze" in env_name:
            import pickle
            with open('/data/harshit_sikchi/work/Recoil/ant_seen_expert.pkl', 'rb') as handle:
                expert_dict = pickle.load(handle)
            dataset = {}
            dataset['observations'] = expert_dict['observations'][:-1]
            dataset['actions'] = expert_dict['actions'][:-1]
            dataset['rewards'] = expert_dict['rewards'][:-1]
            dataset['terminals'] = expert_dict['terminals'][:-1]
            dataset['terminals'][-1]=True
            dataset['next_observations'] = expert_dict['observations'][1:]
            transitions = len(dataset['observations'])
            
        elif 'Grid' in  env_name:
            dataset = {'observations':[],'actions':[],'rewards':[],'terminals':[],'next_observations':[]}
            for traj_id in range(1):
                # start_state = np.random.uniform([env.range_x[0], env.range_y[0]], [env.range_x[1], env.range_y[1]])
                env.reset()
                goal_state = env.goal
                state = np.array([-1.2,-1.2])
                env.set_state(state)
                for i in range(5):
                    act = np.clip(goal_state-state,-0.25,0.25).reshape(-1)
                    # import ipdb;ipdb.set_trace()
                    next_state, rew, done, _ = env.step(act)
                    dataset['observations'].append(state.reshape(-1))
                    dataset['actions'].append(act.reshape(-1))
                    dataset['rewards'].append(rew)
                    dataset['terminals'].append(done)
                    dataset['next_observations'].append(next_state.reshape(-1))
                    state = next_state
            
            dataset['observations'] = np.array(dataset['observations'])
            dataset['actions'] = np.array(dataset['actions'])
            dataset['rewards'] = np.array(dataset['rewards'])
            dataset['terminals'] = np.array(dataset['terminals'])
            dataset['next_observations'] = np.array(dataset['next_observations'])

            # import ipdb;ipdb.set_trace()
        else:
            dataset = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset['actions'] = np.clip(dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(dataset['observations'][i + 1] -
                              dataset['next_observations'][i]
                              ) > 1e-6 or dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0
        # if transitions is not None and "antmaze" in env_name:
        dones_float[-1] = 1
        if transitions is not None:
            super().__init__(dataset['observations'][:transitions].astype(np.float32),
                         actions=dataset['actions'][:transitions].astype(np.float32),
                         rewards=dataset['rewards'][:transitions].astype(np.float32),
                         masks=1.0 - dataset['terminals'][:transitions].astype(np.float32),
                         dones_float=dones_float[:transitions].astype(np.float32),
                         next_observations=dataset['next_observations'][:transitions].astype(
                             np.float32),
                         size=transitions)
        else:    
            super().__init__(dataset['observations'].astype(np.float32),
                         actions=dataset['actions'].astype(np.float32),
                         rewards=dataset['rewards'].astype(np.float32),
                         masks=1.0 - dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=dataset['next_observations'].astype(
                             np.float32),
                         size=len(dataset['observations']))

def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys

def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    return data_dict

class D4RLMixedDataset(MixedDataset):
    def __init__(self,
                 env: gym.Env,
                 expert_env: gym.Env,
                 expert_trajectories: int,
                 clip_to_eps: bool = True,
                 eps: float = 1e-5,
                 transitions= None,
                 env_name=""):
        if 'antmaze' in env_name:
            dataset = get_dataset("/data/harshit_sikchi/work/Recoil/antmaze-umaze-v2-randomstart-noiserandomaction.hdf5")
            
            max_steps = 700
        elif 'Grid' in  env_name:
            dataset = {'observations':[],'actions':[],'rewards':[],'terminals':[],'next_observations':[]}
            for traj_id in range(100000):
                # start_state = np.random.uniform([env.range_x[0], env.range_y[0]], [env.range_x[1], env.range_y[1]])
                goal_state = env.goal
                state = env.reset()
                for i in range(5):
                    act = np.random.uniform(-0.25,0.25,2)
                    next_state, rew, done, _ = env.step(act)
                    dataset['observations'].append(state.reshape(-1))
                    dataset['actions'].append(act.reshape(-1))
                    dataset['rewards'].append(rew)
                    dataset['terminals'].append(done)
                    dataset['next_observations'].append(next_state.reshape(-1))
                    state = next_state
            dataset['observations'] = np.array(dataset['observations'])
            dataset['actions'] = np.array(dataset['actions'])
            dataset['rewards'] = np.array(dataset['rewards'])
            dataset['terminals'] = np.array(dataset['terminals'])
            dataset['next_observations'] = np.array(dataset['next_observations'])
            plt.clf()
            plt.scatter(dataset['observations'][:,0],dataset['observations'][:,1],s=0.1)
            plt.savefig('Offline_dataset.png')
        elif 'kitchen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 280
        elif 'pen' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 100
        elif 'door' in env_name or 'hammer' in env_name or 'relocate' in env_name:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 200    
        else:
            dataset = d4rl.qlearning_dataset(env)
            max_steps = 1000
        
        if 'Grid' in  env_name:
            expert_dataset = {'observations':[],'actions':[],'rewards':[],'terminals':[],'next_observations':[]}
            expert_transitions = 0
            for traj_id in range(1):
                # start_state = np.random.uniform([env.range_x[0], env.range_y[0]], [env.range_x[1], env.range_y[1]])
                env.reset()
                goal_state = env.goal
                state = np.array([-1.2,-1.2])
                env.set_state(state)
                for i in range(5):
                    act = np.clip(goal_state-state,-0.25,0.25)
                    next_state, rew, done, _ = env.step(act)
                    expert_dataset['observations'].append(state.reshape(-1))
                    expert_dataset['actions'].append(act.reshape(-1))
                    expert_dataset['rewards'].append(rew)
                    expert_dataset['terminals'].append(done)
                    expert_dataset['next_observations'].append(next_state.reshape(-1))
                    expert_transitions+=1
                    state = next_state
            expert_dataset['observations'] = np.array(expert_dataset['observations'])
            expert_dataset['actions'] = np.array(expert_dataset['actions'])
            expert_dataset['rewards'] = np.array(expert_dataset['rewards'])
            expert_dataset['terminals'] = np.array(expert_dataset['terminals'])
            expert_dataset['next_observations'] = np.array(expert_dataset['next_observations'])

        else:
            expert_dataset = d4rl.qlearning_dataset(expert_env)

            expert_transitions = 0
            traj_count= 0
            episode_step=0
            for i in range(expert_dataset['observations'].shape[0]):
                episode_step+=1
                
                if episode_step == max_steps or expert_dataset['terminals'][i]:
                    # if episode_step == max_steps:
                    traj_count+=1
                    episode_step = 0
                if traj_count == expert_trajectories:
                    expert_transitions = i
                    break
        
        # import ipdb;ipdb.set_trace()
            
        combined_dataset = copy.copy(dataset)
        combined_dataset['observations'] = np.concatenate((dataset['observations'], expert_dataset['observations'][:expert_transitions]),axis=0)
        combined_dataset['actions'] = np.concatenate((dataset['actions'], expert_dataset['actions'][:expert_transitions]),axis=0)
        combined_dataset['rewards'] = np.concatenate((dataset['rewards'], expert_dataset['rewards'][:expert_transitions]),axis=0)
        combined_dataset['terminals'] = np.concatenate((dataset['terminals'], expert_dataset['terminals'][:expert_transitions]),axis=0)
        combined_dataset['next_observations'] = np.concatenate((dataset['next_observations'], expert_dataset['next_observations'][:expert_transitions]),axis=0)
        combined_dataset['is_expert'] = np.concatenate((np.zeros(dataset['observations'].shape[0]), np.ones(expert_transitions)),axis=0)
        if clip_to_eps:
            lim = 1 - eps
            combined_dataset['actions'] = np.clip(combined_dataset['actions'], -lim, lim)

        dones_float = np.zeros_like(combined_dataset['rewards'])

        for i in range(len(dones_float) - 1):
            if np.linalg.norm(combined_dataset['observations'][i + 1] -
                              combined_dataset['next_observations'][i]
                              ) > 1e-6 or combined_dataset['terminals'][i] == 1.0:
                dones_float[i] = 1
            else:
                dones_float[i] = 0

        dones_float[-1] = 1
        if transitions is not None:
            super().__init__(combined_dataset['observations'][:transitions].astype(np.float32),
                         actions=combined_dataset['actions'][:transitions].astype(np.float32),
                         rewards=combined_dataset['rewards'][:transitions].astype(np.float32),
                         masks=1.0 - combined_dataset['terminals'][:transitions].astype(np.float32),
                         dones_float=dones_float[:transitions].astype(np.float32),
                         next_observations=combined_dataset['next_observations'][:transitions].astype(
                             np.float32),
                        is_expert = combined_dataset['is_expert'][:transitions].astype(np.float32),
                         size=transitions)
        else:    
            super().__init__(combined_dataset['observations'].astype(np.float32),
                         actions=combined_dataset['actions'].astype(np.float32),
                         rewards=combined_dataset['rewards'].astype(np.float32),
                         masks=1.0 - combined_dataset['terminals'].astype(np.float32),
                         dones_float=dones_float.astype(np.float32),
                         next_observations=combined_dataset['next_observations'].astype(
                             np.float32),
                        is_expert=combined_dataset['is_expert'].astype(np.float32),
                         size=len(combined_dataset['observations']))

class ReplayBuffer(Dataset):
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int,
                 capacity: int):

        observations = np.empty((capacity, *observation_space.shape),
                                dtype=observation_space.dtype)
        actions = np.empty((capacity, action_dim), dtype=np.float32)
        rewards = np.empty((capacity, ), dtype=np.float32)
        masks = np.empty((capacity, ), dtype=np.float32)
        dones_float = np.empty((capacity, ), dtype=np.float32)
        next_observations = np.empty((capacity, *observation_space.shape),
                                     dtype=observation_space.dtype)
        super().__init__(observations=observations,
                         actions=actions,
                         rewards=rewards,
                         masks=masks,
                         dones_float=dones_float,
                         next_observations=next_observations,
                         size=0)

        self.size = 0

        self.insert_index = 0
        self.capacity = capacity

    def initialize_with_dataset(self, dataset: Dataset,
                                num_samples: Optional[int]):
        assert self.insert_index == 0, 'Can insert a batch online in an empty replay buffer.'

        dataset_size = len(dataset.observations)

        if num_samples is None:
            num_samples = dataset_size
        else:
            num_samples = min(dataset_size, num_samples)
        assert self.capacity >= num_samples, 'Dataset cannot be larger than the replay buffer capacity.'

        if num_samples < dataset_size:
            perm = np.random.permutation(dataset_size)
            indices = perm[:num_samples]
        else:
            indices = np.arange(num_samples)

        self.observations[:num_samples] = dataset.observations[indices]
        self.actions[:num_samples] = dataset.actions[indices]
        self.rewards[:num_samples] = dataset.rewards[indices]
        self.masks[:num_samples] = dataset.masks[indices]
        self.dones_float[:num_samples] = dataset.dones_float[indices]
        self.next_observations[:num_samples] = dataset.next_observations[
            indices]

        self.insert_index = num_samples
        self.size = num_samples

    def insert(self, observation: np.ndarray, action: np.ndarray,
               reward: float, mask: float, done_float: float,
               next_observation: np.ndarray):
        self.observations[self.insert_index] = observation
        self.actions[self.insert_index] = action
        self.rewards[self.insert_index] = reward
        self.masks[self.insert_index] = mask
        self.dones_float[self.insert_index] = done_float
        self.next_observations[self.insert_index] = next_observation

        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
