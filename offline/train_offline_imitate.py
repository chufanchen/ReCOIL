import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from typing import Tuple
import datetime
import gym
import numpy as np
import time
import pickle
import collections
from absl import app, flags
from ml_collections import config_flags
from dataclasses import dataclass
import wrappers
from dataset_utils import D4RLDataset, D4RLMixedDataset, split_into_trajectories
from evaluation import evaluate, evaluate_ant
from learner_imitate import Learner, compute_rewards
from logging_utils.logx import EpochLogger
from common import Batch, MixedBatch


FLAGS = flags.FLAGS

flags.DEFINE_string('env_name', 'halfcheetah-expert-v2', 'Environment name.')
flags.DEFINE_string('save_dir', './tmp/', 'Tensorboard logging dir.')
flags.DEFINE_string('exp_name', 'dump', 'Epoch logging dir.')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('expert_trajectories', 200, 'Number of expert trajectories')
flags.DEFINE_integer('eval_episodes', 10,
                     'Number of episodes used for evaluation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 5000, 'Eval interval.')
flags.DEFINE_integer('batch_size', 1024, 'Mini batch size.')
flags.DEFINE_float('temp', 1.0, 'Loss temperature')
flags.DEFINE_boolean('double', True, 'Use double q-learning')
flags.DEFINE_integer('max_steps', int(1e6), 'Number of training steps.')
flags.DEFINE_boolean('tqdm', True, 'Use tqdm progress bar.')
flags.DEFINE_integer('sample_random_times', 0, 'Number of random actions to add to smooth dataset')
flags.DEFINE_boolean('grad_pen', False, 'Add a gradient penalty to critic network')
flags.DEFINE_float('lambda_gp', 1, 'Gradient penalty coefficient')
flags.DEFINE_float('max_clip', 7., 'Loss clip value')
flags.DEFINE_integer('num_v_updates', 1, 'Number of value updates per iter')
flags.DEFINE_boolean('log_loss', False, 'Use log gumbel loss')
flags.DEFINE_float('alpha', 0.8, 'f-maximization strength')
flags.DEFINE_float('beta', 0.1, 'imitation strength vs bellman strength')

flags.DEFINE_boolean('noise', False, 'Add noise to actions')
flags.DEFINE_float('noise_std', 0.1, 'Noise std for actions')

config_flags.DEFINE_config_file(
    'config',
    'default.py',
    'File path to the training hyperparameter configuration.',
    lock_config=False)



@dataclass(frozen=True)
class ConfigArgs:
    sample_random_times: int
    grad_pen: bool
    noise: bool
    noise_std: float
    lambda_gp: int
    max_clip: float
    num_v_updates: int
    log_loss: bool


def normalize(dataset):

    trajs = split_into_trajectories(dataset.observations, dataset.actions,
                                    dataset.rewards, dataset.masks,
                                    dataset.dones_float,
                                    dataset.next_observations)

    def compute_returns(traj):
        episode_return = 0
        for _, _, rew, _, _, _ in traj:
            episode_return += rew

        return episode_return

    trajs.sort(key=compute_returns)

    dataset.rewards /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
    dataset.rewards *= 1000.0


def make_env_and_dataset(env_name: str,
                         seed: int) -> Tuple[gym.Env, D4RLDataset]:
    env = gym.make(env_name)
    env = wrappers.EpisodeMonitor(env)
    env = wrappers.SinglePrecision(env)

    env.seed(seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    expert_dataset = None
    
    if 'kitchen' in env_name:
        expert_env = gym.make(f"kitchen-complete-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=183)
    elif 'halfcheetah-random' in env_name:
        expert_env = gym.make(f"halfcheetah-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif 'halfcheetah-medium' in env_name:
        expert_env = gym.make(f"halfcheetah-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif 'hopper-random' in env_name:
        expert_env = gym.make(f"hopper-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif 'hopper-medium' in env_name:
        expert_env = gym.make(f"hopper-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif "walker2d-random" in env_name:
        expert_env = gym.make(f"walker2d-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif "walker2d-medium" in env_name:
        expert_env = gym.make(f"walker2d-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif "ant-random" in env_name:
        expert_env = gym.make(f"ant-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif "ant-medium" in env_name:
        expert_env = gym.make(f"ant-expert-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=1000)
    elif "pen-human" in env_name:
        expert_env = gym.make(f"pen-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=100)
    elif "pen-cloned" in env_name:
        expert_env = gym.make(f"pen-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=100)
    elif "door-human" in env_name:
        expert_env = gym.make(f"door-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)
    elif "door-cloned" in env_name:
        expert_env = gym.make(f"door-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)
    elif "hammer-human" in env_name:
        expert_env = gym.make(f"hammer-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)
    elif "hammer-cloned" in env_name:
        expert_env = gym.make(f"hammer-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)
    elif "relocate-human" in env_name:
        expert_env = gym.make(f"relocate-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)
    elif "relocate-cloned" in env_name:
        expert_env = gym.make(f"relocate-expert-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=200)


    offline_min=None
    offline_max=None
    if 'walker2d' in env_name:
        if FLAGS.expert_trajectories==200:
            expert_trajs = 100 # Following protocol from the SMODICE paper
        else:
            expert_trajs = FLAGS.expert_trajectories
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=expert_trajs,env_name=env_name) 
    elif 'kitchen' not in env_name:
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=FLAGS.expert_trajectories,env_name=env_name)
    else:
        
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=1,env_name=env_name) #D4RLDataset(env)
    print("Expert dataset size: {} Offline dataset size: {}".format(expert_dataset.observations.shape[0],dataset.observations.shape[0]))

    return env, dataset, expert_dataset, offline_min, offline_max


def process_dataset_with_learned_rewards(dataset, agent, batch_size=1024):
    """Process the entire dataset with learned rewards."""
    # Create a copy of the dataset dictionary to modify
    dataset_dict = {
        'observations': dataset.observations.copy(),
        'actions': dataset.actions.copy(),
        'rewards': np.zeros_like(dataset.rewards),  # Will replace with learned rewards
        'masks': dataset.masks.copy(),
        'dones_float': dataset.dones_float.copy(),
        'next_observations': dataset.next_observations.copy()
    }
    
    total_entries = len(dataset.observations)
    num_batches = (total_entries + batch_size - 1) // batch_size  # Ceiling division
    
    print(f"Processing dataset with {total_entries} entries in {num_batches} batches...")
    
    # Process the dataset sequentially in batches
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, total_entries)
        
        # Create batch using Batch namedtuple
        if hasattr(dataset, 'is_expert'):  # Check if it's a MixedDataset
            batch = MixedBatch(
                observations=dataset.observations[start_idx:end_idx],
                actions=dataset.actions[start_idx:end_idx],
                rewards=dataset.rewards[start_idx:end_idx],
                masks=dataset.masks[start_idx:end_idx],
                next_observations=dataset.next_observations[start_idx:end_idx],
                is_expert=dataset.is_expert[start_idx:end_idx] if hasattr(dataset, 'is_expert') else None
            )
        else:
            batch = Batch(
                observations=dataset.observations[start_idx:end_idx],
                actions=dataset.actions[start_idx:end_idx],
                rewards=dataset.rewards[start_idx:end_idx],
                masks=dataset.masks[start_idx:end_idx],
                next_observations=dataset.next_observations[start_idx:end_idx]
            )
        
        # Compute learned rewards for this batch
        learned_batch_rewards = compute_rewards(agent, batch)
        
        # Store the learned rewards
        dataset_dict['rewards'][start_idx:end_idx] = learned_batch_rewards
        
        if (i + 1) % 10 == 0 or i == num_batches - 1:
            print(f"Processed batch {i+1}/{num_batches}")
    
    return dataset_dict


def format_dataset_as_trajectories(dataset_dict):
    """Format the dataset into a list of trajectories (episodes)."""
    N = dataset_dict['rewards'].shape[0]
    data_ = collections.defaultdict(list)
    
    episode_step = 0
    paths = []
    
    use_timeouts = False
    terminals = 1.0 - dataset_dict['masks']  # masks = 1.0 - terminals in our dataset
    
    print(f"Formatting dataset with {N} transitions into trajectories...")
    
    for i in range(N):
        done_bool = bool(terminals[i])
        final_timestep = False  # We don't have timeout info, using only terminal flags
        
        for k in ['observations', 'actions', 'rewards']:
            data_[k].append(dataset_dict[k][i])
        
        # Add terminal flag
        data_['terminals'].append(terminals[i])
        
        if done_bool or final_timestep:
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        else:
            episode_step += 1
            
    # Handle any remaining data (if the last trajectory was incomplete)
    if len(data_['rewards']) > 0:
        episode_data = {}
        for k in data_:
            episode_data[k] = np.array(data_[k])
        paths.append(episode_data)
    
    returns = np.array([np.sum(p['rewards']) for p in paths])
    num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    
    print(f'Number of samples collected: {num_samples}')
    print(f'Number of trajectories: {len(paths)}')
    print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    return paths


def main(_):

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)
    exp_id = f"results/offline_imitation/{FLAGS.env_name}/" + FLAGS.exp_name
    log_folder = exp_id + '/'+FLAGS.exp_name+'_s'+str(FLAGS.seed) 
    logger_kwargs={'output_dir':log_folder, 'exp_name':FLAGS.exp_name}
    e_logger = EpochLogger(**logger_kwargs)


    os.makedirs(save_dir, exist_ok=True)

    env, dataset, expert_dataset, offline_min, offline_max = make_env_and_dataset(FLAGS.env_name, FLAGS.seed)

    kwargs = dict(FLAGS.config)

    args = ConfigArgs(sample_random_times=FLAGS.sample_random_times,
                      grad_pen=FLAGS.grad_pen,
                      lambda_gp=FLAGS.lambda_gp,
                      noise=FLAGS.noise,
                      max_clip=FLAGS.max_clip,
                      num_v_updates=FLAGS.num_v_updates,
                      log_loss=FLAGS.log_loss,
                      noise_std=FLAGS.noise_std)
    agent = Learner(FLAGS.seed,
                    env.observation_space.sample()[np.newaxis],
                    env.action_space.sample()[np.newaxis],
                    max_steps=FLAGS.max_steps,
                    loss_temp=FLAGS.temp,
                    double_q=FLAGS.double,
                    vanilla=False,
                    alpha = FLAGS.alpha,
                     beta = FLAGS.beta,
                    args=args,
                    **kwargs)

    best_eval_returns = -np.inf
    eval_returns = []
    for i in range(1, FLAGS.max_steps + 1): # Remove TQDM
        batch = dataset.sample(FLAGS.batch_size)
        expert_batch = expert_dataset.sample(FLAGS.batch_size)
        update_info = agent.update(batch, expert_batch)

        if i % FLAGS.eval_interval == 0:
            if 'antmaze' in FLAGS.env_name:
                eval_stats = evaluate_ant(agent, env, FLAGS.eval_episodes, offline_min, offline_max)
            else:    
                eval_stats = evaluate(agent, env, FLAGS.eval_episodes)

            if eval_stats['return'] >= best_eval_returns:
                # Store best eval returns
                best_eval_returns = eval_stats['return']
                agent.save(os.path.join(save_dir, f'{FLAGS.seed}', f'best'))
            e_logger.log_tabular('Iterations', i)
            e_logger.log_tabular('AverageNormalizedReturn', eval_stats['return'])
            e_logger.log_tabular('UnseenExpertV', update_info['unseen_v_expert'].item())
            e_logger.log_tabular('UnseenRandomV', update_info['unseen_v_suboptimal'].item())
            e_logger.log_tabular('UnseenExpertQ', update_info['unseen_q_expert'].item())
            e_logger.log_tabular('UnseenRandomQ', update_info['unseen_q_suboptimal'].item())
            e_logger.log_tabular('ClippedAdv', update_info['clipped_adv'].mean().item())
            e_logger.dump_tabular()
            eval_returns.append((i, eval_stats['return']))
            print("Iterations: {} Average Return: {}".format(i,eval_stats['return']))

    print(f"Training finished. Best eval return: {best_eval_returns}")
    print("Computing learned rewards and replacing the ground truth rewards in the D4RL dataset...")
    
    # Process the entire dataset with learned rewards
    modified_dataset = process_dataset_with_learned_rewards(dataset, agent, FLAGS.batch_size)
    
    # Create a meaningful filename based on environment name
    env_name_parts = FLAGS.env_name.split('-')
    if len(env_name_parts) >= 2:
        output_filename = f"{env_name_parts[0]}_{env_name_parts[1]}_proxy_reward.pkl"
    else:
        output_filename = f"{FLAGS.env_name}_proxy_reward.pkl"
    
    # Save the dataset with learned rewards
    with open(output_filename, 'wb') as f:
        pickle.dump(modified_dataset, f)
    
    print(f"Dataset with learned rewards saved to {output_filename}")
    
    # Sample statistics about the learned rewards
    print(f"Learned rewards statistics:")
    print(f"  Min: {modified_dataset['rewards'].min()}")
    print(f"  Max: {modified_dataset['rewards'].max()}")
    print(f"  Mean: {modified_dataset['rewards'].mean()}")
    print(f"  Std: {modified_dataset['rewards'].std()}")

    # Format the dataset as trajectories
    formatted_dataset = format_dataset_as_trajectories(modified_dataset)
    
    # Save the formatted dataset
    formatted_filename = f"{FLAGS.env_name}.pkl"
    with open(formatted_filename, 'wb') as f:
        pickle.dump(formatted_dataset, f)
    
    print(f"Formatted dataset saved to {formatted_filename}")

if __name__ == '__main__':
    app.run(main)
