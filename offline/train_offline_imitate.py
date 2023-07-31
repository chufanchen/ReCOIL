import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax.config import config
from typing import Tuple
import datetime
import gym
import numpy as np
import tqdm
import time
import absl
import sys
from absl import app, flags
from ml_collections import config_flags
from tensorboardX import SummaryWriter
from dataclasses import dataclass

import wrappers
from dataset_utils import D4RLDataset,D4RLMixedDataset, split_into_trajectories
from evaluation import evaluate, evaluate_ant
from learner_imitate import Learner
import warnings
from logging_utils.logx import EpochLogger
import envs
os.environ['WANDB_DISABLED'] = 'true' 


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
    elif "antmaze" in env_name:
        expert_env = gym.make(f"antmaze-umaze-v2")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=700, env_name = env_name)
    elif "Grid" in env_name:
        expert_env = gym.make(f"GoalGrid-v0")
        expert_env = wrappers.EpisodeMonitor(expert_env)
        expert_env = wrappers.SinglePrecision(expert_env)
        expert_dataset = D4RLDataset(expert_env, transitions=5, env_name = env_name)
    

    offline_min=None
    offline_max=None
    if 'walker2d' in env_name:
        if FLAGS.expert_trajectories==200:
            expert_trajs = 100
        else:
            expert_trajs = FLAGS.expert_trajectories
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=expert_trajs,env_name=env_name)
    elif 'Grid' in env_name:
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=1,env_name=env_name)
    elif 'antmaze' in env_name:
        if FLAGS.expert_trajectories==200:
            expert_trajs = 3000
        else:
            expert_trajs = FLAGS.expert_trajectories
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=expert_trajs,env_name="antmaze")
        offline_min = dataset.observations[:,:2].min(axis=0)
        offline_max = dataset.observations[:,:2].max(axis=0)
        expert_dataset.observations[:,:2] = (expert_dataset.observations[:,:2]-dataset.observations[:,:2].min(axis=0))/(dataset.observations[:,:2].max(axis=0)-dataset.observations[:,:2].min(axis=0))
        dataset.observations[:,:2] = (dataset.observations[:,:2]-dataset.observations[:,:2].min(axis=0))/(dataset.observations[:,:2].max(axis=0)-dataset.observations[:,:2].min(axis=0))
        
    elif 'kitchen' not in env_name:
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=FLAGS.expert_trajectories,env_name=env_name)
    else:
        
        dataset = D4RLMixedDataset(env, expert_env, expert_trajectories=1,env_name=env_name) #D4RLDataset(env)
    print("Expert dataset size: {} Offline dataset size: {}".format(expert_dataset.observations.shape[0],dataset.observations.shape[0]))

    return env, dataset, expert_dataset, offline_min, offline_max


def main(_):

    ts_str = datetime.datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(FLAGS.save_dir, ts_str)
    exp_id = f"results/offline_imitation2/{FLAGS.env_name}/" + FLAGS.exp_name
    log_folder = exp_id + '/'+FLAGS.exp_name+'_s'+str(FLAGS.seed) 
    logger_kwargs={'output_dir':log_folder, 'exp_name':FLAGS.exp_name}
    e_logger = EpochLogger(**logger_kwargs)

    hparam_str_dict = dict(seed=FLAGS.seed, env=FLAGS.env_name)
    hparam_str = ','.join([
        '%s=%s' % (k, str(hparam_str_dict[k]))
        for k in sorted(hparam_str_dict.keys())
    ])

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

    sys.exit(0)
    os._exit(0)
    raise SystemExit


if __name__ == '__main__':
    app.run(main)
