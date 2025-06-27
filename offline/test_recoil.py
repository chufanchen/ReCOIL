import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from absl import app, flags
from itertools import count
import gym
import jax

import wrappers
from dataset_utils import D4RLDataset, split_into_trajectories
from evaluation import evaluate
from learner_imitate import Learner, compute_rewards
from common import Batch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import AtariWrapper
from wrappers.atari_wrapper import (
    ScaledFloatFrame,
    FrameStack,
    FrameStackEager,
    PyTorchFrame,
)
from wrappers.normalize_action_wrapper import check_and_normalize_box_actions

FLAGS = flags.FLAGS

flags.DEFINE_string("env_name", "hopper-medium-v2", "Environment name.")
flags.DEFINE_string("model_path", "./tmp/", "Path to the trained model.")
flags.DEFINE_string("save_dir", "./results/correlation/", "Directory to save plots.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer(
    "eval_episodes", 100, "Number of episodes for correlation analysis."
)
flags.DEFINE_float("gamma", 0.99, "Discount factor.")
flags.DEFINE_float("temp", 1.0, "Loss temperature")
flags.DEFINE_boolean("double", True, "Use double q-learning")


def make_atari(env):
    env = AtariWrapper(env)
    env = PyTorchFrame(env)
    env = FrameStack(env, 4)
    return env


def is_atari(env_name):
    return env_name in [
        "PongNoFrameskip-v4",
        "BreakoutNoFrameskip-v4",
        "SpaceInvadersNoFrameskip-v4",
        "BeamRiderNoFrameskip-v4",
        "QbertNoFrameskip-v4",
        "SeaquestNoFrameskip-v4",
    ]


def make_env(env_name, mointor=True):
    env = gym.make(env_name)
    env = Monitor(env, "gym")
    if is_atari(env_name):
        env = make_atari(env)
    # Normalize box actions to [-1, 1]
    env = check_and_normalize_box_actions(env)
    return env


def make_agent(env, model_path):
    """Load trained agent."""
    observation_sample = env.observation_space.sample()[np.newaxis]
    action_sample = env.action_space.sample()[np.newaxis]

    agent = Learner(
        FLAGS.seed, observation_sample, action_sample, max_steps=100, double_q=FLAGS.double
    )

    agent.load(model_path)
    return agent


def eps(rewards):
    """Calculate episode returns."""
    return [sum(x) for x in rewards]


def part_eps(rewards):
    """Calculate partial episode returns."""
    return [np.cumsum(x) for x in rewards]


def measure_correlations(agent, env, gamma=0.99, episodes=100, save_dir=None):
    """Measure correlations between environment rewards and learned rewards."""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    env_rewards = []
    learnt_rewards = []

    for episode in range(episodes):
        part_env_rewards = []
        part_learnt_rewards = []

        state = env.reset()
        done = False
        episode_reward = 0
        episode_learnt_reward = 0

        states = []
        actions = []
        rewards = []
        next_states = []
        masks = []
        dones = []

        # Collect trajectory
        while not done:
            action = agent.sample_actions(
                np.expand_dims(state, axis=0),
                # temperature=0.0
            )[0]

            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            masks.append(1.0 - float(done))
            dones.append(float(done))

            q = agent.critic.apply(
                {"params": agent.critic.params}, np.array(state), np.array(action)
            )
            next_v = agent.value.apply(
                {"params": agent.value.params}, np.array(next_state)
            )
            
            y = (1 - done) * gamma * next_v
            irl_reward = jax.numpy.minimum(q[0], q[1]) - y

            episode_reward += reward
            part_env_rewards.append(reward)
            part_learnt_rewards.append(irl_reward)
            episode_learnt_reward += irl_reward

            state = next_state

        print(
            f"Episode {episode+1}\tEnvironment rewards: {episode_reward:.2f}\tLearned rewards: {episode_learnt_reward:.2f}"
        )

        env_rewards.append(part_env_rewards)
        learnt_rewards.append(part_learnt_rewards)

    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(eps(learnt_rewards), eps(env_rewards))
    pearson_corr, pearson_p = pearsonr(eps(learnt_rewards), eps(env_rewards))

    print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
    print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {pearson_p:.4e})")

    if save_dir:
        # Plot 1: Episode rewards
        sns.set()
        plt.figure(dpi=150)
        plt.scatter(eps(env_rewards), eps(learnt_rewards), s=10, alpha=0.8)
        plt.xlabel("Environment Returns")
        plt.ylabel("Learned Returns")
        plt.title(f"Episode rewards\nPearson Correlation: {pearson_corr:.4f}")
        plt.savefig(os.path.join(save_dir, "Episode_rewards.png"))
        plt.close()

        # Plot 2: Partial rewards (scatter)
        sns.set()
        plt.figure(dpi=150)
        for i in range(min(20, len(env_rewards))):
            plt.scatter(
                part_eps(env_rewards)[i], part_eps(learnt_rewards)[i], s=5, alpha=0.6
            )
        plt.xlabel("Environment rewards")
        plt.ylabel("Learned rewards")
        plt.savefig(os.path.join(save_dir, "Partial_rewards.png"))
        plt.close()

        # Plot 3: Partial rewards (interpolated)
        sns.set()
        plt.figure(dpi=150)
        for i in range(min(20, len(env_rewards))):
            plt.plot(
                part_eps(env_rewards)[i],
                part_eps(learnt_rewards)[i],
                markersize=1,
                alpha=0.8,
            )
        plt.xlabel("Environment rewards")
        plt.ylabel("Learned rewards")
        plt.savefig(os.path.join(save_dir, "Partial_rewards_interpolated.png"))
        plt.close()

        # Plot 4: Step rewards
        sns.set()
        plt.figure(dpi=150)
        for i in range(min(5, len(env_rewards))):
            plt.scatter(env_rewards[i], learnt_rewards[i], s=5, alpha=0.5)
        plt.xlabel("Environment rewards")
        plt.ylabel("Learned rewards")
        plt.savefig(os.path.join(save_dir, "Step_rewards.png"))
        plt.close()

    return {
        "spearman_corr": spearman_corr,
        "pearson_corr": pearson_corr,
        "env_rewards": env_rewards,
        "learnt_rewards": learnt_rewards,
    }


# def analyze_offline_dataset(agent, dataset, save_dir):
#     """Analyze correlation between ground truth and learned rewards for offline dataset."""
#     # Process entire dataset with learned rewards
#     dataset_dict = {
#         "observations": dataset.observations.copy(),
#         "actions": dataset.actions.copy(),
#         "rewards": dataset.rewards.copy(),
#         "masks": dataset.masks.copy(),
#         "dones_float": dataset.dones_float.copy(),
#         "next_observations": dataset.next_observations.copy(),
#     }

#     # Compute learned rewards for the dataset in batches
#     batch_size = 1024
#     total_entries = len(dataset.observations)
#     num_batches = (total_entries + batch_size - 1) // batch_size
#     learned_rewards = np.zeros_like(dataset.rewards)

#     for i in range(num_batches):
#         start_idx = i * batch_size
#         end_idx = min(start_idx + batch_size, total_entries)

#         batch = Batch(
#             observations=dataset.observations[start_idx:end_idx],
#             actions=dataset.actions[start_idx:end_idx],
#             rewards=dataset.rewards[start_idx:end_idx],
#             masks=dataset.masks[start_idx:end_idx],
#             next_observations=dataset.next_observations[start_idx:end_idx],
#         )

#         learned_rewards[start_idx:end_idx] = compute_rewards(agent, batch)

#     dataset_dict["learned_rewards"] = learned_rewards

#     # Analyze correlations using trajectories
#     trajectories = split_into_trajectories(
#         dataset_dict["observations"],
#         dataset_dict["actions"],
#         dataset_dict["rewards"],
#         dataset_dict["masks"],
#         dataset_dict["dones_float"],
#         dataset_dict["next_observations"],
#     )

#     learned_trajectories = split_into_trajectories(
#         dataset_dict["observations"],
#         dataset_dict["actions"],
#         dataset_dict["learned_rewards"],
#         dataset_dict["masks"],
#         dataset_dict["dones_float"],
#         dataset_dict["next_observations"],
#     )

#     # Calculate returns for each trajectory
#     gt_returns = [sum(traj[2]) for traj in trajectories]
#     learned_returns = [sum(traj[2]) for traj in learned_trajectories]
#     part_gt_returns = [np.cumsum(traj[2]) for traj in trajectories]
#     part_learned_returns = [np.cumsum(traj[2]) for traj in learned_trajectories]

#     # Calculate correlations
#     pearson_corr, p_value = pearsonr(gt_returns, learned_returns)
#     spearman_corr, spearman_p = spearmanr(gt_returns, learned_returns)

#     print(f"Pearson correlation: {pearson_corr:.4f} (p-value: {p_value:.4e})")
#     print(f"Spearman correlation: {spearman_corr:.4f} (p-value: {spearman_p:.4e})")
#     print(f"Number of trajectories analyzed: {len(gt_returns)}")

#     if save_dir:
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         # Plot 1: Episode returns correlation
#         sns.set()
#         plt.figure(dpi=150)
#         plt.scatter(gt_returns, learned_returns, s=10, alpha=0.8)
#         plt.xlabel("Ground Truth Returns")
#         plt.ylabel("Learned Returns")
#         plt.title(f"Episode rewards\nPearson Correlation: {pearson_corr:.4f}")
#         plt.savefig(os.path.join(save_dir, "Offline_episode_returns.png"))
#         plt.close()

#         # Plot 2: Partial returns (scatter)
#         sns.set()
#         plt.figure(dpi=150)
#         for i in range(min(20, len(part_gt_returns))):
#             plt.scatter(part_gt_returns[i], part_learned_returns[i], s=5, alpha=0.6)
#         plt.xlabel("Ground Truth rewards")
#         plt.ylabel("Learned rewards")
#         plt.savefig(os.path.join(save_dir, "Offline_partial_rewards.png"))
#         plt.close()

#         # Plot 3: Partial returns (interpolated)
#         sns.set()
#         plt.figure(dpi=150)
#         for i in range(min(20, len(part_gt_returns))):
#             plt.plot(
#                 part_gt_returns[i], part_learned_returns[i], markersize=1, alpha=0.8
#             )
#         plt.xlabel("Ground Truth rewards")
#         plt.ylabel("Learned rewards")
#         plt.savefig(os.path.join(save_dir, "Offline_partial_rewards_interpolated.png"))
#         plt.close()

#         # Plot 4: Step rewards
#         sns.set()
#         plt.figure(dpi=150)
#         for i in range(min(5, len(trajectories))):
#             plt.scatter(trajectories[i][2], learned_trajectories[i][2], s=5, alpha=0.5)
#         plt.xlabel("Ground Truth rewards")
#         plt.ylabel("Learned rewards")
#         plt.savefig(os.path.join(save_dir, "Offline_step_rewards.png"))
#         plt.close()

#     return {
#         "pearson_corr": pearson_corr,
#         "spearman_corr": spearman_corr,
#         "gt_returns": gt_returns,
#         "learned_returns": learned_returns,
#         "dataset_dict": dataset_dict,
#     }


def main(_):
    env = make_env(FLAGS.env_name)
    agent = make_agent(env, FLAGS.model_path)
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)
    # Online correlation measurement
    print("Measuring online correlations...")
    online_save_dir = os.path.join(FLAGS.save_dir, "online")
    online_results = measure_correlations(
        agent=agent,
        env=env,
        gamma=FLAGS.gamma,
        episodes=FLAGS.eval_episodes,
        save_dir=online_save_dir,
    )

    # # Offline correlation measurement using the dataset
    # print("\nAnalyzing offline dataset correlations...")
    # offline_save_dir = os.path.join(FLAGS.save_dir, 'offline')
    # offline_results = analyze_offline_dataset(
    #     agent=agent,
    #     dataset=dataset,
    #     save_dir=offline_save_dir
    # )

    # print("\nAnalysis complete. Results saved to:", FLAGS.save_dir)


if __name__ == "__main__":
    app.run(main)
