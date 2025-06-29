"""Implementations of algorithms for continuous control."""

from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import os

import policy
import value_net
from actor import update_imitate as awr_update_actor
from common import Batch, MixedBatch, InfoDict, Model, PRNGKey
from critic_imitate import update_q_imitate, update_v_imitate

from functools import partial


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda p, tp: p * tau + tp * (1 - tau), critic.params,
        target_critic.params)

    return target_critic.replace(params=new_target_params)


@partial(jax.jit, static_argnames=['double', 'vanilla', 'args'])
def _update_jit(
    rng: PRNGKey, actor: Model, critic: Model, value: Model,
    target_critic: Model, batch: MixedBatch,expert_batch: Batch, discount: float, tau: float,
    expectile: float, temperature: float, loss_temp: float,alpha: float, beta: float, double: bool, vanilla: bool, args,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, InfoDict]:
    
    combined_observations = jnp.concatenate((batch.observations,expert_batch.observations),axis=0)
    combined_actions = jnp.concatenate((batch.actions,expert_batch.actions),axis=0)
    combined_rewards = jnp.concatenate((batch.rewards,expert_batch.rewards),axis=0)*0
    combined_masks = jnp.concatenate((batch.masks,expert_batch.masks),axis=0)
    combined_next_observations = jnp.concatenate((batch.next_observations,expert_batch.next_observations),axis=0)
    combined_batch = Batch(observations=combined_observations, actions=combined_actions, rewards=combined_rewards, masks=combined_masks, next_observations=combined_next_observations)
    is_expert_mask = batch.is_expert
    key, rng = jax.random.split(rng)
    for i in range(args.num_v_updates):
        new_value, value_info = update_v_imitate(target_critic, value, combined_batch,is_expert_mask, expectile, loss_temp, alpha,beta, double, vanilla, key, args)
        # value = new_value # TODO: parity with IQL_pytorch
    new_actor, actor_info = awr_update_actor(key, actor, target_critic,
                                             new_value, combined_batch,is_expert_mask, temperature, double)

    new_critic, critic_info = update_q_imitate(critic, new_value, combined_batch,is_expert_mask, discount, double, key, loss_temp, args)

    new_target_critic = target_update(new_critic, target_critic, tau)

    return rng, new_actor, new_critic, new_value, new_target_critic, {
        **critic_info,
        **value_info,
        **actor_info
    }


def compute_rewards(agent, batch: Batch, num_samples: int = 10) -> jnp.ndarray:
    """Computes rewards for a batch of transitions."""

    @jax.jit
    def _get_rewards(
        actor_params: jnp.ndarray,
        critic_params: jnp.ndarray,
        value_params: jnp.ndarray,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> jnp.ndarray:
        # Compute Q(s, a)
        q1_s_a, q2_s_a = agent.critic.apply_fn.apply({'params': critic_params}, observations, actions)
        q_s_a = jnp.minimum(q1_s_a, q2_s_a)

        next_v = agent.value.apply_fn.apply({'params': value_params}, next_observations)
        rewards = q_s_a - masks * agent.discount * next_v

        return rewards

    rewards = _get_rewards(
        agent.actor.params, agent.critic.params, agent.value.params,
        batch.observations, batch.actions, batch.next_observations, batch.masks
    )
    return rewards


class Learner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 value_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (256, 256),
                 discount: float = 0.99,
                 tau: float = 0.005,
                 expectile: float = 0.8,
                 temperature: float = 0.1,
                 dropout_rate: Optional[float] = None,
                 layernorm: bool = False,
                 value_dropout_rate: Optional[float] = None,
                 max_steps: Optional[int] = None,
                 loss_temp: float = 1.0,
                 double_q: bool = True,
                 vanilla: bool = True,
                 opt_decay_schedule: str = "cosine",
                 alpha: float = 0.7,
                 beta: float = 0.1,
                 args=None):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1801.01290
        """

        self.expectile = expectile
        self.tau = tau
        self.discount = discount
        self.temperature = temperature
        self.loss_temp = loss_temp
        self.double_q = double_q
        self.vanilla = vanilla
        self.alpha = alpha
        self.beta=beta
        self.args = args
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        action_dim = actions.shape[-1]
        actor_def = policy.NormalTanhPolicy(hidden_dims,
                                            action_dim,
                                            log_std_scale=1e-3,
                                            log_std_min=-5.0,
                                            dropout_rate=dropout_rate,
                                            state_dependent_std=False,
                                            tanh_squash_distribution=False)

        if opt_decay_schedule == "cosine":
            schedule_fn = optax.cosine_decay_schedule(-actor_lr, max_steps)
            optimiser = optax.chain(optax.scale_by_adam(),
                                    optax.scale_by_schedule(schedule_fn))
        else:
            optimiser = optax.adam(learning_rate=actor_lr)

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=optimiser)

        critic_def = value_net.DoubleCritic(hidden_dims)

        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=optax.adam(learning_rate=critic_lr))

        value_def = value_net.ValueCritic(hidden_dims,
                                          layer_norm=layernorm,
                                          dropout_rate=value_dropout_rate)
        value = Model.create(value_def,
                             inputs=[value_key, observations],
                             tx=optax.adam(learning_rate=value_lr))

        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])

        self.actor = actor
        self.critic = critic
        self.value = value
        self.target_critic = target_critic
        self.rng = rng

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> jnp.ndarray:
        rng, actions = policy.sample_actions(self.rng, self.actor.apply_fn,
                                             self.actor.params, observations,
                                             temperature)
        self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: MixedBatch, expert_batch: Batch) -> InfoDict:
        new_rng, new_actor, new_critic, new_value, new_target_critic, info = _update_jit(
            self.rng, self.actor, self.critic, self.value, self.target_critic,
            batch,expert_batch, self.discount, self.tau, self.expectile, self.temperature, self.loss_temp, self.alpha,self.beta,  self.double_q, self.vanilla, self.args)

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.value = new_value
        self.target_critic = new_target_critic

        return info

    def load(self, save_dir: str):
        self.actor = self.actor.load(os.path.join(save_dir, 'actor'))
        self.critic = self.critic.load(os.path.join(save_dir, 'critic'))
        self.value = self.value.load(os.path.join(save_dir, 'value'))
        self.target_critic = self.target_critic.load(os.path.join(save_dir, 'critic'))

    def save(self, save_dir: str):
        self.actor.save(os.path.join(save_dir, 'actor'))
        self.critic.save(os.path.join(save_dir, 'critic'))
        self.value.save(os.path.join(save_dir, 'value'))
