from typing import Tuple

import jax.numpy as jnp
import jax
from functools import partial

from common import Batch, InfoDict, Model, Params, PRNGKey


def rkl_loss_imitate(diff,v, alpha,beta, args=None):
    z = diff/alpha
    if args.max_clip is not None:
        z = jnp.minimum(z, args.max_clip) # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = (jnp.exp(z - max_z) - z*jnp.exp(-max_z) - jnp.exp(-max_z))[:z.shape[0]//2] # scale by e^max_z
    return loss


def rkl_implicit_maximizer(diff,v, alpha,beta, args=None):
    z = diff/alpha
    if args.max_clip is not None:
        z = jnp.minimum(z, args.max_clip) # clip max value
    max_z = jnp.max(z, axis=0)
    max_z = jnp.where(max_z < -1.0, -1.0, max_z)
    max_z = jax.lax.stop_gradient(max_z)  # Detach the gradients
    loss = (jnp.exp(z - max_z) - z*jnp.exp(-max_z) - jnp.exp(-max_z))[:z.shape[0]//2] # scale by e^max_z
    return loss

def expectile_loss_imitate(diff,v, beta, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return  (weight * (diff**2))[:v.shape[0]//2].mean() + beta*(v[:v.shape[0]//2]-v[v.shape[0]//2:]).mean()



def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)



def update_v_imitate(critic: Model, value: Model, batch: Batch, is_expert_mask,
             expectile: float, loss_temp: float, alpha:float, beta:float, double: bool, vanilla: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:
    actions = batch.actions

    rng1, rng2 = jax.random.split(key)
    if args.sample_random_times > 0:
        # add random actions to smooth loss computation (use 1/2(rho + Unif))
        times = args.sample_random_times
        random_action = jax.random.uniform(
            rng1, shape=(times * actions.shape[0],
                         actions.shape[1]),
            minval=-1.0, maxval=1.0)
        obs = jnp.concatenate([batch.observations, jnp.repeat(
            batch.observations, times, axis=0)], axis=0)
        acts = jnp.concatenate([batch.actions, random_action], axis=0)
    else:
        obs = batch.observations
        acts = batch.actions

    if args.noise:
        std = args.noise_std
        noise = jax.random.normal(rng2, shape=(acts.shape[0], acts.shape[1]))
        noise = jnp.clip(noise * std, -0.5, 0.5)
        acts = (batch.actions + noise)
        acts = jnp.clip(acts, -1, 1)

    q1, q2 = critic(obs, acts) # this is target critic
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        if vanilla:
            value_loss = expectile_loss_imitate(q - v,v,beta, expectile).mean()
        else:

            value_loss = rkl_loss_imitate(q - v,v, alpha=loss_temp,beta=beta, args=args).mean()
        return value_loss, {
            'unseen_v_expert': (v[:v.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
            'unseen_v_suboptimal':(v[:v.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    
    return new_value, info

def update_v_recoil(critic: Model, value: Model, batch: Batch, is_expert_mask,
             expectile: float, loss_temp: float, alpha:float, beta:float, double: bool, vanilla: bool, key: PRNGKey, args) -> Tuple[Model, InfoDict]:
    actions = batch.actions

    rng1, rng2 = jax.random.split(key)
    if args.sample_random_times > 0:
        # add random actions to smooth loss computation (use 1/2(rho + Unif))
        times = args.sample_random_times
        random_action = jax.random.uniform(
            rng1, shape=(times * actions.shape[0],
                         actions.shape[1]),
            minval=-1.0, maxval=1.0)
        obs = jnp.concatenate([batch.observations, jnp.repeat(
            batch.observations, times, axis=0)], axis=0)
        acts = jnp.concatenate([batch.actions, random_action], axis=0)
    else:
        obs = batch.observations
        acts = batch.actions

    if args.noise:
        std = args.noise_std
        noise = jax.random.normal(rng2, shape=(acts.shape[0], acts.shape[1]))
        noise = jnp.clip(noise * std, -0.5, 0.5)
        acts = (batch.actions + noise)
        acts = jnp.clip(acts, -1, 1)

    q1, q2 = critic(obs, acts) # this is target critic
    if double:
        q = jnp.minimum(q1, q2)
    else:
        q = q1

    def value_loss_fn(value_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        v = value.apply({'params': value_params}, obs)
        if vanilla:
            value_loss = expectile_loss_imitate(q - v,v,beta, expectile).mean()
        else:
            value_loss = rkl_loss_imitate(q - v,v, alpha=loss_temp,beta=beta, args=args).mean()
        return value_loss, {
            'unseen_v_expert': (v[:v.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
            'unseen_v_suboptimal':(v[:v.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
            'value_loss': value_loss,
            'v': v.mean(),
        }

    new_value, info = value.apply_gradient(value_loss_fn)
    
    return new_value, info



def update_q_imitate(critic: Model, target_value: Model, batch: Batch, is_expert_mask,
             discount: float, double: bool, key: PRNGKey, loss_temp: float, args) -> Tuple[Model, InfoDict]:
    next_v = target_value(batch.next_observations)
    ###### LSIQ stability trick!
    target_q_imitate = -2 + discount * batch.masks * next_v + discount * (1-batch.masks) * (-200)
    # target_q_imitate = -10 + discount * batch.masks * next_v + discount * (1-batch.masks) * (-1000)
    # target_q_imitate = 0 + discount * batch.masks * next_v + discount * (1-batch.masks) * (0)
    target_q = target_q_imitate
    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        acts = batch.actions
        q1, q2 = critic.apply({'params': critic_params}, batch.observations, acts)
        v = target_value(batch.observations)

        def mse_loss(q, q_target, *args):
            loss_dict = {}

            x = (q-q_target)[:q.shape[0]//2]
            loss = x**2
            # loss = huber_loss(x, delta=20.0)  # Huber loss didnt work well here
            loss_dict['critic_loss'] = loss.mean()

            return loss.mean(), loss_dict

        def ls_iq_loss(q,q_target,*args):
            loss_dict = {}
            loss = 0.5*(q[q.shape[0]//2:]-200)**2+0.5*(q[:q.shape[0]//2]-q_target[:q.shape[0]//2])**2
            loss_dict['critic_loss'] = loss.mean()

            return loss.mean(), loss_dict
        
        # critic_loss = mse_loss
        critic_loss = ls_iq_loss #mse_loss

        if double:
            loss1, dict1 = critic_loss(q1, target_q, v, loss_temp)
            loss2, dict2 = critic_loss(q2, target_q, v, loss_temp)

            critic_loss = (loss1 + loss2).mean()
            for k, v in dict2.items():
                dict1[k] += v
            loss_dict = dict1
        else:
            # critic_loss, loss_dict = dual_q_loss(q1, target_q, v, loss_temp)
            critic_loss, loss_dict = critic_loss(q1, target_q,  v, loss_temp)

        if args.grad_pen:
            # print("Using grad_pen")
            lambda_ =args.lambda_gp
            q1_grad, q2_grad = grad_norm(critic, critic_params, batch.observations, acts)
            loss_dict['q1_grad'] = q1_grad.mean()
            loss_dict['q2_grad'] = q2_grad.mean()

            if double:
                gp_loss = (q1_grad + q2_grad).mean()
            else:
                gp_loss = q1_grad.mean()

            critic_loss += lambda_ * gp_loss
        loss_dict.update({
            'unseen_q_expert':(q1[:q1.shape[0]//2]*is_expert_mask).sum()/is_expert_mask.sum(),
            'unseen_q_suboptimal':(q1[:q1.shape[0]//2]*(1-is_expert_mask)).sum()/(1-is_expert_mask).sum(),
            'q1': q1.mean(),
            'q2': q2.mean()
        })
        return critic_loss, loss_dict

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, info


def grad_norm(model, params, obs, action, lambda_=10):

    @partial(jax.vmap, in_axes=(0, 0))
    @partial(jax.jacrev, argnums=1)
    def input_grad_fn(obs, action):
        return model.apply({'params': params}, obs, action)

    def grad_pen_fn(grad):
        # We use gradient penalties inspired from WGAN-LP loss which penalizes grad_norm > 1
        penalty = jnp.maximum(jnp.linalg.norm(grad1, axis=-1) - 1, 0)**2
        return penalty

    grad1, grad2 = input_grad_fn(obs, action)

    return grad_pen_fn(grad1), grad_pen_fn(grad2)


def huber_loss(x, delta: float = 1.):
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero.
    See "Robust Estimation of a Location Parameter" by Huber.
    (https://projecteuclid.org/download/pdf_1/euclid.aoms/1177703732).
    Args:
    x: a vector of arbitrary shape.
    delta: the bounds for the huber loss transformation, defaults at 1.
    Note `grad(huber_loss(x))` is equivalent to `grad(0.5 * clip_gradient(x)**2)`.
    Returns:
    a vector of same shape of `x`.
    """
    # 0.5 * x^2                  if |x| <= d
    # 0.5 * d^2 + d * (|x| - d)  if |x| > d
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta)
    # Same as max(abs_x - delta, 0) but avoids potentially doubling gradient.
    linear = abs_x - quadratic
    return 0.5 * quadratic**2 + delta * linear

