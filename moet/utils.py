from tensorflow_probability.substrates import jax as tfp
from jaxtyping import Float, Array, Bool, Integer
import jax.numpy as jnp
import jax

from functools import partial
from genjax import exact_density
import numpy as np


tfd = tfp.distributions


def get_marginals(
    x1: Float[Array, "batch_size input_dim"], x2: Float[Array, "batch_size input_dim"]
) -> Float[Array, ""]:
    N = x1.shape[0]
    n_missing1 = jnp.sum(jnp.all(x1 == 1, axis=1))
    n_missing2 = jnp.sum(jnp.all(x2 == 1, axis=1))
    p_xy = x1.T @ x2
    p_xy = p_xy / jnp.sum(p_xy)
    logp_xy = jnp.log(p_xy)
    logp_x = jnp.log(jnp.sum(x1, axis=0) / (N - n_missing1))
    logp_y = jnp.log(jnp.sum(x2, axis=0) / (N - n_missing2))
    return logp_xy, logp_x, logp_y

def mi_from_logp(logp_xy, logp_x, logp_y):
    retval = jnp.sum(
        jnp.where(jnp.exp(logp_xy) > 0, jnp.exp(logp_xy) * (logp_xy - logp_x[:, None] - logp_y[None, :]), 0)
    )
    return retval


def categorical2d_sample(key: jax.random.PRNGKey, logits: Float[Array, "n n"]):
    logits_perturbed = logits + jax.random.gumbel(key, shape=logits.shape)
    max_val = jnp.max(logits_perturbed)

    return jnp.argwhere(logits_perturbed == max_val, size=1)[0]


def categorical2d_logpdf(v: Integer[Array, "2"], logits: Float[Array, "n n"]):
    return logits.at[v[0], v[1]].get()


categorical2d = exact_density(
    categorical2d_sample, categorical2d_logpdf, "categorical2d"
)


def to_tuple(x):
    try:
        len(x)
        return tuple(to_tuple(element) for element in x)
    except TypeError:
        return x

def get_all_marginals(train_data, n_mi_samples=10000, eps=1e-10):
    jitted_marginals = jax.jit(jax.vmap(jax.vmap(get_marginals, in_axes=(1, None)), in_axes=(None, 1)))
    bool_data = (train_data[:n_mi_samples, :] == 0).astype(jnp.uint8)
    logp_xy, logp_x, logp_y = jitted_marginals(
        bool_data,
        bool_data,
    )
    return logp_xy, logp_x, logp_y

# def get_mi_theta(train_data, n_mi_samples=10000, eps=1e-10):
#     jitted_mi = jax.jit(jax.vmap(jax.vmap(mi, in_axes=(1, None)), in_axes=(None, 1)))
#     bool_data = (train_data[:n_mi_samples, :] == 0).astype(jnp.uint8)
#     mi_estimates = jitted_mi(
#         bool_data,
#         bool_data,
#     )
#     theta = jnp.maximum(mi_estimates, eps)
#     theta = jnp.where(jnp.eye(len(mi_estimates)), 0, theta)
#     theta /= jnp.sum(theta)
#     return jnp.log(theta)

def make_1d_obs(var_idx, feature_idx, var_size, feature_size):
    obs = jnp.zeros((var_size, feature_size))
    obs = obs.at[var_idx].set(-jnp.inf)
    obs = obs.at[var_idx, feature_idx].set(0)
    return obs

def make_2d_obs(var_idx1, var_idx2, feature_idx1, feature_idx2, var_size, feature_size):
    obs = jnp.zeros((var_size, feature_size))
    obs = obs.at[var_idx1].set(-jnp.inf)
    obs = obs.at[var_idx2].set(-jnp.inf)
    obs = obs.at[var_idx1, feature_idx1].set(0)
    obs = obs.at[var_idx2, feature_idx2].set(0)
    return obs

@partial(jax.jit, static_argnums=(0, 1))
def make_all_1d_obs(var_size, feature_size):
    return jax.vmap(jax.vmap(make_1d_obs, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None))(
        jnp.arange(var_size), jnp.arange(feature_size), var_size, feature_size)

@partial(jax.jit, static_argnums=(0, 1))
def make_all_2d_obs(var_size, feature_size):
    return jax.vmap(
            jax.vmap(
                jax.vmap(
                    jax.vmap(
                        make_2d_obs, in_axes=(0, None, None, None, None, None))
                    , in_axes=(None, 0, None, None, None, None)
                ), in_axes=(None, None, 0, None, None, None),
            ), in_axes=(None, None, None, 0, None, None))(
        jnp.arange(var_size), jnp.arange(feature_size), jnp.arange(var_size), jnp.arange(feature_size), var_size, feature_size)

# %%

