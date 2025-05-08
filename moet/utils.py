from tensorflow_probability.substrates import jax as tfp
from jaxtyping import Float, Array, Bool, Integer
import jax.numpy as jnp
import jax

from functools import partial
from genjax import exact_density
import numpy as np
from tqdm import tqdm

tfd = tfp.distributions

@jax.jit
def get_all_joint_counts(
    x: Float[Array, "batch_size n_categories max_categories"], 
) -> Float[Array, "n_categories n_categories max_categories max_categories"]:
    return jax.vmap(jax.vmap(joint_counts, in_axes=(1, None)), in_axes=(None, 1))(x, x)

def joint_counts(
    x: Float[Array, "batch_size input_dim"], y: Float[Array, "batch_size input_dim"]
) -> Float[Array, ""]:
    x_nonmissing = jnp.sum(x, axis=1)
    y_nonmissing = jnp.sum(y, axis=1)
    n = jnp.sum(jnp.logical_and(x_nonmissing, y_nonmissing))
    sum_xy = x.T @ y
    return sum_xy, n

@jax.jit
def get_all_counts(x: Float[Array, "batch_size n_categories max_categories"]):
    return jax.vmap(counts, in_axes=(1,))(x)

def weighted_marginals(data, weights=None, batch_size=10000):
    n_categories, max_categories = data.shape[1:]
    sum_x = jnp.zeros((n_categories, max_categories))
    n_x = jnp.zeros(n_categories)
    sum_xy = jnp.zeros((n_categories, n_categories, max_categories, max_categories))
    n_xy = jnp.zeros((n_categories, n_categories))
    for i in tqdm(range(0, data.shape[0], batch_size)):
        batch = np.copy(data[i:i+batch_size])
        preprocessed_batch = count_preprocess(batch)
        if weights is not None:
            preprocessed_batch = preprocessed_batch * weights[i:i+batch_size][:, None, None]
        batch_sum_x, batch_n_x = get_all_counts(preprocessed_batch)
        sum_x = sum_x + batch_sum_x
        n_x = n_x + batch_n_x
        batch_sum_xy, batch_n_xy = get_all_joint_counts(preprocessed_batch)
        sum_xy = sum_xy + batch_sum_xy
        n_xy = n_xy + batch_n_xy
        if np.isnan(batch).any():
            import ipdb; ipdb.set_trace()
        del batch, preprocessed_batch

    p_x = sum_x / n_x[:, None]
    p_xy = sum_xy / n_xy[:, :, None, None]

    return p_x, p_xy

def counts(x: Float[Array, "batch_size max_categories"]):
    x_sum = jnp.sum(x, axis=0)
    n = jnp.sum(x_sum)
    return x_sum, n

def mi_from_logp(logp_xy, logp_x, logp_y):
    retval = jnp.sum(
        jnp.where(jnp.exp(logp_xy) > 0, jnp.exp(logp_xy) * (logp_xy - logp_x[:, None] - logp_y[None, :]), 0)
    )
    return retval

def get_l_candidates(ps_sorted, beta):
    N = ps_sorted.shape[0]
    cumulative_pdata = jnp.arange(1, N+1) / N 
    cumulative_pmodel = jnp.cumsum(ps_sorted)
    # algorithm 2 from adagan: https://arxiv.org/pdf/1701.02386
    # sort logps in increasing order
    factor1 = beta  / cumulative_pdata
    factor2 = (1 + (1-beta) * cumulative_pmodel / beta)

    l_candidates = factor1 * factor2
    return l_candidates


def get_l(logps, beta):
    N = logps.shape[0]
    logps_sorted = jnp.sort(logps)
    ps_sorted = jnp.exp(logps_sorted)
    l_candidates = get_l_candidates(ps_sorted, beta)

    condition = l_candidates <= (1-beta) * ps_sorted * N
    masked_candidates = jnp.where(condition, l_candidates, jnp.inf)
    l = jnp.min(masked_candidates)
    return l

def count_preprocess(x: Float[Array, "batch_size n_categories max_categories"]):
    x_exp = jnp.exp(x)
    missing = jnp.all(x_exp == 1, axis=1)
    return x_exp - missing[:, None]

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

def get_all_mis(p_x: Float[Array, "n_categories max_categories"], p_xy: Float[Array, "n_categories n_categories max_categories max_categories"]):
    return jax.vmap(jax.vmap(get_mi, in_axes=(0, None, 0)), in_axes=(None, 0, 0))(p_x, p_x, p_xy)

def get_mi(p_x: Float[Array, "max_categories"], p_y: Float[Array, "max_categories"], p_xy: Float[Array, "max_categories max_categories"]):
    return jnp.sum(jnp.where(p_xy > 0, p_xy * (jnp.log(p_xy) - jnp.log(p_x[:, None]) - jnp.log(p_y[None, :])), 0))

def get_theta(mi, temperature=1e-5, eps=1e-10):
    theta = jnp.where(jnp.eye(mi.shape[0]), eps, mi)
    theta = theta / temperature
    logz = jax.nn.logsumexp(theta.flatten(), axis=-1)
    theta = theta - logz
    return theta

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