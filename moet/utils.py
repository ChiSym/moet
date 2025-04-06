from genjax.adev import reinforce
from tensorflow_probability.substrates import jax as tfp
from jaxtyping import Float, Array, Bool, Integer
import jax.numpy as jnp
import jax
from genjax import exact_density

tfd = tfp.distributions


def cat_logpdf(v, logits):
    return tfd.Categorical(logits=logits).log_prob(v)


cat_reinforce = reinforce(
    lambda key, logits: tfd.Categorical(logits=logits).sample(seed=key),
    cat_logpdf,
)


def mi(
    x1: Float[Array, "batch_size input_dim"], x2: Float[Array, "batch_size input_dim"]
) -> Float[Array, ""]:
    p_xy = x1.T @ x2
    p_xy = p_xy / jnp.sum(p_xy)
    logp_xy = jnp.log(p_xy)
    logp_x = jnp.log(jnp.sum(x1, axis=0) / jnp.sum(x1))
    logp_y = jnp.log(jnp.sum(x2, axis=0) / jnp.sum(x2))

    retval = jnp.sum(
        jnp.where(p_xy > 0, p_xy * (logp_xy - logp_x[:, None] - logp_y[None, :]), 0)
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
