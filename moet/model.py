from jaxtyping import Float, Array, Integer
import jax.numpy as jnp
import jax


def layer(
    X: Float[Array, "n_inputs input_dim"],
    Q: Float[Array, "n_inputs output_dim input_dim"],
    merge: Integer[Array, "k arity"],
) -> Float[Array, "k+n_inputs-k*arity output_dim"]:
    n_inputs = X.shape[0]
    k, arity = merge.shape
    Y: Float[Array, "n_inputs output_dim"] = jax.nn.logsumexp(
        Q + X[:, None, :], axis=-1
    )
    not_in_merge = ~jnp.isin(jnp.arange(n_inputs), merge)
    pass_through = jnp.argwhere(not_in_merge, size=n_inputs - k * arity)[:, 0]

    Y_merge = jax.vmap(lambda m: Y[m, :].sum(axis=0))(merge)
    Y_pass_through = Y[pass_through, :]

    return jnp.concatenate([Y_merge, Y_pass_through], axis=0)
