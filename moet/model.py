from jaxtyping import Float, Array, Integer, jaxtyped, PyTree
import jax.numpy as jnp
import jax
from beartype import beartype as typechecker
from functools import partial


@jaxtyped(typechecker=typechecker)
def circuit_layer(
    X: Float[Array, "n_inputs input_dim"],
    Q: Float[Array, "n_inputs output_dim input_dim"],
    merge: PyTree[Integer[Array, ""]],
) -> Float[Array, "n_outputs output_dim"]:
    n_inputs = X.shape[0]
    k, arity = len(merge), len(merge[0])
    Y: Float[Array, "n_inputs output_dim"] = jax.nn.logsumexp(
        Q + X[:, None, :], axis=-1
    )

    merge_flat, treedef = jax.tree.flatten(merge)
    merge_flat: Integer[Array, "k_times_arity"] = jnp.array(merge_flat)
    not_in_merge = ~jnp.isin(jnp.arange(n_inputs), merge_flat)
    pass_through = jnp.argwhere(not_in_merge, size=n_inputs - k * arity)[:, 0]
    Y_pass_through: Float[Array, "n_inputs-k_times_arity output_dim"] = Y[pass_through]

    Y_merge_flat: Float[Array, "k_times_arity output_dim"] = Y[merge_flat]
    Y_merge_tree: PyTree[Float[Array, "output_dim"]] = jax.tree.unflatten(
        treedef, Y_merge_flat
    )
    Y_merge: Float[Array, "k arity output_dim"] = jnp.stack(
        [jnp.stack(y) for y in Y_merge_tree]
    )
    Y_merge: Float[Array, "k output_dim"] = Y_merge.sum(axis=1)

    output: Float[Array, "k+n_inputs-k_times_arity output_dim"] = jnp.concatenate(
        [Y_merge, Y_pass_through], axis=0
    )
    return output


@jaxtyped(typechecker=typechecker)
def circuit(
    X: Float[Array, "n_inputs input_dim"],
    Qs: PyTree[Float[Array, "?n_inputs ?output_dim ?input_dim"], "T"],
    W: Float[Array, "n_outputs"],
    layers: PyTree[Integer[Array, ""]],
) -> Float[Array, ""]:
    n_inputs = X.shape[0]
    for Q, layer in zip(Qs, layers):
        n_outputs = n_inputs // len(layer[0])
        X = circuit_layer(X, Q, layer)
        n_inputs -= n_outputs

    assert X.shape[0] == 1
    X = W + X[0]
    return jax.nn.logsumexp(X)


@jax.jit
def loss_fn(
    X: Float[Array, "batch_size n_inputs input_dim"],
    Qs: PyTree[Float[Array, "?n_inputs ?output_dim ?input_dim"], "T"],
    W: Float[Array, "n_outputs"],
    layers: PyTree[int],
) -> Float[Array, ""]:
    n_batch = X.shape[0]
    pad = jnp.zeros_like(X[0:1])
    X_pad = jnp.concatenate([X, pad], axis=0)
    out = jax.vmap(circuit, in_axes=(0, None, None, None))(X_pad, Qs, W, layers)
    log_Z = out[-1]
    return -(jnp.sum(out[:-1]) - n_batch * log_Z) / n_batch
