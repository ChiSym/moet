from jaxtyping import Float, Array, Integer, jaxtyped, PyTree, PyTreeDef
import jax.numpy as jnp
import jax
from beartype import beartype as typechecker
from functools import partial
import functools
from jax import tree_flatten, tree_unflatten

@jaxtyped(typechecker=typechecker)
def circuit_layer(
    X: Float[Array, "n_inputs input_dim"],
    Q: Float[Array, "n_inputs output_dim input_dim"],
    merge: PyTree[Integer[Array, ""]],
    normalize: bool = False,
):
    n_inputs = X.shape[0]
    k, arity = len(merge), len(merge[0])
    Y: Float[Array, "n_inputs output_dim"] = jax.nn.logsumexp(
        Q + X[:, None, :], axis=-1
    )
    merge_flat, treedef = tree_flatten(merge)
    merge_flat: Integer[Array, "k_times_arity"] = jnp.array(merge_flat)
    not_in_merge = ~jnp.isin(jnp.arange(n_inputs), merge_flat)
    pass_through = jnp.argwhere(not_in_merge, size=n_inputs - k * arity)[:, 0]
    Y_pass_through: Float[Array, "n_inputs-k_times_arity output_dim"] = Y[pass_through]

    Y_merge_flat: Float[Array, "k_times_arity output_dim"] = Y[merge_flat]
    Y_merge_tree: PyTree[Float[Array, "output_dim"]] = tree_unflatten(
        treedef, Y_merge_flat
    )
    Y_merge: Float[Array, "k arity output_dim"] = jnp.stack(
        [jnp.stack(y) for y in Y_merge_tree]
    )
    Y_merge: Float[Array, "k output_dim"] = Y_merge.sum(axis=1)

    output: Float[Array, "k+n_inputs-k_times_arity output_dim"] = jnp.concatenate(
        [Y_merge, Y_pass_through], axis=0
    )

    if normalize:
        Q = Q + X[:, None, :] - Y[..., None]
        return output, Q

    return output, None


@jaxtyped(typechecker=typechecker)
def circuit(
    X: Float[Array, "n_inputs input_dim"],
    Qs: PyTree[Float[Array, "?n_inputs ?output_dim ?input_dim"], "T"],
    W: Float[Array, "n_outputs"],
    layers: PyTree[Integer[Array, ""]],
    normalize: bool = False,
):
    new_Qs = []
    for Q, layer in zip(Qs, layers):
        X, new_Q = circuit_layer(X, Q, layer, normalize)
        if normalize:
            new_Qs.append(new_Q)

    new_Qs = tuple(new_Qs)

    assert X.shape[0] == 1
    X = W + X[0]

    logp = jax.nn.logsumexp(X)

    if normalize:
        new_W = W - jax.nn.logsumexp(W)
        return logp, new_Qs, new_W
    return logp

# @functools.lru_cache(maxsize=None)
def make_step(step_key):
    k, arity, mapping = step_key

    # @jax.jit
    def step(W, key, Q):
        key, subkey = jax.random.split(key)
        z = jax.nn.one_hot(
            jax.random.categorical(subkey, W, axis=-1),
            W.shape[-1])
        z_merge, z_pass = jnp.split(z, [k], axis=0)
        Y_merge = z_merge[mapping]  # could be a problem with ordering
        Y = jnp.concatenate([Y_merge, z_pass], axis=0)

        idx = jnp.argmax(Y, axis=1)
        newW = Q[jnp.arange(Y.shape[0]), idx, :]

        return newW, z, key

    return step

# @partial(jax.jit, static_argnames=("layers_treedef"))
def sample_circuit(layers_treedef, key, Qs, W, flat_layers):
    layers = tree_unflatten(layers_treedef, flat_layers)
    W = W[None, :]
    zs = []
    for Q, layer in zip(Qs[::-1], layers[::-1]):
        k = len(layer)
        merge_leaves, _ = tree_flatten(layer)
        arity = len(merge_leaves) // k
        merge_leaves = jnp.array(merge_leaves)
        groups = jnp.arange(merge_leaves.shape[0]) // arity
        mapping = groups[jnp.argsort(merge_leaves)]
        step_key = (k, arity, mapping)
        step = make_step(step_key)
        W, z, key = step(W, key, Q)
        zs.append(z)

    z = jax.nn.one_hot(
        jax.random.categorical(key, W, axis=-1),
        W.shape[-1])

    zs.append(z)
    return zs


@partial(jax.jit, static_argnames=("layers_treedef", "max_batch"))
def loss_fn_per_example(
    X: Float[Array, "batch_size n_inputs input_dim"],
    Qs: PyTree[Float[Array, "?n_inputs ?output_dim ?input_dim"], "T"],
    W: Float[Array, "n_outputs"],
    layers_flat: Integer[Array, "flat_layers"],
    layers_treedef: PyTreeDef,
    max_batch: int,
) -> Float[Array, "batch_size"]:
    layers = tree_unflatten(layers_treedef, layers_flat)
    pad = jnp.zeros_like(X[0:1])
    X_pad = jnp.concatenate([X, pad], axis=0)
    def compute_loss_per_batch(x):
        return circuit(x, Qs, W, layers)
    out = jax.vmap(compute_loss_per_batch)(X_pad)
    log_Z = out[-1]
    return -(out[:-1] - log_Z)


@partial(jax.jit, static_argnames=("layers_treedef"))
def loss_fn(
    X: Float[Array, "batch_size n_inputs input_dim"],
    Qs: PyTree[Float[Array, "?n_inputs ?output_dim ?input_dim"], "T"],
    W: Float[Array, "n_outputs"],
    layers_flat: Integer[Array, "flat_layers"],
    layers_treedef: PyTreeDef,
) -> Float[Array, ""]:
    return jnp.mean(loss_fn_per_example(X, Qs, W, layers_flat, layers_treedef))


def make_circuit_parameters(key, depth, n_clusters, n_categories, max_categories):
    Qs = []
    n_inputs = n_categories
    current_dim = max_categories
    for i in range(depth):
        key, subkey = jax.random.split(key)
        Q = jax.random.normal(subkey, (n_inputs, n_clusters[i], current_dim))
        Qs.append(Q)
        n_outputs = n_inputs // 2
        n_inputs -= n_outputs
        current_dim = n_clusters[i]

    key, subkey = jax.random.split(key)
    final_layer = jnp.arange(n_inputs + n_outputs)[None, :]
    W = jax.random.normal(subkey, (current_dim,))
    return Qs, W, final_layer
