from genjax import adev
import jax
from jaxtyping import Array
import jax.numpy as jnp
from jaxtyping import Int, Float
from functools import partial
from moet.utils import cat_reinforce, categorical2d
from genjax import gen, categorical


@gen
def tree_node(θ: Float[Array, "n n"], _) -> tuple[Float[Array, "n n"], Int[Array, "2"]]:
    merge = categorical2d(θ) @ f"merge"
    θ = renormalize(θ, merge)
    return θ, merge


@gen
def tree_layer(
    θ: Float[Array, "n n"], i: int = 0
) -> tuple[Float[Array, "n%2+n//2 n%2+n//2"], Int[Array, "n%2+n//2 2"]]:
    n = jnp.shape(θ)[0]
    tree_node_scan = tree_node.scan(n=n // 2)
    _, merge_pairs = tree_node_scan(θ, None) @ f"{i}"
    merge_pairs = jnp.sort(merge_pairs)
    no_merge_idxs = jnp.argwhere(~jnp.isin(jnp.arange(n), merge_pairs), size=n % 2)[
        :, 0
    ]
    no_merge_pairs = jnp.stack((no_merge_idxs, no_merge_idxs), axis=-1)
    pairs = jnp.concatenate([merge_pairs, no_merge_pairs])
    mi_pairs_array: Float[Array, "n%2+n//2 n%2+n//2 4"] = jax.vmap(
        jax.vmap(mi_pairs, in_axes=(None, 0, None)), in_axes=(None, None, 0)
    )(θ, pairs, pairs)
    θ = jnp.max(mi_pairs_array, axis=-1)
    θ = jnp.where(jnp.eye(len(θ)), -jnp.inf, θ)
    logZ = jax.nn.logsumexp(θ)
    θ = θ - logZ
    return θ, pairs


@gen
def tree(θ: Float[Array, "n n"], depth: int):
    depth = depth.val
    layer_list = []
    for i in range(depth):
        θ, pairs = tree_layer.inline(θ, i)
        layer_list.append(pairs)
    return layer_list


def mi_pairs(θ: Float[Array, "n n"], idxs1: Int[Array, "2"], idxs2: Int[Array, "2"]):
    # TODO make this neater
    θ1 = θ.at[idxs1[0], idxs2[0]].get()
    θ2 = θ.at[idxs1[0], idxs2[1]].get()
    θ3 = θ.at[idxs1[1], idxs2[0]].get()
    θ4 = θ.at[idxs1[1], idxs2[1]].get()
    return jnp.array([θ1, θ2, θ3, θ4])


def renormalize(θ: Float[Array, "n n"], idxs: Int[Array, "2"]):
    n = jnp.shape(θ)[0]
    all_idxs = jnp.arange(n)
    mask1d = ~jnp.isin(all_idxs, idxs)
    mask = mask1d[:, None] & mask1d[None, :]
    θ = jnp.where(mask, θ, -jnp.inf)
    logZ = jax.nn.logsumexp(θ)
    θ_normalized = θ - logZ
    return θ_normalized


@gen
def pseudomarginal(
    layer_list: list[Int[Array, "k"]], θ_list: list[Float[Array, "n n"]]
):
    return jnp.sum(
        [
            pseudomarginal_layer(θ, layer) @ f"{i}"
            for i, (θ, layer) in enumerate(zip(θ_list, layer_list))
        ]
    )


def pseudomarginal_layer(θ: Float[Array, "n n"], merges: Int[Array, "n//2"]):
    # q(r|x) where r is the merge order and x is the set of merged nodes
    k = jnp.shape(merges)[0]
    logprobs = θ.take(merges)
    cumsum = jax.vmap(lambda idx: jax.nn.logsumexp(logprobs[idx:]))(jnp.arange(k))
    return jnp.sum(logprobs - cumsum)
