# %%
import jax
from jax import random as jrand
import jax.numpy as jnp
import jax.tree_util as jtu
from moet.model import make_circuit_parameters
from moet.trees import get_layer
from moet.trees import tree
from moet.model import circuit
from moet.model import sample_circuit
from functools import partial
import genjax
key = jrand.key(1)

# %%
jax.default_backend()

# %%
n_clusters = [5, 3]
circuit_depth = 2
n_categories = 4
max_categories= 3

# %%
uniform_theta = 1 - jnp.eye(n_categories)
uniform_theta = uniform_theta / jnp.sum(uniform_theta)
uniform_theta = jnp.log(uniform_theta)
trace = tree.simulate(key, (uniform_theta, genjax.Pytree.const(circuit_depth)))

# %%
layers = (
    # ((3, 2), (1, 0), (4, 6), (5, 7)),
    ((1, 2), (3, 0)),
    ((1, 0),),
)
layers = jtu.tree_map(jnp.array, layers)
flat_layers, treedef = jtu.tree_flatten(layers)

# flat_layers, treedef = get_layer(trace, circuit_depth)
# layers = jtu.tree_unflatten(treedef, flat_layers)

# %%
new_Qs, new_W, _ = make_circuit_parameters(
    key, circuit_depth, n_clusters, n_categories, max_categories
)

# %%
mock_data = jnp.zeros((n_categories, max_categories))
logZ, new_Qs, new_W = circuit(mock_data, jax.tree_util.tree_map(jnp.array, new_Qs), jnp.array(new_W), layers, True)
# %%
from tqdm import tqdm
n_samples = 1000
keys = jax.random.split(key, n_samples)
sum_choices = []
for k in tqdm(keys):
    zs = sample_circuit(treedef, k, jax.tree_util.tree_map(jnp.array, new_Qs), jnp.array(new_W), flat_layers)
    sum_choices.append(zs[-1])  # or whichever layer you want
sum_choices = jnp.array(sum_choices)

# %%
sum_choices.mean(axis=0)
# %%
import numpy as np
all_marginals = np.zeros((n_categories, max_categories))
for category in range(n_categories):
    for idx in range(max_categories):
        all_obs = jnp.zeros((n_categories, max_categories))
        for i in range(max_categories):
            if i != idx:
                all_obs = all_obs.at[category, i].set(-jnp.inf)
        logp = circuit(all_obs, jax.tree_util.tree_map(jnp.array, new_Qs), jnp.array(new_W), layers)
        all_marginals[category, idx] = np.exp(logp)
all_marginals
# %%
all_obs
# %%
layers
# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
# %%
unflattened_layers = jtu.tree_unflatten(treedef, flat_layers)
unflattened_layers
# %%
layers
# %%
flat_layers
# %%

def make_mapping(layer):
    merge_leaves = jnp.array([int(x) for group in layer for x in group])
    groups = jnp.array([[int(x) for x in group] for group in layer])
    group_sizes = jnp.array([len(group) for group in layer])

    def reverse_idx_in_group(leaf):
        # For each group, get the index if present, else -1
        eq = groups == leaf  # shape (n_groups, arity)
        idxs = jnp.arange(groups.shape[1])
        idx_in_group = jnp.where(eq, idxs, -1)
        idx = jnp.max(idx_in_group, axis=1)
        group_idx = jnp.argmax(idx >= 0)
        # Reverse index: (group_size - 1) - idx
        rev_idx = (group_sizes[group_idx] - 1) - idx[group_idx]
        return rev_idx

    mapping = jax.vmap(reverse_idx_in_group)(merge_leaves)
    return mapping

layer = ((1, 2), (3, 0))
print(make_mapping(layer))  # [1, 0, 0, 1]

# %%
arity = 2
flat_layer = jnp.array([1, 2, 3, 0])
groups = jnp.arange(4) // arity
groups
# %%
groups[flat_layer]

# %%
groups[jnp.argsort(flat_layer)]

# %%
