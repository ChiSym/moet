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
n_clusters = [512, 256, 128, 64, 32]
circuit_depth = len(n_clusters)
n_categories = 26
max_categories=128

# %%
uniform_theta = 1 - jnp.eye(n_categories)
uniform_theta = uniform_theta / jnp.sum(uniform_theta)
uniform_theta = jnp.log(uniform_theta)
trace = tree.simulate(key, (uniform_theta, genjax.Pytree.const(circuit_depth)))

flat_layers, treedef = get_layer(trace, circuit_depth)
layers = jtu.tree_unflatten(treedef, flat_layers)

# %%
Qs, W, _ = make_circuit_parameters(
    key, circuit_depth, n_clusters, n_categories, max_categories
)
# %%
mock_data = jnp.zeros((n_categories, max_categories))
logZ, new_Qs, new_W = circuit(mock_data, Qs, W, layers, True)
# %%
subkeys = jax.random.split(key, 10000)
partial_sample_circuit = partial(sample_circuit, treedef)
sample_jit = jax.jit(jax.vmap(partial_sample_circuit,
                     in_axes=(0, None, None, None)
                     ))
samples = sample_jit(subkeys, new_Qs, new_W, flat_layers)
samples.shape
# %%
jax.default_backend()
# %%
