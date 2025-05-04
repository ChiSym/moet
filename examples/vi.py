# %%
# %load_ext autoreload
# %autoreload 2

# %%
# import treescope

# treescope.basic_interactive_setup(autovisualize_arrays=True)
# from moet.model import circuit_layer
# from moet.io import (
#     load_huggingface,
#     discretize_dataframe,
#     dummies_to_padded_array,
#     to_dummies,
# )

# %%
import jax.numpy as jnp
import numpy as np
from moet.io import load_data
import jax

# %%
jax.__version__

# %%
dataset_path = "data/CTGAN/covertype"
train_data, test_data, col_names = load_data(dataset_path)
train_data.shape

# %%
from moet.utils import get_mi_theta

theta = get_mi_theta(train_data)
import ipdb

ipdb.set_trace()

# %%
from moet.trees import tree
import genjax


n_categories, max_categories = train_data.shape[1:]
n_categories = int(n_categories)
max_categories = int(max_categories)

n_clusters = [512, 256, 128, 64, 32, 16]
# %%
import optax
from moet.model import loss_fn


# %%
from moet.model import make_circuit_parameters
from functools import partial
from moet.trees import get_layer

learning_rate = 1e-1
optimizer = optax.adam(learning_rate)


@partial(jax.jit, static_argnames=("circuit_treedef",))
def gradient_step(opt_state, Qs, W, flat_layers, batch, circuit_treedef):
    loss, grads = jax.value_and_grad(loss_fn, argnums=(1, 2))(
        batch, Qs, W, flat_layers, circuit_treedef
    )
    updates, opt_state = optimizer.update(grads, opt_state, (Qs, W))
    Qs, W = optax.apply_updates((Qs, W), updates)
    return (opt_state, Qs, W), loss


def run_depth_experiment(
    key, theta, train_data, batch_idxs, n_trees, circuit_depth, jitted_gradient_step
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_trees)
    traces = jax.vmap(tree.simulate, in_axes=(0, None))(
        keys, (theta, genjax.Pytree.const(circuit_depth))
    )

    def get_flat_layer(trace):
        return get_layer(trace, circuit_depth)[0]

    flat_layers = jax.vmap(get_flat_layer)(traces)

    Qs, W, _ = jax.vmap(make_circuit_parameters, in_axes=(0, None, None, None, None))(
        keys, circuit_depth, n_clusters, n_categories, max_categories
    )

    opt_state = jax.vmap(optimizer.init)((Qs, W))

    all_train_losses = np.zeros((max_batches, n_trees))
    for i, idxs in enumerate(batch_idxs):
        batch = train_data[idxs]
        (opt_state, Qs, W), train_losses = jax.vmap(
            jitted_gradient_step, in_axes=(0, 0, 0, 0, None)
        )(opt_state, Qs, W, flat_layers, batch)
        all_train_losses[i] = train_losses

    return (Qs, W, flat_layers), all_train_losses


# %%
# get a treedef
depth = 6

key = jax.random.PRNGKey(0)
subkey = jax.random.PRNGKey(1)
trace = tree.simulate(key, (theta, genjax.Pytree.const(depth)))

_, treedef = get_layer(trace, depth)
Qs, W, _ = make_circuit_parameters(
    subkey, depth, n_clusters, n_categories, max_categories
)

# %%
jitted_gradient_step = jax.jit(partial(gradient_step, circuit_treedef=treedef))

# %%
from tqdm import tqdm
from functools import partial
import jaxopt

results_dict = {}


def boost_iter(
    key,
    theta,
    train_data,
    batch_idxs,
    n_trees,
    depth,
    jitted_gradient_step,
    current_logps,
):
    (Qs, W, flat_layers), train_losses = run_depth_experiment(
        key, theta, train_data, batch_idxs, n_trees, depth, jitted_gradient_step
    )
    best_idx = jnp.argmin(train_losses[-1])
    best_Qs, best_W, best_flat_layers = jax.tree.map(
        lambda x: x[best_idx], (Qs, W, flat_layers)
    )

    layers = jax.tree.unflatten(treedef, best_flat_layers)
    partial_circuit = jax.jit(
        jax.checkpoint(partial(circuit, Qs=best_Qs, W=best_W, layers=layers))
    )
    out = jax.lax.map(partial_circuit, train_data, batch_size=1000)
    pad = jnp.zeros_like(train_data[0])
    logZ = circuit(pad, best_Qs, best_W, layers)
    logps = out - logZ

    def fn_to_minimize(unnormalized_alpha):
        alpha = jax.nn.sigmoid(unnormalized_alpha)
        new_logps = jax.nn.logsumexp(
            jnp.stack([current_logps, logps]),
            b=jnp.array([1 - alpha, alpha])[:, None],
            axis=0,
        )
        return -jnp.mean(new_logps)

    solver = jaxopt.LBFGS(fun=fn_to_minimize, maxiter=100)
    res = solver.run(jnp.array([0.0]))
    params, state = res
    import ipdb

    ipdb.set_trace()

    new_logps = jax.nn.logsumexp(
        jnp.stack([current_logps, logps]),
        b=jnp.array([1 - alpha, alpha])[:, None],
        axis=0,
    )

    return best_Qs, best_W, best_flat_layers, alpha, new_logps


# %%
@jax.jit
def sample_categorical(key, logits):
    return jax.random.categorical(key, logits, shape=(max_batches, batch_size))


n_trees = 5
n_boost_iter = 2
batch_size = 200
max_batches = 100
batch_idxs = jnp.arange(batch_size * max_batches).reshape(max_batches, batch_size)
current_logps = -jnp.inf * jnp.ones(train_data.shape[0])
current_logp = -jnp.inf
models = []
for i in range(n_boost_iter):
    key, subkey = jax.random.split(key)
    Qs, W, flat_layers, alpha, current_logp, current_logps = boost_iter(
        subkey,
        theta,
        train_data,
        batch_idxs,
        n_trees,
        depth,
        jitted_gradient_step,
        current_logps,
    )
    key, subkey = jax.random.split(key)
    batch_idxs = sample_categorical(subkey, -current_logps)
    models.append((Qs, W, flat_layers, alpha, current_logps))


# %%
jnp.mean(models[0][-1])

# %%
jnp.mean(models[1][-1])

# %%
models[0][3]

# %%
models[1][3]
