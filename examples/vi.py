# %%
%load_ext autoreload
%autoreload 2

# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
# %%
import jax.numpy as jnp
import numpy as np
from moet.io import load_data
import jax


# %%
dataset_path = "data/lpm/PUMS"
train_data, test_data, col_names = load_data(dataset_path)
train_data.shape

# %%
# %%
from moet.trees import tree
import genjax


n_categories, max_categories = train_data.shape[1:]
n_categories = int(n_categories)
max_categories = int(max_categories)

# n_clusters = [512, 256, 128, 64, 32, 16]
# %%
import optax
from moet.model import loss_fn


# %%
from moet.model import make_circuit_parameters
from functools import partial
from moet.trees import get_layer

n_epochs = 1
batch_size = 1000
max_batches = train_data.shape[0] // batch_size
peak_value = 2.5e-2

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.lion(
        learning_rate=peak_value,
    ),
)


from moet.model import loss_fn_per_example

def combined_tree_loss_per_example(logps, Qs, W, flat_layers, batch, circuit_treedef):
    logps = jax.nn.log_softmax(logps)
    loss_per_example = jax.vmap(loss_fn_per_example, in_axes=(None, 0, 0, 0, None, None))(
        batch, Qs, W, flat_layers, circuit_treedef, batch_size 
    )
    return jax.nn.logsumexp(logps[:, None] + loss_per_example, axis=0)

def combined_tree_loss(logps, Qs, W, flat_layers, batch, circuit_treedef):
    loss = combined_tree_loss_per_example(logps, Qs, W, flat_layers, batch, circuit_treedef)
    return jnp.mean(loss)
 

@partial(jax.jit, static_argnames=("circuit_treedef",))
def gradient_step(opt_state, logps, Qs, W, flat_layers, batch, circuit_treedef):
    loss, grads = jax.value_and_grad(combined_tree_loss, argnums=(0, 1, 2))(logps, Qs, W, flat_layers, batch, circuit_treedef)
    updates, opt_state = optimizer.update(grads, opt_state, (logps, Qs, W))
    logps, Qs, W = optax.apply_updates((logps, Qs, W), updates)
    return (opt_state, logps, Qs, W), loss


def run_depth_experiment(
    key, thetas, train_data, batch_idxs, n_trees, circuit_depth, jitted_gradient_step, prev_params=None, n_epochs=2
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_trees)
    traces = jax.vmap(tree.simulate, in_axes=(0, (0, None)))(
        keys, (thetas, genjax.Pytree.const(circuit_depth))
    )

    def get_flat_layer(trace):
        return get_layer(trace, circuit_depth)[0]

    flat_layers = jax.vmap(get_flat_layer)(traces)

    Qs, W, _ = jax.vmap(make_circuit_parameters, in_axes=(0, None, None, None, None))(
        keys, circuit_depth, n_clusters, n_categories, max_categories
    )
    if prev_params is not None:
        
        def select_params(idx):
            return jax.lax.cond(
                idx == n_trees - 1, 
                lambda: jax.tree_util.tree_map(lambda x: x[idx], (Qs, W, flat_layers)), 
                lambda: jax.tree_util.tree_map(lambda x: x[idx], prev_params))

        trees = jnp.arange(n_trees)
        Qs, W, flat_layers = jax.vmap(select_params)(trees)

    key, subkey = jax.random.split(key)
    logps = jax.random.normal(key, (n_trees, ))

    opt_state = optimizer.init((logps, Qs, W))

    all_train_losses = np.zeros(max_batches * n_epochs)
    all_test_losses = np.zeros((max_batches * n_epochs) // 100)

    for epoch in range(n_epochs):
        for i, idxs in tqdm(enumerate(batch_idxs)):
            batch = train_data[idxs]
            (opt_state, logps, Qs, W), train_losses = jitted_gradient_step(opt_state, logps, Qs, W, flat_layers, batch)
            all_train_losses[i + epoch * max_batches] = train_losses

            if i % 100 == 0:
                # test_losses = combined_tree_loss(logps, Qs, W, flat_layers, test_data, jitted_gradient_step._fun.keywords['circuit_treedef'])
                test_losses = 0
                all_test_losses[i // 100 + epoch * (max_batches // 100)] = test_losses

    return (logps, Qs, W, flat_layers), all_train_losses, all_test_losses

# %%
# get a treedef

depth = 5
n_clusters = [128 // 2**i for i in range(depth)]
# n_clusters = [512, 256, 128, 64, 32]

key = jax.random.PRNGKey(1234)
key, subkey = jax.random.split(key)
uniform_theta = 1 - jnp.eye(n_categories)
uniform_theta = uniform_theta / jnp.sum(uniform_theta)
uniform_theta = jnp.log(uniform_theta)
trace = tree.simulate(key, (uniform_theta, genjax.Pytree.const(depth)))

_, treedef = get_layer(trace, depth)
Qs, W, _ = make_circuit_parameters(
    subkey, depth, n_clusters, n_categories, max_categories
)

# %%
jitted_gradient_step = jax.jit(partial(gradient_step, circuit_treedef=treedef))

# %%
from tqdm import tqdm
from functools import partial


def boost_iter(
    key,
    thetas,
    train_data,
    batch_idxs,
    n_trees,
    depth,
    jitted_gradient_step,
    prev_params,
):
    (logps, Qs, W, flat_layers), train_losses, test_losses = run_depth_experiment(
        key, thetas, train_data, batch_idxs, n_trees, depth, jitted_gradient_step, prev_params, n_epochs=n_epochs
    )

    partial_combined_tree_loss_per_example = partial(combined_tree_loss_per_example, circuit_treedef=jitted_gradient_step._fun.keywords['circuit_treedef'])
    partial_combined_tree_loss_per_example = jax.jit(jax.checkpoint(partial_combined_tree_loss_per_example))
    loss_per_example = np.zeros(train_data.shape[0])
    for i in range(0, train_data.shape[0], batch_size):
        loss_per_example[i:i+batch_size] = partial_combined_tree_loss_per_example(logps, Qs, W, flat_layers, train_data[i:i+batch_size])

    return logps, Qs, W, flat_layers, loss_per_example, train_losses, test_losses


# %%
@partial(jax.jit, static_argnames=("shape",))
def sample_categorical(key, logits, shape):
    return jax.random.categorical(key, logits, shape=shape)

# %%
# theta = get_mi_theta(train_data, n_mi_samples=10000)
theta = uniform_theta

n_trees = 1
n_boost_iter = 1
N = train_data.shape[0]
# l = 1
l = .9
key = jax.random.PRNGKey(1234)
batch_idxs = jnp.arange(batch_size * max_batches).reshape(max_batches, batch_size)
current_logps = -jnp.inf * jnp.ones(train_data.shape[0])
models = []

theta_list = []
prev_params = None
for i in range(n_boost_iter):
    key, subkey = jax.random.split(key)
    eps = 1e-5
    chow_liu_theta = theta / eps
    logz = jax.nn.logsumexp(chow_liu_theta.flatten(), axis=-1)
    chow_liu_theta = chow_liu_theta - logz
    theta_list.append(chow_liu_theta)

    logps, Qs, W, flat_layers, loss_per_example, train_losses, test_losses = boost_iter(
        subkey,
        jnp.array(theta_list),
        train_data,
        batch_idxs,
        n_trees,
        depth,
        jitted_gradient_step,
        prev_params,
    )
    O = (-l * jnp.exp(-loss_per_example) + 1./N) / (1 - l)
    # batch_idxs = sample_categorical(subkey, importance_weights, (max_batches, batch_size))
    # mi_idxs = sample_categorical(subkey, jnp.log(O), (10000,))
    # mi_data = train_data[mi_idxs]
    # theta = get_mi_theta(mi_data, n_mi_samples=10000)
    models.append({
        "logps": logps,
        "n_trees": n_trees,
        "theta_list": theta_list,
        "Qs": Qs,
        "W": W,
        "flat_layers": flat_layers,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "O": O,
        "l": l,
        # "mi_idxs": mi_idxs,
        "loss_per_example": loss_per_example,
    })
    prev_params = (Qs, W, flat_layers)
    n_trees += 1

# %%
jax.config.update("jax_debug_nans", True)

from moet.model import circuit
mock_data = jnp.zeros_like(train_data[0])
layers = jax.vmap(jax.tree_unflatten, in_axes=(None, 0))(treedef, flat_layers)

# %%
logZ, new_Qs, new_W = jax.vmap(circuit, in_axes=(None, 0, 0, 0, None))(mock_data, Qs, W, layers, True)
# logZ, new_Qs, new_W = jax.jit(jax.vmap(circuit, in_axes=(None, 0, 0, 0, None)))(mock_data, Qs, W, layers, True)

# %%
new_Qs = jax.tree_util.tree_map(lambda x: x[0], new_Qs)
new_W = new_W[0]
layers = jax.tree_util.tree_map(lambda x: x[0], layers)
flat_layers, treedef = jax.tree_util.tree_flatten(layers)
flat_layers

# %%
from moet.model import sample_circuit
key = jax.random.PRNGKey(0)
partial_sample_circuit = partial(sample_circuit, treedef)
subkeys = jax.random.split(key, 10000)
samples = jax.vmap(partial_sample_circuit, in_axes=(0, None, None, None))(subkeys, new_Qs, new_W, flat_layers)


# %%
from moet.utils import get_marginals
x_idx = 2
y_idx = 3
sample_logp_xy, sample_logp_x, sample_logp_y = jax.jit(get_marginals)(samples[:, x_idx], samples[:, y_idx])
sample_logp_x


# %%
data_logp_xy, data_logp_x, data_logp_y = jax.jit(get_marginals)(jnp.exp(train_data[:, x_idx]), jnp.exp(train_data[:, y_idx]))
data_logp_x

# %%
from moet.utils import make_1d_obs
all_x = jax.vmap(make_1d_obs, in_axes=(None, 0, None, None))(x_idx, jnp.arange(max_categories), n_categories, max_categories)
all_y = jax.vmap(make_1d_obs, in_axes=(None, 0, None, None))(y_idx, jnp.arange(max_categories), n_categories, max_categories)
all_x

# %%
partial_circuit = partial(circuit, layers=layers)
exact_logp_x = jax.vmap(jax.jit(partial_circuit), in_axes=(0, None, None))(all_x, new_Qs, new_W)
exact_logp_y = jax.vmap(jax.jit(partial_circuit), in_axes=(0, None, None))(all_y, new_Qs, new_W)

# %%
jnp.exp(exact_logp_x)

# %%
jnp.exp(sample_logp_x)

# %%
jnp.exp(data_logp_x)

# %%
jnp.exp(exact_logp_y)

# %%
jnp.exp(sample_logp_y)

# %%
jnp.exp(data_logp_y)

# %%
jnp.exp(exact_logp_y)