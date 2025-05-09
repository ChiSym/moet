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
from moet.trees import tree
import genjax

n_categories, max_categories = train_data.shape[1:]
n_categories = int(n_categories)
max_categories = int(max_categories)

# %%
from tqdm import tqdm
from moet.utils import weighted_marginals
p_x, p_xy = weighted_marginals(train_data)

# %%
import optax
from moet.model import make_circuit_parameters
from functools import partial
from moet.trees import get_layer

# n_epochs = 1
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
    loss_per_example = jax.vmap(loss_fn_per_example, in_axes=(None, 0, 0, 0, None))(
        batch, Qs, W, flat_layers, circuit_treedef, 
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
    key, thetas, train_data, importance_weights, batch_size, max_batches, n_trees, circuit_depth, jitted_gradient_step, prev_params=None, 
):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_trees)
    traces = jax.vmap(tree.simulate, in_axes=(0, (0, None)))(
        keys, (thetas, genjax.Pytree.const(circuit_depth))
    )

    def get_flat_layer(trace):
        return get_layer(trace, circuit_depth)[0]

    flat_layers = jax.vmap(get_flat_layer)(traces)

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, n_trees)
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
    logps = jax.random.normal(subkey, (n_trees, ))

    opt_state = optimizer.init((logps, Qs, W))

    all_train_losses = np.zeros(max_batches)
    all_test_losses = np.zeros((max_batches) // 100)

    for i in tqdm(range(max_batches)):
        key, subkey = jax.random.split(key)
        batch_idxs = sample_categorical(subkey, importance_weights, (batch_size, ))
        batch = train_data[batch_idxs]
        (opt_state, logps, Qs, W), train_losses = jitted_gradient_step(opt_state, logps, Qs, W, flat_layers, batch)
        all_train_losses[i] = train_losses

            # if i % 100 == 0:
            #     # test_losses = combined_tree_loss(logps, Qs, W, flat_layers, test_data, jitted_gradient_step._fun.keywords['circuit_treedef'])
            #     test_losses = 0
            #     all_test_losses[i // 100 + epoch * (max_batches // 100)] = test_losses

    return (logps, Qs, W, flat_layers), all_train_losses, all_test_losses

# %%
# get a treedef
import math

depth = math.ceil(math.log2(n_categories))
# depth = 1
n_clusters = [256 // 2**i for i in range(depth)]
# n_clusters = [128, 1]
key = jax.random.PRNGKey(1234)
key, subkey = jax.random.split(key)
uniform_theta = 1 - jnp.eye(n_categories)
uniform_theta = uniform_theta / jnp.sum(uniform_theta)
uniform_theta = jnp.log(uniform_theta)
trace = tree.simulate(key, (uniform_theta, genjax.Pytree.const(depth)))

flat_layers, treedef = get_layer(trace, depth)
layer = jax.tree_unflatten(treedef, flat_layers)
layer

# %%
jitted_gradient_step = jax.jit(partial(gradient_step, circuit_treedef=treedef))

# %%
from tqdm import tqdm
from functools import partial


def boost_iter(
    key,
    thetas,
    train_data,
    importance_weights,
    batch_size,
    max_batches,
    n_trees,
    depth,
    jitted_gradient_step,
    prev_params,
):
    (logps, Qs, W, flat_layers), train_losses, test_losses = run_depth_experiment(
        key, thetas, train_data, importance_weights, batch_size, max_batches, n_trees, depth, jitted_gradient_step, prev_params, 
    )

    partial_combined_tree_loss_per_example = partial(combined_tree_loss_per_example, circuit_treedef=jitted_gradient_step._fun.keywords['circuit_treedef'])
    partial_combined_tree_loss_per_example = jax.jit(jax.checkpoint(partial_combined_tree_loss_per_example))
    loss_per_example = np.zeros(train_data.shape[0])
    for i in range(0, train_data.shape[0], batch_size):
        loss_per_example[i:i+batch_size] = partial_combined_tree_loss_per_example(logps, Qs, W, flat_layers, train_data[i:i+batch_size])

    return logps, Qs, W, flat_layers, loss_per_example, train_losses, test_losses


# %%
@partial(jax.jit, static_argnames=("shape",))
def sample_categorical(key, probs, shape):
    return jax.random.choice(key, jnp.arange(len(probs)), p=probs, shape=shape)


# %%
from moet.utils import get_all_mis, get_theta, get_l

n_trees = 1
n_boost_iter = 2
N = train_data.shape[0]
beta = .05
key = jax.random.PRNGKey(1234)
models = []
theta_list = []
prev_params = None
prev_logps = 0
importance_weights = jnp.ones(N) / N
for i in range(n_boost_iter):
    key, subkey = jax.random.split(key)
    mi = get_all_mis(p_x, p_xy)
    theta = get_theta(mi)
    # theta_list.append(theta)

    logps, Qs, W, flat_layers, loss_per_example, train_losses, test_losses = boost_iter(
        subkey,
        # jnp.array(theta_list),
        jnp.array([theta]),
        train_data,
        importance_weights / jnp.sum(importance_weights),
        batch_size,
        max_batches,
        n_trees,
        depth,
        jitted_gradient_step,
        prev_params,
    )
    if i == 0:
        running_loss_per_example = loss_per_example
    else:
        running_loss_per_example = jax.nn.logsumexp(jnp.array([jnp.log(beta) + loss_per_example, jnp.log(1 - beta) + running_loss_per_example]), axis=0)
    p_g = jnp.exp(-running_loss_per_example)
    l = get_l(p_g, beta)
    importance_weights = jnp.maximum(l - (1 - beta) * N * p_g, 0) / beta
    p_x, p_xy = weighted_marginals(train_data, importance_weights)
    models.append({
        "logps": logps,
        "n_trees": n_trees,
        # "theta_list": theta_list,
        "Qs": Qs,
        "W": W,
        "flat_layers": flat_layers,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "l": l,
        "loss_per_example": loss_per_example,
        "importance_weights": importance_weights,
        "beta": beta,
        "mi": mi,
        "theta": theta,
        "p_x": p_x,
        "p_xy": p_xy,
    })
    # prev_params = (Qs, W, flat_layers)
    # n_trees += 1
    print(jnp.mean(running_loss_per_example))

# %%
loss_per_example[6035]

# %%
beta = .5
p_g = jnp.exp(-loss_per_example)
l = get_l(p_g, beta)
weights = jnp.maximum(l - (1-beta) * N * p_g, 0) / beta
weights

# %%


# %%
jnp.sum(weights)

# %%
jnp.sum(weights > 0)

# %%

import matplotlib.pyplot as plt

# Sort and exponentiate the running loss per example
# boosting_weights = 1/jnp.exp(jnp.sort(-running_loss_per_example))
# boosting_weights = boosting_weights / jnp.sum(boosting_weights)
# Plot the importance weights
plt.figure(figsize=(10, 6))
plt.plot(jnp.sort(weights), label='Boosting Weights')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Boosting Weights Distribution')
plt.legend()
plt.grid(True)
plt.show()

# %%
jnp.sum(weights)

# %%
import matplotlib.pyplot as plt

# Sort and exponentiate the running loss per example
boosting_weights = jnp.sort(1/jnp.exp(-running_loss_per_example))
boosting_weights = boosting_weights / jnp.sum(boosting_weights)
# Plot the importance weights
plt.figure(figsize=(10, 6))
plt.plot(jnp.sort(boosting_weights), label='Boosting Weights')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Boosting Weights Distribution')
plt.legend()
plt.grid(True)
plt.show()

# %%
import matplotlib.pyplot as plt

# Sort and exponentiate the running loss per example
boosting_weights = jnp.sort(1/jnp.exp(-running_loss_per_example))
boosting_weights = boosting_weights / jnp.sum(boosting_weights)
# Plot the importance weights
plt.figure(figsize=(10, 6))
plt.plot(jnp.sort(boosting_weights), label='Boosting Weights')
plt.xlabel('Index')
plt.ylabel('Weight')
plt.title('Boosting Weights Distribution')
plt.legend()
plt.grid(True)
plt.show()



# %%
circuit_depth = depth
key, subkey = jax.random.split(key)
keys = jax.random.split(subkey, n_trees)
traces = jax.vmap(tree.simulate, in_axes=(0, (0, None)))(
    keys, (jnp.array([theta]), genjax.Pytree.const(circuit_depth))
)

def get_flat_layer(trace):
    return get_layer(trace, circuit_depth)[0]

flat_layers = jax.vmap(get_flat_layer)(traces)
flat_layers

# %%
p_x

# %%
data_p_x, data_p_xy = weighted_marginals(train_data)
data_p_x


# %%
p_x.sum(axis=-1)

# %%
get_all_mis(p_x, p_xy)

# %%
models[0]["mi"]

# %%
models[1]["mi"]

# %%
models[0]["train_losses"][-100:]

# %%
models[1]["train_losses"][-100:]

# %%


# %%
models[0]["importance_weights"]

# %%
running_loss_per_example = jax.nn.logsumexp(jnp.array([jnp.log(beta) + loss_per_example, jnp.log(1 - beta) + running_loss_per_example]), axis=0)

# %%
jnp.mean(running_loss_per_example)

# %%
# normalized_probs = importance_weights / N
batch_idxs = sample_categorical(importance_weights / jnp.sum(importance_weights)).reshape(max_batches, batch_size)

# %%
jnp.sum(importance_weights / N)

# %%




# %%
p_x, p_xy = weighted_marginals(train_data, importance_weights)

# %%
idx_x = 0
idx_y = 1

# %%
- jnp.log(p_x[idx_x])[:, None] - jnp.log(p_x[idx_y])[None, :]

# %%
- jnp.sum(p_xy * (jnp.log(p_xy) - jnp.log(p_x[:, None]) - jnp.log(p_y[None, :])))

# %%
p_xy

# %%
mi = get_all_mis(p_x, p_xy)
mi

# %%
theta = get_theta(mi)
theta

# %%
print(jnp.mean(loss_per_example))

# %%
data_px, data_pxy = weighted_marginals(train_data)
data_px

# %%
logp, new_Qs, new_W = circuit(jnp.array(train_data[-100]), jax.tree_util.tree_map(lambda x: x[0], models[0]["Qs"]), models[0]["W"][0], layer)


# %%
from moet.model import circuit
layer = jax.tree_unflatten(treedef, flat_layers[0])
logp = circuit(train_data[0], jax.tree_util.tree_map(lambda x: x[0], models[0]["Qs"]), models[0]["W"][0], layer, True)


# %%
logZ, new_Qs, new_W = circuit(jnp.zeros_like(train_data[0]), jax.tree_util.tree_map(lambda x: x[0], Qs), W[0], layer, True)
logZ

# %%
layers


# %%
models[0]["Qs"][0]


# %%
from moet.utils import get_l_candidates
# compute l_candidates
logps = -models[0]["loss_per_example"]
logps -= jax.nn.logsumexp(logps)
ps_sorted = jnp.sort(jnp.exp(logps))
l_candidates = get_l_candidates(ps_sorted, models[0]["beta"])
l_candidates
# %%
normalized_loss_per_example = logps
beta = .25
l = get_l(normalized_loss_per_example, beta)
l

# %%
l

# %%
# importance_weights = jnp.maximum((l / N) - (1 - beta) * jnp.exp(normalized_loss_per_example), 0) / beta
# importance_weights = jnp.minimum((1 / N) - l * (1 - beta) * jnp.exp(normalized_loss_per_example), 0) / beta
sorted_importance_weights = jnp.sort(models[0]["importance_weights"])
sorted_importance_weights

# %%
# plot sorted importance weights
# skip = 10000
# sorted_normalized_loss_per_example = sorted_normalized_loss_per_example - jax.nn.logsumexp(sorted_normalized_loss_per_example)
import matplotlib.pyplot as plt

importance_weights = models[0]["importance_weights"]
delta = models[0]["loss_per_example"] - models[1]["loss_per_example"]
sort_idxs = jnp.argsort(delta)
importance_weights = importance_weights[sort_idxs]
delta = delta[sort_idxs]
plt.plot(delta)
plt.plot(importance_weights)
plt.show()


# %%
jnp.sum(sorted_importance_weights)

# %%
train_losses

# %%
def compute_weight_variance(beta):
    l = get_l(normalized_loss_per_example, beta)
    importance_weights = jnp.maximum((l / N) - (1 - beta) * jnp.exp(normalized_loss_per_example), 0) / beta
    mu = jnp.mean(importance_weights)
    var = jnp.sum((importance_weights - mu) ** 2) / jnp.sum(importance_weights > 0)
    return var


betas = jnp.linspace(0, 1, 100)
weight_variances = jax.jit(jax.vmap(compute_weight_variance))(betas)
weight_variances

# %%
sorted_normalized_loss_per_example

# %%
betas[-1]
# %%
plt.plot(betas, weight_variances)
plt.show()

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