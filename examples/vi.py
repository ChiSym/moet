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
n_clusters = [512, 256, 128, 64, 32]
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

# schedule = optax.linear_onecycle_schedule(
#     transition_steps=n_epochs * max_batches,       # total steps
#     peak_value=peak_value,             # your max LR
#     pct_start=0.1,               # 10% of 2k steps spent ramping up
#     div_factor=100,              # start at LR_max/100
#     final_div_factor=1e4,        # end at LR_max/1e4
# )
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.lion(
        learning_rate=peak_value,
    ),
)


# # 1. Hyper‑parameters for the LR Finder
# lr_start, lr_end = 1e-5, 10.0

# # 2. Build a "log‑space" linear scheduler, then exponentiate to get an exponential ramp
# log_lr_schedule = optax.linear_schedule(
#     init_value=jnp.log(lr_start),
#     end_value=jnp.log(lr_end),
#     transition_steps=max_batches
# )
# exp_lr_schedule = lambda step: jnp.exp(log_lr_schedule(step))

# # 3. Plug that schedule into your optimizer
# optimizer = optax.chain(
#     optax.clip_by_global_norm(1.0),
#     optax.lion(learning_rate=exp_lr_schedule)
# )

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
    logps = jax.random.normal(key, (n_trees))

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

key = jax.random.PRNGKey(0)
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
def compute_ess(logw):
    L1 = jax.nn.logsumexp(logw)
    L2 = jax.nn.logsumexp(2*logw)
    log_ESS = 2*L1 - L2
    ESS = jnp.exp(log_ESS)
    return ESS

def clip_k(logw, k):
    sorted_logw = jnp.sort(logw)
    k_plus_1_value = sorted_logw[-k-1]
    
    clipped_logw = jnp.where(logw > k_plus_1_value, k_plus_1_value, logw)
    
    return clipped_logw


# %%
@partial(jax.jit, static_argnames=("shape",))
def sample_categorical(key, logits, shape):
    return jax.random.categorical(key, logits, shape=shape)

from moet.utils import get_mi_theta

theta = get_mi_theta(train_data, n_mi_samples=10000)

n_trees = 1
n_boost_iter = 5
target_ess_ratio = 1
key = jax.random.PRNGKey(0)
# batch_idxs = jax.random.permutation(key, jnp.arange(batch_size * max_batches)).reshape(max_batches, batch_size)
batch_idxs = jnp.arange(batch_size * max_batches).reshape(max_batches, batch_size)
current_logps = -jnp.inf * jnp.ones(train_data.shape[0])
models = []

key = jax.random.PRNGKey(0)
theta_list = []
prev_params = None
for i in range(n_boost_iter):
    target_ess_ratio *= 2/3
    eps = 1e-5
    chow_liu_theta = theta / eps
    logz = jax.nn.logsumexp(chow_liu_theta.flatten(), axis=-1)
    chow_liu_theta = chow_liu_theta - logz
    theta_list.append(chow_liu_theta)
    key, subkey = jax.random.split(key)

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
    key, subkey = jax.random.split(key)
    importance_weights = jax.nn.log_softmax(-loss_per_example)
    ess = compute_ess(importance_weights)
    if ess < target_ess_ratio * train_data.shape[0]:
        importance_weights = clip_k(importance_weights, int(target_ess_ratio * train_data.shape[0]))
        importance_weights = jax.nn.log_softmax(importance_weights)
    # batch_idxs = sample_categorical(subkey, importance_weights, (max_batches, batch_size))
    mi_idxs = sample_categorical(subkey, importance_weights, (10000,))
    mi_data = train_data[mi_idxs]
    theta = get_mi_theta(mi_data, n_mi_samples=10000)
    models.append({
        "logps": logps,
        "n_trees": n_trees,
        "theta_list": theta_list,
        "Qs": Qs,
        "W": W,
        "flat_layers": flat_layers,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "importance_weights": importance_weights,
        "mi_idxs": mi_idxs,
        "loss_per_example": loss_per_example,
    })
    prev_params = (Qs, W, flat_layers)
    n_trees += 1

# %%
theta_list[1]

# %%
jax.nn.softmax(importance_weights).sort()

# %%
sorted_logw = jnp.sort(importance_weights)
k = int(.05 * train_data.shape[0])
k_plus_1_value = sorted_logw[-k-1]
k_plus_1_value


# %%
# Plot the softmax of current_logps for model[0]
import matplotlib.pyplot as plt
import jax.nn

# Get the current_logps from the first model
# current_logps = models[0]["current_logps"]

# Apply softmax to convert log probabilities to probabilities
probabilities = jnp.exp(models[0]["importance_weights"]).sort()

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the probabilities
plt.plot(probabilities)
plt.xlabel('Data Point Index')
plt.ylabel('Probability')
plt.title('Softmax of -current_logps for Model 0')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# %%
import matplotlib.pyplot as plt

# Extract train_losses from each model
all_train_losses = [model["train_losses"] for model in models]
all_test_losses = [model["test_losses"] for model in models]

# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot each model's training losses
for i, losses in enumerate(all_train_losses):
    plt.plot(losses, label=f'Model {i+1}')

for i, losses in enumerate(all_test_losses):
    plt.plot(losses.repeat(100, axis=0), label=f'Test {i+1}', linestyle='--')



# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# %%
# Create a figure and axis
plt.figure(figsize=(10, 6))
for i, losses in enumerate(models[0][5].T):
    plt.plot(losses[2:], label=f'Test {i+1}')

# Add labels and title
plt.xlabel('Iteration')
plt.ylabel('Training Loss')
plt.title('Training Loss Over Iterations')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()


# %%
jnp.mean(models[0]["loss_per_example"])

# %%
jnp.mean(models[1]["loss_per_example"])

# %%
jnp.mean(models[2]["loss_per_example"])

# %%
jnp.mean(models[3]["loss_per_example"])


# %%
models[0]["train_losses"][-1]

# %%
models[1]["train_losses"][-1]

# %%
models[0][-2]
# %%

mi_idxs = sample_categorical(subkey, -models[0][-2], (10000,))
mi_idxs
# %%
jnp.unique(mi_idxs)
# %%
# use python counter to count the number of unique values in mi_idxs
from collections import Counter
counter = Counter(np.array(mi_idxs))
# %%
max(counter.keys(), key=counter.get)
# %%
counter.most_common(10)


# %%
models[0][-2][33007]

# %%
probs = jax.nn.softmax(-models[0][-2])
probs
# %%
probs[33007]
# %%
1/jnp.sum(probs**2)
# %%
data = az.load_arviz_data("non_centered_eight")
log_likelihood = data.log_likelihood["obs"].stack(
    __sample__=["chain", "draw"]
)
log_likelihood


# %%

# %%
1/jnp.sum(jnp.exp(new_logprobs[0]) ** 2)


# %%
