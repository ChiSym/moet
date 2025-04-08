# %%
%load_ext autoreload
%autoreload 2

# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
from moet.model import circuit_layer
from moet.io import (
    load_huggingface,
    discretize_dataframe,
    dummies_to_padded_array,
    to_dummies,
)
from jaxtyping import Float, Array, Bool
# %%
import polars as pl
import jax.numpy as jnp
import numpy as np

dataset_path = "data/lpm/PUMS"
train_df, test_df = load_huggingface(dataset_path)
df = pl.concat((train_df, test_df), how="vertical")
schema, discretized_df, categorical_idxs = discretize_dataframe(df)
dummies_df = to_dummies(discretized_df)
bool_data, col_names = dummies_to_padded_array(dummies_df, categorical_idxs)

bool_data = np.where(np.any(bool_data, axis=-1)[..., None], bool_data, True)
data: Float[Array, "batch_size n_inputs input_dim"] = jnp.where(bool_data, 0, -jnp.inf)

# %%
train_data = data[: len(train_df)]
test_data = data[len(train_df) :]


# %%
from moet.utils import mi
import jax

n_mi_samples = 10000
jitted_mi = jax.jit(jax.vmap(jax.vmap(mi, in_axes=(1, None)), in_axes=(None, 1)))
mi_estimates = jitted_mi(bool_data[:n_mi_samples, :].astype(float), bool_data[:n_mi_samples, :].astype(float))
mi_estimates

# %%
eps = 1e-10
theta = jnp.maximum(mi_estimates, eps)
theta = jnp.where(jnp.eye(len(mi_estimates)), 0, theta)
theta

# %%
theta /= jnp.sum(theta)
theta_flat = theta.flatten()

# %%
from moet.trees import tree

# %%
key = jax.random.PRNGKey(0)
import genjax
depth = jnp.log2(bool_data.shape[1]).astype(int) + 1
trace = tree.simulate(key, (jnp.log(theta), genjax.Pytree.const(depth)))
trace

# %%
layers = [trace.subtraces[f"{i}"].get_choices()['merge'] 
          for i in range(depth)]
layers
# %%

n_categories, max_categories = train_data.shape[1:]

def make_circuit_parameters(key, depth, n_categories, max_categories):
    n_clusters = [512, 256, 128, 64, 32]
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

# %%
n_categories = bool_data.shape[1]
max_categories = bool_data.shape[-1]
from moet.model import circuit
from moet.utils import to_tuple
import optax
from moet.model import loss_fn


# %%
def run_depth_experiment(key, circuit_depth, batch_size=512, test_loss_size=10000, max_batches=1000, learning_rate=1e-1):
    Qs, W, final_layer = make_circuit_parameters(key, circuit_depth, n_categories, max_categories)

    new_layers = layers[:circuit_depth - 1] + [final_layer]
    new_layers = tuple(np.array(l) for l in new_layers)

    int_layers = to_tuple(new_layers)


    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)

    n_batches = train_data.shape[0] // batch_size

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init((Qs, W))

    jitted_loss_fn = jax.jit(loss_fn, static_argnames=("layers",))

    train_losses = np.zeros(max_batches)
    test_losses = np.zeros((max_batches + 1) // 100)

    for i in range(max_batches):
        batch = train_data[i * batch_size:(i + 1) * batch_size]
        loss, grads = jax.value_and_grad(jitted_loss_fn, argnums=(1, 2))(batch, Qs, W, int_layers)
        train_losses[i] = loss
        if (i + 1) % 100 == 0:
            test_losses[-1 + (i + 1) // 100] = loss_fn(test_data[:test_loss_size], Qs, W, int_layers)
        # print(f"Train loss: {train_losses[i]}, test loss: {test_losses[i]}")
        updates, opt_state = optimizer.update(grads, opt_state, (Qs, W))
        Qs, W = optax.apply_updates((Qs, W), updates)

    return train_losses, test_losses

# %%
from tqdm import tqdm
results_dict = {}
key = jax.random.PRNGKey(0)
n_replicates = 5
for i in tqdm(range(n_replicates)):
    replicates_dict = {}
    for depth in tqdm(range(1, 6)):
        key, subkey = jax.random.split(key)
        train_losses, test_losses = run_depth_experiment(subkey, depth)
        replicates_dict[depth] = {
            "train_losses": train_losses,
            "test_losses": test_losses,
        }
    results_dict[i] = replicates_dict

# %%
import polars as pl

# Create a list to store all the data
data = []

# Iterate through the results dictionary
for replicate, results in results_dict.items():
    for depth, losses in results.items():
        test_losses = losses["test_losses"]
        for i, test_loss in enumerate(test_losses):
            data.append({
                "depth": str(depth),
                "iter": i,
                "replicate": replicate,
                "test loss": float(test_loss)
            })

# Create the polars DataFrame
results_df = pl.DataFrame(data)
results_df


# %%
avg_df = results_df.group_by(["depth", "iter"]).agg(pl.col("test loss").mean())
avg_df

# %%
avg_df = avg_df.with_columns(
    test_probability = (-pl.col("test loss")).exp(),
)

# %%
avg_df = avg_df.with_columns(
    iter = 100 * (pl.col("iter") + 1)
)

# %%
results_df = results_df.with_columns(
    test_probability = (-pl.col("test loss")).exp(),
)

# %%
len(train_df)


# %%
# Import plotnine for creating the line plot
from plotnine import ggplot, aes, geom_line, labs, theme_bw, scale_color_manual, geom_smooth, geom_point, element_text, theme, scale_y_continuous

# Create the line plot with iter as x-axis, train loss as y-axis, and depth as color
plot = (
    ggplot(avg_df, aes(x='iter', y='test_probability', color='depth'))
    + geom_line(size=1.5)
    + labs(
        title='Average test set $p(x)$ by model depth',
        x='Iteration',
        y='$p(x)$',
        color='Depth'
    )
    + theme_bw()
    + scale_y_continuous(labels=lambda x: [f"{val:.1e}" for val in x])  # Format y-axis labels in scientific notation
    + scale_color_manual(values=['#000004', '#3b0f70', '#8c2981', '#de4968', '#fe9f6d'])  # Magma color palette
)

plot
# %%
# Save the plot to a PNG file with high resolution
plot.save("figures/test_probability_by_depth.png", dpi=300, width=6, height=4)
# %%
