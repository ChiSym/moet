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
from moet.io import load_data
import jax

dataset_path = "data/lpm/PUMS"
train_data, test_data, col_names = load_data(dataset_path)
# %%
from moet.utils import get_mi_theta
theta = get_mi_theta(train_data)


# %%
from moet.trees import tree
import genjax

n_categories, max_categories = train_data.shape[1:]
n_categories = int(n_categories)
max_categories = int(max_categories)

n_clusters = [512, 256, 128, 64, 32]
# %%
from moet.utils import to_tuple
import optax
from moet.model import loss_fn


# %%
from moet.model import make_circuit_parameters
def run_depth_experiment(key, circuit_depth, batch_size=512, test_loss_size=10000, max_batches=1000, learning_rate=1e-1):
    key, subkey = jax.random.split(key)
    trace = tree.simulate(key, (jnp.log(theta), genjax.Pytree.const(5)))

    layers = [trace.subtraces[f"{i}"].get_choices()['merge'] 
            for i in range(5)]
    Qs, W, final_layer = make_circuit_parameters(subkey, circuit_depth, n_clusters, n_categories, max_categories)

    new_layers = layers[:circuit_depth - 1] + [final_layer]
    new_layers = tuple(np.array(l) for l in new_layers)

    int_layers = to_tuple(new_layers)


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

    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers(): buf.delete()    

    return np.array(train_losses), np.array(test_losses)

# %%
from tqdm import tqdm
results_dict = {}
n_replicates = 20
for depth in tqdm(range(1, 5)):
    depth_dict = {}
    for i in tqdm(range(n_replicates)):
        key = jax.random.PRNGKey(i)
        train_losses, test_losses = run_depth_experiment(key, depth)
        depth_dict[i] = {
            "train_losses": train_losses,
            "test_losses": test_losses,
        }
    results_dict[depth] = depth_dict

# %%
import polars as pl

# Create a list to store all the data
data = []

# Iterate through the results dictionary
for depth, results in results_dict.items():
    for replicate, losses in results.items():
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
from plotnine import (ggplot, aes, geom_line, labs, theme_bw, 
                      scale_color_manual, geom_smooth, geom_point, element_text,
                       theme, scale_y_continuous, position_jitter)

# Create the line plot with iter as x-axis, train loss as y-axis, and depth as color
plot = (
    ggplot(results_df, aes(x='iter', y='test_probability', color='depth'))
    # + geom_line(size=1.5)
    # + geom_point()
    + geom_smooth(method='loess', se=True)
    + geom_point(position=position_jitter(width=.5, height=0), alpha=0.5)
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
