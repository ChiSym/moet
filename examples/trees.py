# %%
%load_ext autoreload
%autoreload 2

# %%
import treescope
treescope.basic_interactive_setup(autovisualize_arrays=True)
from moet.model import layer
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

dataset_path = "data/lpm/PUMS"
train_df, test_df = load_huggingface(dataset_path)
df = pl.concat((train_df, test_df), how="vertical")
df = df[["State", "Sex", "Race", "Citizen_status", "Insurance_Medicare", "Educational_attainment", "Total_income", "Age"]]
schema, discretized_df, categorical_idxs = discretize_dataframe(df)
dummies_df = to_dummies(discretized_df)
bool_data, col_names = dummies_to_padded_array(dummies_df, categorical_idxs)
data: Float[Array, "batch_size n_inputs input_dim"] = jnp.where(bool_data, 0, -jnp.inf)
train_data = data[: len(train_df)]
test_data = data[len(train_df) :]

# %%
from moet.utils import mi
import jax

n_mi_samples = 10000
n_subset_vars = 8
jitted_mi = jax.jit(jax.vmap(jax.vmap(mi, in_axes=(1, None)), in_axes=(None, 1)))
mi_estimates = jitted_mi(bool_data[:n_mi_samples, :n_subset_vars].astype(float), bool_data[:n_mi_samples, :n_subset_vars].astype(float))
mi_estimates

# %%
col_names


# %%
import scipy.cluster.hierarchy as sch
import numpy as np

def cluster_mi(mi_estimates, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    normalized_mi_estimates = mi_estimates / np.max(mi_estimates)
    pdist_uncondensed = 1.0 - normalized_mi_estimates
    pdist_condensed = np.concatenate([row[i+1:] for i, row in enumerate(pdist_uncondensed)])
    linkage = sch.linkage(pdist_condensed, method='single', optimal_ordering=True)
    idx = sch.fcluster(linkage, 0, 'distance')
    idx = np.argsort(idx)
    return mi_estimates[idx, :][:, idx]

# %%
sorted_mi_estimates = cluster_mi(np.array(mi_estimates))
sorted_mi_estimates


# %%
key = jax.random.PRNGKey(0)
eps = 1e-10
theta = jnp.maximum(mi_estimates, eps)
theta = jnp.where(jnp.eye(len(mi_estimates)), 0, theta)
theta

# %%
theta /= jnp.sum(theta)
theta_flat = theta.flatten()

# %%
from moet.trees import tree_node, tree_layer, tree

# %%
key = jax.random.PRNGKey(0)
trace = tree_node.simulate(key, (jnp.log(theta), None))
trace

# %%
trace = tree_layer.simulate(key, (jnp.log(theta),))
trace

# %%
import genjax
depth = 3
trace = tree.simulate(key, (jnp.log(theta), genjax.Pytree.const(depth)))
trace

# %%
layer_list = trace.get_retval()
layer_list

# %%
theta_list = [trace.subtraces[f"{i}"].args[0] for i in range(depth)]
theta_list

# %%
[jax.nn.logsumexp(theta_list[i]) for i in range(depth)]


# %%
from jaxtyping import Int
def build_layer(prev_layer: Int[Array, "n"], pairs: Int[Array, "n%2+n//2 2"]):
    adjacency: Bool[Array, "n%2+n//2 n"] = jax.vmap(jax.vmap(lambda x, y: jnp.isin(x, y), in_axes=(0, None)), in_axes=(None, 0))(prev_layer, pairs)
    output = jax.vmap(lambda x: jnp.argwhere(x, size=1)[0][0], in_axes=1)(adjacency)
    return output

def build_all_layers(n, layer_list):
    layer = jnp.arange(n)
    outputs = [layer]
    for merge_pairs in layer_list:
        layer = build_layer(layer, merge_pairs)
        outputs.append(layer)
    return outputs

# %%
layers = build_all_layers(len(theta), layer_list)
layers
# %%
import plotly.express as px
import numpy as np
import polars as pl
tree_df = pl.DataFrame(
    {
        f"layer_{i}": [f"Layer {i} Node {int(j)}" for j in layers[i]] for i in range(len(layers))
    } | {
        "name": ['State', 'Sex', 'Race', 'Citizen', 'Medicare', 'Education', 'Income', 'Age']
    }
)

# %%
fig = px.treemap(tree_df, path=['layer_3', 'layer_2', 'layer_1', 'name'], 
                #   color='layer_2', hover_data=['layer_0'],
                #   color_continuous_scale='RdBu',
                #   color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
)
# fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.update_traces(marker=dict(cornerradius=5))
fig.show()
# %%
colors_list = ["lightblue" if i % 2 == 0 else "lightgreen" for i in range(len(fig.data[0]['ids']))]

mi_values = []
for i, id in enumerate(fig.data[0]['ids']):
    layers = id.split('/')
    layer_idx = depth + 1 - len(layers)
    if layer_idx == 0:
        continue
    else:
        layer_idx -= 1
        theta_idx = theta_list[layer_idx]
        layer_list_idx = layer_list[layer_idx]
        merge_idx = int(layers[-1].split(' ')[-1])
        pair_idx = layer_list_idx[merge_idx]
        # pair = theta_idx[pair_idx]
        mi = theta[pair_idx[0], pair_idx[1]]
        mi_values.append(float(mi))


# %%
import plotly.colors as pc
log_mi_values = [float(jnp.log(mi)) for mi in mi_values]
log_mi_values


# %%
# Create a colorscale for the MI values
min_mi = min(log_mi_values)
max_mi = max(log_mi_values)

# Normalize the MI values to range [0, 1]
normalized_mi = [(mi - min_mi) / (max_mi - min_mi) if max_mi > min_mi else 0.5 for mi in log_mi_values]

# Get a continuous colorscale (using Viridis as default, but you can change it)
colorscale = px.colors.sequential.Blues

# Map each normalized MI value to a color in the colorscale
leaf_colors = ["#cccccc" for _ in range(8)]
mi_colors = [pc.sample_colorscale(colorscale, val)[0] for val in normalized_mi]

# Update the treemap with the new colors
fig.update_traces(marker=dict(
    colors=leaf_colors + mi_colors,
    cornerradius=5
))

# Add a colorbar to show the MI value scale
fig.update_layout(
    coloraxis=dict(
        colorscale=colorscale,
        colorbar=dict(
            title="Mutual Information",
            tickvals=[0, 0.5, 1],
            ticktext=[f"{min_mi:.2f}", f"{(min_mi + max_mi)/2:.2f}", f"{max_mi:.2f}"]
        ),
        cmin=min_mi,
        cmax=max_mi
    )
)
fig.update_layout(
    # textfont_size=20,
    font=dict(size=20)
)
# Update text alignment to center horizontally
fig.update_traces(
    textposition='middle center',  # Center align text both horizontally and vertically
    # insidetextanchor='middle'      # Ensure text stays anchored to the middle
)

# Add a pattern (smoker effect) only to the leaf nodes
# The first 8 elements in our colors list are leaf nodes
# Update the marker with a single pattern configuration
# In Plotly, the pattern property for treemap.marker should be a single Pattern object, not a list
fig.update_traces(
    marker=dict(
        pattern=dict(shape=["+"] * 8, solidity=.2, size=16, fgcolor="white", bgcolor="#dddddd")
    )
)

# Alternatively, if we need different patterns for different nodes,
# we would need to create separate traces for leaf nodes and non-leaf nodes


# Show the updated figure
fig.show()

# %%
fig.write_image("tree_sampling.pdf")


# %%
theta
# %%
fig_heatmap = px.imshow(theta, color_continuous_scale='Blues')
fig_heatmap.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
fig_heatmap.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')

# Rename tick labels to use tree_df["name"]
# Assuming tree_df is available and contains a "name" column
# For x-axis (columns)
fig_heatmap.update_xaxes(
    ticktext=tree_df["name"],
    tickvals=list(range(len(tree_df["name"])))
)

# For y-axis (rows)
fig_heatmap.update_yaxes(
    ticktext=tree_df["name"],
    tickvals=list(range(len(tree_df["name"])))
)
# Update the color scale to use logarithmic scaling
# Handle zero/negative values by making them appear as lightgray (not on the scale)
import numpy as np
# Create a mask for zero or very small values
mask = theta <= 1e-10
# Apply log scaling to valid values
log_data = np.log10(theta.copy())
# Replace -inf values (from log of zeros/negatives) with None to exclude from colorscale
log_data[mask] = None

# Update the heatmap with the log-scaled data
fig_heatmap.data[0].z = log_data
# Set the pattern for cells with None values to appear as lightgray
fig_heatmap.update_traces(
    zmin=np.nanmin(log_data[~mask]),  # Set min value excluding masked cells
    zmid=np.nanmedian(log_data[~mask]),  # Set middle value for color scale
    zmax=np.nanmax(log_data[~mask])  # Set max value excluding masked cells
)

# Update the color scale to use logarithmic scaling
fig_heatmap.update_traces(
    colorbar=dict(
        title="log(value)",
        tickmode="auto"
    )
)


# Show the updated figure
# Unfortunately, heatmaps don't support pattern fills directly
# We can use a custom hover template to indicate masked cells
fig_heatmap.update_traces(
    hovertemplate='x: %{x}<br>y: %{y}<br>value: %{z}<extra></extra>',
    # Set a custom color for masked cells
    # We can't use patterns, but we can use a specific color for masked values
    zmin=np.nanmin(log_data[~mask]),
    zmid=np.nanmedian(log_data[~mask]),
    zmax=np.nanmax(log_data[~mask])
)
# Update the colorbar title to "log(MI)"
# fig_heatmap.update_layout(
#     coloraxis_colorbar=dict(
#         title="log(MI)",
#         tickmode="auto"
#     )
# )
# Update the layout to place the colorbar on the left side
fig_heatmap.update_layout(
    coloraxis=dict(
        colorbar=dict(
            # title="log(value)",
            tickmode="auto",
            x=-0.1,  # Position at the left (0) instead of right (1)
            xpad=10,  # Add some padding between the colorbar and the plot
            len=0.8,  # Length of the colorbar (80% of the plot height)
            yanchor="middle",  # Anchor the colorbar in the middle vertically
            y=0.5,  # Position in the middle vertically
            # ticks="inside",  # Place ticks outside
            ticklabelposition="inside right"  # Place tick labels outside to the left of the colorbar
        )
    )
)




fig_heatmap.show()
fig_heatmap.write_image("heatmap.png")
# %%
