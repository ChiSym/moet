import jax
import jax.numpy as jnp
from jaxtyping import Float, Array
from moet.model import layer
from beartype import beartype as typechecker
import optax


@typechecker
def mixture_model(
    X: Float[Array, "n_inputs input_dim"],
    Q: Float[Array, "n_inputs n_clusters input_dim"],
    W: Float[Array, "n_clusters"],
) -> Float[Array, ""]:
    merge = jnp.arange(X.shape[0])[None, :]  # merge all inputs
    Y: Float[Array, "n_clusters"] = layer(X, Q, merge)[0]
    weighted_Y = W + Y
    return jax.nn.logsumexp(weighted_Y)


def test_normalized_mixture_model():
    X = jnp.array([[-jnp.inf, 0], [-jnp.inf, 0]])
    Q = jnp.array(
        [
            [
                [jnp.log(0.25), jnp.log(0.75)],
                [jnp.log(0.5), jnp.log(0.5)],
            ],
            [
                [jnp.log(0.3), jnp.log(0.7)],
                [jnp.log(0.5), jnp.log(0.5)],
            ],
        ]
    )
    W = jnp.array([jnp.log(0.25), jnp.log(0.75)])
    logprob = mixture_model(X, Q, W)

    assert jnp.isclose(logprob, jnp.log(0.75 * 0.5 * 0.5 + 0.25 * 0.75 * 0.7))


def test_unnormalized_mixture_model():
    X = jnp.array([[-jnp.inf, 0], [-jnp.inf, 0]])
    Q = jnp.ones((2, 2, 2))
    W = jnp.array([0.0, 0.0])
    logprob = mixture_model(X, Q, W)
    logZ = mixture_model(jnp.zeros_like(X), Q, W)

    assert jnp.isclose(logprob - logZ, jnp.log(0.5 * 0.5))


@jax.jit
def loss_fn(
    X: Float[Array, "batch_size n_inputs input_dim"],
    Q: Float[Array, "n_inputs n_clusters input_dim"],
    W: Float[Array, "n_clusters"],
) -> Float[Array, ""]:
    n_batch = X.shape[0]
    pad = jnp.zeros_like(X[0:1])
    X_pad = jnp.concatenate([X, pad], axis=0)
    out = jax.vmap(mixture_model, in_axes=(0, None, None))(X_pad, Q, W)
    log_Z = out[-1]
    return -(jnp.sum(out[:-1]) - n_batch * log_Z) / n_batch


def test_learn_mixture_model():
    n_samples = 200
    n_inputs = 2
    input_dim = 2
    n_clusters = 2
    X1: Float[Array, "n_samples//2 n_inputs input_dim"] = jnp.array(
        [[[-jnp.inf, 0], [0, -jnp.inf]]]
    ).repeat(n_samples // 2, axis=0)
    X2: Float[Array, "n_samples//2 n_inputs input_dim"] = jnp.array(
        [[[0, -jnp.inf], [-jnp.inf, 0]]]
    ).repeat(n_samples // 2, axis=0)
    X: Float[Array, "n_samples n_inputs input_dim"] = jnp.concatenate((X1, X2), axis=0)

    key = jax.random.PRNGKey(1234)
    Q: Float[Array, "n_inputs n_clusters input_dim"] = jax.random.normal(
        key, (n_inputs, n_clusters, input_dim)
    )
    W: Float[Array, "n_clusters"] = jax.random.normal(key, (n_clusters,))

    optimizer = optax.adam(1)
    opt_state = optimizer.init((Q, W))

    n_epochs = 100
    for _ in range(n_epochs):
        loss, grads = jax.value_and_grad(loss_fn, argnums=(1, 2))(X, Q, W)
        updates, opt_state = optimizer.update(grads, opt_state, (Q, W))
        Q, W = optax.apply_updates((Q, W), updates)

    assert jnp.isclose(loss, -jnp.log(0.5), atol=1e-2)

    # normalize via algorithm 1 here: http://proceedings.mlr.press/v38/peharz15.pdf
    Q_Z: Float[Array, "n_inputs n_clusters"] = jax.nn.logsumexp(Q, axis=-1)
    Q_normalized: Float[Array, "n_inputs n_clusters input_dim"] = jnp.exp(
        Q - Q_Z[..., None]
    )
    W_normalized = jax.nn.softmax(W + Q_Z.sum(axis=0))

    Q_true1 = jnp.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]])
    Q_true2 = jnp.array([[[0, 1], [1, 0]], [[1, 0], [0, 1]]])
    W_true = jnp.array([0.5, 0.5])

    assert jnp.allclose(Q_normalized, Q_true1, atol=1e-2) or jnp.allclose(
        Q_normalized, Q_true2, atol=1e-2
    )
    assert jnp.allclose(W_normalized, W_true, atol=2e-2)
