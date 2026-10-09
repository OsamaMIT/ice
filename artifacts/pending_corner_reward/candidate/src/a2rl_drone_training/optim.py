from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

Params = Any
OptState = dict[str, Any]


def adam_init(params: Params) -> OptState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return {"m": zeros, "v": zeros, "t": jnp.array(0, dtype=jnp.int32)}


def adam_update(
    params: Params,
    grads: Params,
    state: OptState,
    *,
    lr: float,
    max_grad_norm: float | None = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> tuple[Params, OptState, Array]:
    if max_grad_norm is not None:
        grads, grad_norm = clip_by_global_norm(grads, max_grad_norm)
    else:
        grad_norm = tree_l2_norm(grads)

    t = state["t"] + 1
    m = jax.tree_util.tree_map(lambda m_, g: beta1 * m_ + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(
        lambda v_, g: beta2 * v_ + (1.0 - beta2) * jnp.square(g), state["v"], grads
    )
    t_float = t.astype(jnp.float32)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**t_float), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**t_float), v)
    new_params = jax.tree_util.tree_map(
        lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
    )
    return new_params, {"m": m, "v": v, "t": t}, grad_norm


def clip_by_global_norm(grads: Params, max_norm: float) -> tuple[Params, Array]:
    norm = tree_l2_norm(grads)
    scale = jnp.minimum(1.0, max_norm / (norm + 1.0e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm


def tree_l2_norm(tree: Params) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.array(0.0, dtype=jnp.float32)
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in leaves))
