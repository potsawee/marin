# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math
import time

import jax
import jax.numpy as jnp

print("device:", jax.devices()[0])
B, H, S, D = 2, 6, 4096, 128  # p1-flat per-device attention shape (d=768, heads=6)
k = jax.random.PRNGKey(0)
q, kk, v = [jax.random.normal(jax.random.fold_in(k, i), (B, S, H, D), jnp.bfloat16) for i in range(3)]


def materialized(q, kk, v):  # what VANILLA/reference does: O(S^2) scores
    qs = q.transpose(0, 2, 1, 3)
    ks = kk.transpose(0, 2, 1, 3)
    vs = v.transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhqd,bhkd->bhqk", qs, ks) / math.sqrt(D)
    m = jnp.tril(jnp.ones((S, S), bool))
    scores = jnp.where(m, scores, -1e30)
    p = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(jnp.bfloat16)
    return jnp.einsum("bhqk,bhkd->bhqd", p, vs)


def dpa(impl):
    def f(q, kk, v):
        return jax.nn.dot_product_attention(q, kk, v, is_causal=True, implementation=impl)

    return f


for name, fn in [("materialized", materialized), ("dpa-xla", dpa("xla")), ("dpa-cudnn", dpa("cudnn"))]:
    try:
        g = jax.jit(fn)
        o = g(q, kk, v)
        o.block_until_ready()
        ts = []
        for _ in range(10):
            t = time.time()
            o = g(q, kk, v)
            o.block_until_ready()
            ts.append(time.time() - t)
        print(f"  {name:14s} {min(ts)*1000:7.2f} ms/call")
    except Exception as e:
        print(f"  {name:14s} FAILED: {type(e).__name__}: {str(e)[:100]}")
