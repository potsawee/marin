import time, equinox as eqx, jax, jax.numpy as jnp, haliax as hax
from haliax import Axis
from experiments.audio.isoflop_audio_target import flat_dims
from experiments.audio.audio_flops import arm_f_fwd_per_token
from experiments.audio.audio_vocab import FULL_VOCAB
from levanter.models.lm_model import LmExample
from levanter.layers.attention import AttentionMask

print("device:", jax.devices()[0])
cfg = flat_dims(768); Vocab = Axis("vocab", FULL_VOCAB)
model = cfg.build(Vocab, key=jax.random.PRNGKey(0))
model = hax.auto_sharded(model) if False else model
fwd_per_tok = arm_f_fwd_per_token(cfg); PEAK = 362.05e12
Pos = cfg.max_Pos.resize(4096)

def make(bs):
    toks = jax.random.randint(jax.random.PRNGKey(1), (bs, 4096), 0, FULL_VOCAB)
    return LmExample(
        tokens=hax.named(toks, ("batch","position")),
        loss_weight=hax.named(jnp.ones((bs,4096), jnp.float32), ("batch","position")),
        attn_mask=AttentionMask.causal(),
    )

@eqx.filter_jit
def step(m, ex):
    def loss(m): return m.compute_next_token_loss(ex, key=None).scalar()
    return eqx.filter_value_and_grad(loss)(m)

for bs in (2, 8, 32):
    ex = make(bs)
    (l, g) = step(model, ex); l.block_until_ready()
    ts=[]
    for _ in range(8):
        t=time.time(); (l,g)=step(model, ex); jax.block_until_ready(g); ts.append(time.time()-t)
    dt=min(ts); flops=3*fwd_per_tok*bs*4096
    print(f"  batch={bs:2d} (1 GPU): {dt*1000:7.1f} ms/step  MFU={flops/(PEAK*dt)*100:5.1f}%  ({bs*4096/dt/1e3:.0f}k tok/s)")
