# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HF modeling for the SODA hierarchical (backbone + depth) audio LM.

This file is copied verbatim into every exported checkpoint directory and
loaded via trust_remote_code, so it may import only torch/transformers.

The model consumes and produces the same flat frame-interleaved token stream
as the flattened arm (8 audio ids per Mimi frame: semantic then 7 acoustics),
so it is a drop-in for likelihood evals that gather next-token log-probs from
``model(ids).logits`` and for ``model.generate``:

- Internally, positions are grouped into backbone "steps": one text/special
  token, or one whole frame (its 8 codebook embeddings summed).
- ``logits[t]`` is the model's true factorized conditional for token ``t+1``:
  the 130,308-way unified head (text/special/semantic ids — identical to flat
  ids 0..130307) when ``t+1`` starts a step, or the 2,048-way depth head for
  codebook k mapped into its flat id block when ``t+1`` is acoustic. All other
  vocabulary entries are -inf, so ``log_softmax`` reproduces the factorized
  log-prob exactly.
- The depth factorization matches training exactly: codebook k is predicted
  from the backbone hidden of the PREVIOUS step plus codebooks 0..k-2 of the
  same frame (the immediately preceding codebook is not in the conditioning
  set for k >= 2 — a property of the trained shifted-prefix scheme, replicated
  verbatim).
"""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from .configuration_soda_hier import SodaHierConfig


def _group_steps(ids: torch.Tensor, audio_id_lo: int, num_codebooks: int):
    """Map a flat interleaved stream to steps.

    Returns (steps, step_of, slot_of): ``steps`` is (S, num_codebooks) with -1
    padding on non-frame steps; ``step_of[t]``/``slot_of[t]`` locate position t.
    """
    T = ids.shape[0]
    step_of = torch.empty(T, dtype=torch.long)
    slot_of = torch.empty(T, dtype=torch.long)
    rows: list[list[int]] = []
    run = 0
    for t in range(T):
        tok = int(ids[t])
        if tok < audio_id_lo:
            rows.append([tok] + [-1] * (num_codebooks - 1))
            run = 0
        else:
            if run % num_codebooks == 0:
                rows.append([-1] * num_codebooks)
            rows[-1][run % num_codebooks] = tok
            slot_of[t] = run % num_codebooks
            step_of[t] = len(rows) - 1
            run += 1
            continue
        step_of[t] = len(rows) - 1
        slot_of[t] = 0
    steps = torch.tensor(rows, dtype=torch.long)
    return steps, step_of, slot_of


class SodaHierForCausalLM(PreTrainedModel):
    config_class = SodaHierConfig
    _no_split_modules = ["Qwen3DecoderLayer"]
    main_input_name = "input_ids"
    _tied_weights_keys = []

    def __init__(self, config: SodaHierConfig):
        super().__init__(config)
        self.backbone = Qwen3Model(config.backbone_config())
        self.depth = Qwen3Model(config.depth_config())
        e, e_d = config.hidden_size, config.depth_hidden_size
        self.unified_head = nn.Linear(e, config.unified_vocab_size, bias=False)
        self.bd_proj = nn.Linear(e, e_d, bias=False)
        self.acoustic_heads = nn.ModuleList(
            nn.Linear(e_d, config.codebook_size, bias=False) for _ in range(config.num_codebooks - 1)
        )
        self.post_init()

    # ------------------------------------------------------------------ core

    def _embed_steps(self, steps: torch.Tensor) -> torch.Tensor:
        """(S, num_codebooks) step ids (-1 = empty slot) -> (S, E) summed embeddings."""
        valid = steps >= 0
        emb = self.backbone.embed_tokens(steps.clamp(min=0))
        return (emb * valid.unsqueeze(-1)).sum(dim=1)

    def _depth_hidden_for_frames(self, cond: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        """Teacher-forced depth pass. cond (F, E_d); frames (F, 8) LM ids -> (F, 8, E_d)."""
        cfg = self.config
        audio_idx = (frames - cfg.audio_id_lo).clamp(0, cfg.num_codebooks * cfg.codebook_size - 1)
        prefix = self.depth.embed_tokens(audio_idx)  # (F, 8, E_d)
        shifted = torch.roll(prefix, 1, dims=1)
        shifted[:, 0] = 0.0
        x = cond.unsqueeze(1) + shifted
        return self.depth(inputs_embeds=x).last_hidden_state

    def _forward_one(self, ids: torch.Tensor) -> torch.Tensor:
        """One unpadded row (T,) -> logits (T, vocab)."""
        cfg = self.config
        dev = ids.device
        steps, step_of, slot_of = _group_steps(ids.cpu(), cfg.audio_id_lo, cfg.num_codebooks)
        steps, step_of, slot_of = steps.to(dev), step_of.to(dev), slot_of.to(dev)
        T = ids.shape[0]

        emb = self._embed_steps(steps)  # (S, E)
        h = self.backbone(inputs_embeds=emb.unsqueeze(0)).last_hidden_state[0]  # (S, E)
        u = self.unified_head(h)  # (S, unified)

        is_audio = ids >= cfg.audio_id_lo
        is_frame_step = steps[:, 1] >= 0  # frame steps have slot-1 filled
        frame_steps = torch.nonzero(is_frame_step, as_tuple=False)[:, 0]
        # depth conditions on the hidden of the step BEFORE the frame
        cond_frames = frame_steps[frame_steps >= 1]
        d = None
        frame_row = torch.full((steps.shape[0],), -1, dtype=torch.long, device=dev)
        if len(cond_frames):
            d = self._depth_hidden_for_frames(self.bd_proj(h[cond_frames - 1]), steps[cond_frames])
            frame_row[cond_frames] = torch.arange(len(cond_frames), device=dev)

        logits = torch.full((T, cfg.vocab_size), float("-inf"), dtype=h.dtype, device=dev)
        # positions whose NEXT token starts a step: non-audio positions and slot-7 audio positions
        primary = (~is_audio) | (slot_of == cfg.num_codebooks - 1)
        logits[primary, : cfg.unified_vocab_size] = u[step_of[primary]]
        # positions whose next token is acoustic codebook k+1 of the SAME frame
        for k in range(cfg.num_codebooks - 1):
            pos = torch.nonzero(is_audio & (slot_of == k), as_tuple=False)[:, 0]
            if not len(pos):
                continue
            rows = frame_row[step_of[pos]]
            ok = rows >= 0
            lo = cfg.audio_id_lo + (k + 1) * cfg.codebook_size
            if ok.any():
                logits[pos[ok], lo : lo + cfg.codebook_size] = self.acoustic_heads[k](d[rows[ok], k])
            if (~ok).any():
                # frame at step 0: the factorization defines no conditional; keep finite
                logits[pos[~ok]] = 0.0
        return logits

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("SodaHierForCausalLM.forward requires input_ids")
        rows = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b]
            if attention_mask is not None:
                length = int(attention_mask[b].sum())
                logits_b = torch.zeros(
                    (ids.shape[0], self.config.vocab_size), dtype=torch.float32, device=ids.device
                )
                logits_b[:length] = self._forward_one(ids[:length])
                rows.append(logits_b)
            else:
                rows.append(self._forward_one(ids))
        logits = torch.stack(rows)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].reshape(-1, self.config.vocab_size)
            shift_labels = labels[:, 1:].reshape(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # -------------------------------------------------------------- sampling

    @staticmethod
    def _sample(logits: torch.Tensor, do_sample: bool, temperature: float, top_p: float) -> int:
        if not do_sample or temperature <= 0:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = probs.sort(descending=True)
            cum = sorted_probs.cumsum(-1)
            # drop tokens entirely beyond the nucleus; the crossing token stays
            remove = (cum - sorted_probs) > top_p
            sorted_probs[remove] = 0.0
            sorted_probs /= sorted_probs.sum()
            return int(sorted_idx[torch.multinomial(sorted_probs, 1)])
        return int(torch.multinomial(probs, 1))

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        eos_token_id=None,
        pad_token_id=None,
        **ignored,
    ) -> torch.Tensor:
        """Two-stage autoregressive decode over the flat interleaved stream.

        The backbone advances once per step with a KV cache; whenever the
        unified head emits a semantic token, the depth transformer fills in
        the frame's 7 acoustic codebooks before the backbone moves on. Only
        whole frames are emitted: if fewer than 8 tokens of budget remain
        when a frame starts, generation stops early instead.
        """
        cfg = self.config
        if input_ids.shape[0] != 1:
            raise NotImplementedError("SodaHierForCausalLM.generate supports batch size 1")
        dev = input_ids.device
        eos = set()
        if eos_token_id is not None:
            eos = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id)

        ids = input_ids[0]
        steps, _, _ = _group_steps(ids.cpu(), cfg.audio_id_lo, cfg.num_codebooks)
        steps = steps.to(dev)
        emb = self._embed_steps(steps)

        cache = DynamicCache()
        out = self.backbone(inputs_embeds=emb.unsqueeze(0), past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        h_last = out.last_hidden_state[0, -1]
        step_pos = steps.shape[0]

        generated: list[int] = []
        while len(generated) < max_new_tokens:
            primary = self._sample(self.unified_head(h_last), do_sample, temperature, top_p)
            if primary < cfg.audio_id_lo:  # text or special: a one-token step
                generated.append(primary)
                next_emb = self.backbone.embed_tokens(torch.tensor([primary], device=dev))[0]
                if primary in eos:
                    break
            else:  # semantic token: emit a whole frame via the depth transformer
                if max_new_tokens - len(generated) < cfg.num_codebooks:
                    break
                cond = self.bd_proj(h_last)
                frame = [primary]
                xs = [cond]
                for j in range(cfg.num_codebooks - 1):
                    d = self.depth(inputs_embeds=torch.stack(xs).unsqueeze(0)).last_hidden_state[0, -1]
                    idx = self._sample(self.acoustic_heads[j](d), do_sample, temperature, top_p)
                    frame.append(cfg.audio_id_lo + (j + 1) * cfg.codebook_size + idx)
                    xs.append(cond + self.depth.embed_tokens(torch.tensor(frame[-1] - cfg.audio_id_lo, device=dev)))
                generated.extend(frame)
                frame_t = torch.tensor(frame, device=dev)
                next_emb = self.backbone.embed_tokens(frame_t).sum(dim=0)
                if eos & set(frame):
                    break
            out = self.backbone(
                inputs_embeds=next_emb.view(1, 1, -1),
                past_key_values=cache,
                use_cache=True,
                position_ids=torch.tensor([[step_pos]], device=dev),
            )
            cache = out.past_key_values
            h_last = out.last_hidden_state[0, -1]
            step_pos += 1

        return torch.cat([ids, torch.tensor(generated, dtype=ids.dtype, device=dev)]).unsqueeze(0)
