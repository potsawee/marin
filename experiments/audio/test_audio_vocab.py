# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire-format contract tests for the mm-pretrain audio id space.

The Unicode encoding and the BPE-tokenizer id layout ARE the dataset wire
format; expected values below are hand-derived from the released tokenizer
(``potsawee/marin-mimi-bpe-8cb-16k-tokenizer``) and cross-checked against it
in ``test_codes_to_lm_ids_matches_released_tokenizer``.
"""

import numpy as np
import pytest

from experiments.audio.audio_vocab import (
    NUM_CODEBOOKS,
    TOKENIZER_ID,
    acoustic_codebook_of,
    chars_to_codes,
    codes_to_chars,
    codes_to_lm_ids,
    is_acoustic_id,
    is_semantic_id,
    is_special_id,
    is_text_id,
    lm_ids_to_codes,
)

# One frame of raw codebook indices (cb0..cb7) used as the known-value fixture.
FRAME = [5, 2047, 0, 1023, 7, 511, 255, 1]
# chr(0xE000 + cb*2048 + idx) for each position.
FRAME_CHARS = "\ue005\uefff\uf000\ufbff\U00010007\U000109ff\U000110ff\U00011801"
# Verified against the released tokenizer: tok(FRAME_CHARS)["input_ids"].
FRAME_LM_IDS = [128265, 132355, 132356, 135427, 136459, 139011, 140803, 142597]


def test_chars_to_codes_known_frame_exact_values():
    codes = chars_to_codes(FRAME_CHARS)
    assert codes.shape == (1, NUM_CODEBOOKS)
    assert codes[0].tolist() == FRAME
    assert codes_to_lm_ids(codes)[0].tolist() == FRAME_LM_IDS


def test_roundtrip_codes_chars_codes():
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 2048, size=(17, NUM_CODEBOOKS), dtype=np.int32)
    assert np.array_equal(chars_to_codes(codes_to_chars(codes)), codes)


def test_roundtrip_lm_ids():
    rng = np.random.default_rng(1)
    codes = rng.integers(0, 2048, size=(5, NUM_CODEBOOKS), dtype=np.int32)
    assert np.array_equal(lm_ids_to_codes(codes_to_lm_ids(codes)), codes)


def test_chars_to_codes_rejects_partial_frame():
    with pytest.raises(ValueError):
        chars_to_codes(FRAME_CHARS[:-1])


def test_chars_to_codes_rejects_char_in_wrong_codebook_block():
    # A codebook-1 char in the codebook-0 slot must be rejected, not silently wrapped.
    swapped = FRAME_CHARS[1] + FRAME_CHARS[1:]
    with pytest.raises(ValueError):
        chars_to_codes(swapped)


@pytest.mark.parametrize(
    ("lm_id", "expected"),
    [
        (0, "text"),
        (128000, "text"),  # <|begin_of_text|> counts as text vocab
        (128255, "text"),
        (128256, "special"),  # <|text_start|>
        (128259, "special"),  # <|audio_end|>
        (128260, "semantic"),  # first audio id
        (130307, "semantic"),
        (130308, "acoustic"),
        (144643, "acoustic"),
    ],
)
def test_id_range_predicates_classify_hand_picked_boundaries(lm_id, expected):
    kinds = {
        "text": is_text_id(lm_id),
        "special": is_special_id(lm_id),
        "semantic": is_semantic_id(lm_id),
        "acoustic": is_acoustic_id(lm_id),
    }
    assert kinds.pop(expected)
    assert not any(kinds.values())


def test_acoustic_codebook_of_boundaries():
    assert acoustic_codebook_of(np.array([130308, 132355, 144643])).tolist() == [1, 1, 7]


@pytest.mark.data_integration
@pytest.mark.timeout(300)  # cold torch/transformers import on shared NFS exceeds the 60s default
def test_codes_to_lm_ids_matches_released_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except OSError:
        pytest.skip("released tokenizer not available (offline and not cached)")
    rng = np.random.default_rng(2)
    codes = rng.integers(0, 2048, size=(3, NUM_CODEBOOKS), dtype=np.int32)
    ids = tok(codes_to_chars(codes), add_special_tokens=False)["input_ids"]
    assert ids == codes_to_lm_ids(codes).reshape(-1).tolist()
