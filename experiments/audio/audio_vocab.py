# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Id-space constants and Unicode<->code helpers for SODA mm-pretrain audio data.

The ``soda-research/*-mm-pretrain`` datasets store Mimi RVQ audio inside each
document string as Unicode private-use characters: frame-major, 8 codebooks per
frame (1 semantic + 7 acoustic), codebook size 2048, with
``char = chr(UNICODE_OFFSET + codebook * CODEBOOK_SIZE + index)``. The released
BPE tokenizer (``potsawee/marin-mimi-bpe-8cb-16k-tokenizer``, vocab 144,644)
maps each audio char to the LM id ``AUDIO_ID_LO + codebook * CODEBOOK_SIZE +
index``; all constants below are verified against that tokenizer.

Orientation note: helpers here use frame-major ``(frames, codebooks)`` arrays,
matching the on-disk char order. The original blueberry-eval helpers used the
transposed ``(codebooks, frames)`` orientation.
"""

import numpy as np

TOKENIZER_ID = "potsawee/marin-mimi-bpe-8cb-16k-tokenizer"

UNICODE_OFFSET = 0xE000
NUM_CODEBOOKS = 8  # 1 semantic + 7 acoustic
CODEBOOK_SIZE = 2048
NUM_AUDIO_TOKENS = NUM_CODEBOOKS * CODEBOOK_SIZE  # 16384

# LM id layout: [0, TEXT_VOCAB) Llama-3 BPE text; 4 specials; then audio.
TEXT_VOCAB = 128256
BOS_ID = 128000  # <|begin_of_text|>
EOS_ID = 128001  # <|end_of_text|>
TEXT_START_ID = 128256  # <|text_start|>
TEXT_END_ID = 128257  # <|text_end|>
AUDIO_START_ID = 128258  # <|audio_start|>
AUDIO_END_ID = 128259  # <|audio_end|>
AUDIO_ID_LO = 128260
SEMANTIC_LO = AUDIO_ID_LO  # codebook 0
SEMANTIC_HI = AUDIO_ID_LO + CODEBOOK_SIZE  # 130308, exclusive
ACOUSTIC_LO = SEMANTIC_HI  # codebooks 1..7
FULL_VOCAB = AUDIO_ID_LO + NUM_AUDIO_TOKENS  # 144644
# The hierarchical backbone predicts text + specials + semantic codebook only.
UNIFIED_VOCAB = SEMANTIC_HI  # 130308

FRAME_RATE = 12.5  # Mimi frames per second


def chars_to_codes(chars: str) -> np.ndarray:
    """Decode an audio Unicode span into raw codebook indices.

    Args:
        chars: the contents of one ``<|audio_start|>..<|audio_end|>`` block,
            frame-major (8 consecutive chars per frame, codebook 0 first).

    Returns:
        ``(frames, NUM_CODEBOOKS)`` int32 array of raw indices in
        ``[0, CODEBOOK_SIZE)``.

    Raises:
        ValueError: if the length is not a multiple of ``NUM_CODEBOOKS`` or any
            char falls outside its position's codebook block.
    """
    if len(chars) % NUM_CODEBOOKS != 0:
        raise ValueError(f"audio span length {len(chars)} is not a multiple of {NUM_CODEBOOKS}")
    flat = np.frombuffer(chars.encode("utf-32-le"), dtype=np.uint32).astype(np.int64)
    codes = flat.reshape(-1, NUM_CODEBOOKS) - UNICODE_OFFSET
    codes -= np.arange(NUM_CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    if codes.min(initial=0) < 0 or codes.max(initial=0) >= CODEBOOK_SIZE:
        bad = np.argwhere((codes < 0) | (codes >= CODEBOOK_SIZE))[0]
        raise ValueError(
            f"char at frame {bad[0]}, codebook {bad[1]} is outside its codebook block: "
            f"U+{flat.reshape(-1, NUM_CODEBOOKS)[bad[0], bad[1]]:04X}"
        )
    return codes.astype(np.int32)


def codes_to_chars(codes: np.ndarray) -> str:
    """Encode raw codebook indices ``(frames, NUM_CODEBOOKS)`` back into the Unicode span."""
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != NUM_CODEBOOKS:
        raise ValueError(f"expected (frames, {NUM_CODEBOOKS}) array, got shape {codes.shape}")
    if codes.min(initial=0) < 0 or codes.max(initial=0) >= CODEBOOK_SIZE:
        raise ValueError(f"codes must lie in [0, {CODEBOOK_SIZE})")
    points = codes.astype(np.int64) + UNICODE_OFFSET + np.arange(NUM_CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    return points.reshape(-1).astype(np.uint32).tobytes().decode("utf-32-le")


def codes_to_lm_ids(codes: np.ndarray) -> np.ndarray:
    """Map raw codebook indices ``(frames, NUM_CODEBOOKS)`` to BPE-tokenizer LM ids."""
    codes = np.asarray(codes, dtype=np.int64)
    return (codes + AUDIO_ID_LO + np.arange(NUM_CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE).astype(np.int32)


def lm_ids_to_codes(ids: np.ndarray) -> np.ndarray:
    """Inverse of :func:`codes_to_lm_ids`, validating that ids sit in the right codebook blocks."""
    ids = np.asarray(ids, dtype=np.int64)
    codes = ids - AUDIO_ID_LO - np.arange(NUM_CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    if codes.min(initial=0) < 0 or codes.max(initial=0) >= CODEBOOK_SIZE:
        raise ValueError("ids are not a frame-major audio block in the expected codebook layout")
    return codes.astype(np.int32)


def is_text_id(ids: np.ndarray) -> np.ndarray:
    """True for plain Llama-3 BPE text ids (BOS/EOS and llama specials included)."""
    ids = np.asarray(ids)
    return ids < TEXT_VOCAB


def is_special_id(ids: np.ndarray) -> np.ndarray:
    """True for the four audio/text block-marker specials."""
    ids = np.asarray(ids)
    return (ids >= TEXT_VOCAB) & (ids < AUDIO_ID_LO)


def is_semantic_id(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    return (ids >= SEMANTIC_LO) & (ids < SEMANTIC_HI)


def is_acoustic_id(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    return (ids >= ACOUSTIC_LO) & (ids < FULL_VOCAB)


def is_audio_id(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    return (ids >= AUDIO_ID_LO) & (ids < FULL_VOCAB)


def acoustic_codebook_of(ids: np.ndarray) -> np.ndarray:
    """Acoustic codebook number (1..7) of each id; only meaningful where :func:`is_acoustic_id`."""
    ids = np.asarray(ids)
    return (ids - AUDIO_ID_LO) // CODEBOOK_SIZE
