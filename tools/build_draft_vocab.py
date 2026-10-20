#!/usr/bin/env python3
"""Rebuild patches/draft_vocab_latin.npy, the token ids the MTP drafter may propose with DRAFT_VOCAB=latin.

A token is dropped when its text holds a character of one of the scripts in OTHER_SCRIPTS (Chinese,
Japanese, Korean, Cyrillic, Arabic, Thai, Devanagari, Hebrew) and is not all Latin. Everything else stays:
ASCII, accented Latin, Greek, digits, symbols, byte pieces, and every special token. The list is padded to a
multiple of 64 with the lowest dropped ids.

Run it inside the image, which has `tokenizers`:
  docker run --rm --entrypoint python3 -v "$PWD:/r" -v "$HOME/.cache/huggingface:/hf:ro" \
    eugr/spark-vllm-b12x:pin-a7f76ae9f18c -I /r/tools/build_draft_vocab.py \
    /hf/hub/models--nvidia--Qwen3.8-Flash-Next-FP4-Local/snapshots/<rev>/tokenizer.json /r/patches/draft_vocab_latin.npy
"""
import sys
import unicodedata

import numpy as np
from tokenizers import Tokenizer, pre_tokenizers

OTHER_SCRIPTS = {"CJK", "CYRILLIC", "ARABIC", "HANGUL", "HIRAGANA", "KATAKANA", "THAI", "DEVANAGARI", "HEBREW"}
LATIN_SYMBOLS = set("°±×÷·«»¿¡€£§©®")


def latin(c):
    o = ord(c)
    return o < 128 or 0xC0 <= o <= 0x24F or 0x2000 <= o <= 0x206F or c in LATIN_SYMBOLS


def keep(text):
    if not text or "�" in text or all(latin(c) for c in text):
        return True
    first = next(c for c in text if ord(c) >= 128)
    return unicodedata.name(first, "?").split()[0] not in OTHER_SCRIPTS


tok = Tokenizer.from_file(sys.argv[1])
n = tok.get_vocab_size(with_added_tokens=True)
mask = np.array([keep(tok.decode([i], skip_special_tokens=False)) for i in range(n)])
for i in tok.get_added_tokens_decoder():
    mask[i] = True
for ch in pre_tokenizers.ByteLevel.alphabet():
    mask[tok.token_to_id(ch)] = True
ids = np.flatnonzero(mask)
ids = np.sort(np.concatenate([ids, np.flatnonzero(~mask)[: (-ids.size) % 64]]))
np.save(sys.argv[2], ids.astype(np.int32))
print(f"{ids.size} of {n} ids -> {sys.argv[2]}")
