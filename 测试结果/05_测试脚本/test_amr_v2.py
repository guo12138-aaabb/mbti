# -*- coding: utf-8 -*-
"""Test AMR with pre-tokenized input"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import hanlp
import warnings
warnings.filterwarnings("ignore")

sentence = "小明把书放在桌子上。"

print("=" * 60)
print("Test 1: Pre-tokenized word list")
print("=" * 60)
try:
    amr = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
    # Pass as list of tokens (words)
    tokens = ['小明', '把', '书', '放', '在', '桌子上', '。']
    result = amr(tokens)
    print(f"Input tokens: {tokens}")
    print(f"AMR Result: {result}")
    print("AMR: OK")
except Exception as e:
    print(f"AMR Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Test 2: Use HanLP word segmentation first, then AMR")
print("=" * 60)
try:
    # Load a Chinese word segmenter
    tok = hanlp.load(hanlp.pretrained.ctb.CTB5_CWS_RNN)
    words = tok(sentence)
    print(f"Segmented: {words}")
    result = amr(words)
    print(f"AMR Result: {result}")
    print("AMR: OK")
except Exception as e:
    print(f"AMR Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Test 3: Pass as nested list (batch mode)")
print("=" * 60)
try:
    tokens = [['小明', '把', '书', '放', '在', '桌子上', '。']]
    result = amr(tokens)
    print(f"Input tokens: {tokens}")
    print(f"AMR Result: {result}")
    print("AMR: OK")
except Exception as e:
    print(f"AMR Error: {e}")
    import traceback
    traceback.print_exc()
