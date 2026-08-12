# -*- coding: utf-8 -*-
"""Test AMR with perin_parser-based MRP2020 model"""
import os
# Try HF mirror in case we need it
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import hanlp
import warnings
warnings.filterwarnings("ignore")

sentence = "小明把书放在桌子上。"

print("=" * 60)
print("Testing AMR (MRP2020_AMR_ZHO_MENGZI_BASE)...")
print("=" * 60)
try:
    amr = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
    result = amr(sentence)
    print(f"Input: {sentence}")
    print(f"AMR Result: {result}")
    print("AMR: OK")
except Exception as e:
    print(f"AMR Error: {e}")
    import traceback
    traceback.print_exc()
