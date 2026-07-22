# -*- coding: utf-8 -*-
"""Test AMR and SRL with HanLP + perin_parser"""
import hanlp
import warnings
warnings.filterwarnings("ignore")

sentence = "小明把书放在桌子上。"

print("=" * 60)
print("Testing AMR...")
print("=" * 60)
try:
    amr = hanlp.load(hanlp.pretrained.amr.AMR3_GRAPH_PRETRAIN_PARSER)
    result = amr(sentence)
    print(f"Input: {sentence}")
    print(f"AMR Result: {result}")
    print("AMR: OK")
except Exception as e:
    print(f"AMR Error: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 60)
print("Testing SRL...")
print("=" * 60)
try:
    srl = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
    result = srl(sentence)
    print(f"Input: {sentence}")
    print(f"SRL Result: {result}")
    print("SRL: OK")
except Exception as e:
    print(f"SRL Error: {e}")
    import traceback
    traceback.print_exc()
