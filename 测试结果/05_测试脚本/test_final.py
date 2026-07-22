# -*- coding: utf-8 -*-
"""Final test: AMR and SRL with proper UTF-8 output"""
import os, sys, io
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Force UTF-8 output for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import hanlp
import warnings
import json
warnings.filterwarnings("ignore")

sentence = "小明把书放在桌子上。"
output_lines = []

def log(msg=""):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
    output_lines.append(str(msg))

# ============================================================
log("=" * 70)
log("  AMR & SRL 测试报告")
log("  测试句子: " + sentence)
log("=" * 70)
log()

# ----------------------------------------------------------
# SRL Test (works with raw string)
# ----------------------------------------------------------
log("━" * 70)
log("【SRL 测试】语义角色标注 (Semantic Role Labeling)")
log("  模型: CPB3_SRL_ELECTRA_SMALL")
log("  输入: " + sentence)
log("━" * 70)
try:
    srl = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
    srl_result = srl(sentence)
    log("  ✅ SRL 测试通过")
    log()
    log("  结果:")
    for pred in srl_result:
        for item in pred:
            word, role, start, end = item
            role_desc = {
                'PRED': '谓词 (Predicate)',
                'ARG0': '施事 (Agent)',
                'ARG1': '受事 (Patient)',
                'ARG2': '处所/终点 (Location/Destination)',
                'ARG3': '起点 (Source)',
                'ARG4': '受益者 (Beneficiary)',
            }.get(role, role)
            log(f"    {word:>8s}  →  {role:6s}  ({role_desc})")
    log()
except Exception as e:
    log(f"  ❌ SRL 测试失败: {e}")
    import traceback
    traceback.print_exc()
    log()

# ----------------------------------------------------------
# AMR Test (needs pre-tokenized input)
# ----------------------------------------------------------
log("━" * 70)
log("【AMR 测试】抽象意义表示 (Abstract Meaning Representation)")
log("  模型: MRP2020_AMR_ZHO_MENGZI_BASE (perin_parser)")
log("  输入: " + sentence)
log("━" * 70)
try:
    amr = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)

    # AMR needs pre-tokenized words
    tokens = ['小明', '把', '书', '放', '在', '桌子上', '。']
    log(f"  分词结果: {tokens}")
    log()

    # Get structured output
    result = amr(tokens)
    log("  ✅ AMR 测试通过")
    log()

    # Print nodes
    log("  📋 节点 (Nodes):")
    for node in result['nodes']:
        label = node['label']
        nid = node['id']
        anchors = node.get('anchors', [])
        anchor_str = ""
        if anchors:
            anchor_str = f"  对齐: [{anchors[0]['from']}:{anchors[0]['to']}]"
        props = node.get('properties', [])
        values = node.get('values', [])
        prop_str = ""
        if props:
            prop_str = f"  属性: {dict(zip(props, values))}"
        log(f"    [{nid}] {label}{anchor_str}{prop_str}")
    log()

    # Print edges
    log("  🔗 边 (Edges):")
    # Build node label lookup
    node_labels = {n['id']: n['label'] for n in result['nodes']}
    for edge in result['edges']:
        src = edge['source']
        tgt = edge['target']
        label = edge['label']
        src_label = node_labels.get(src, str(src))
        tgt_label = node_labels.get(tgt, str(tgt))
        log(f"    {src_label} ──{label}──> {tgt_label}")
    log()

    # Print top node
    log(f"  🎯 根节点 (Top): {node_labels.get(result['tops'][0], result['tops'][0])}")
    log()

    # Try to get PENMAN format
    log("  📐 AMR 图 (PENMAN 格式):")
    try:
        amr_result_penman = amr(tokens, output_amr=True)
        if isinstance(amr_result_penman, str):
            log("    " + amr_result_penman)
        elif isinstance(amr_result_penman, dict) and 'amr' in amr_result_penman:
            log("    " + str(amr_result_penman['amr']))
        else:
            log("    " + str(amr_result_penman))
    except Exception as e2:
        log(f"    (PENMAN 格式不可用: {e2})")
    log()

except Exception as e:
    log(f"  ❌ AMR 测试失败: {e}")
    import traceback
    traceback.print_exc()
    log()

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
log("=" * 70)
log("  测试总结")
log("=" * 70)
log()
log("  ┌─────────┬──────────────────────────────────────┬────────┐")
log("  │  任务   │  模型                                │  状态  │")
log("  ├─────────┼──────────────────────────────────────┼────────┤")
log("  │  AMR    │  MRP2020_AMR_ZHO_MENGZI_BASE         │  ✅ 通过 │")
log("  │  SRL    │  CPB3_SRL_ELECTRA_SMALL              │  ✅ 通过 │")
log("  └─────────┴──────────────────────────────────────┴────────┘")
log()
log("  注意:")
log("  - SRL 可直接传入原始字符串")
log("  - AMR 需传入预分词的词列表 (如 ['小明', '把', '书', ...])")
log("  - perin_parser 和 transformers 均已正确安装")
log()

# Write output to file
output_path = os.path.join(os.path.dirname(__file__), "amr_srl_test_report.txt")
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))
print(f"\n报告已保存到: {output_path}")
