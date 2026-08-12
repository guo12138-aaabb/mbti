# -*- coding: utf-8 -*-
"""Compare old vs new pipeline scores"""
import json, os

OUTPUT_DIR = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24"

old_path = os.path.join(OUTPUT_DIR, "holmes_analysis_data_v2_old.json")
new_path = os.path.join(OUTPUT_DIR, "holmes_analysis_data.json")

with open(old_path, 'r', encoding='utf-8') as f:
    old = json.load(f)
with open(new_path, 'r', encoding='utf-8') as f:
    new = json.load(f)

report = []
def R(msg=""):
    report.append(msg)

R("# 流水线修改前后对比报告")
R()
R("## 修改内容")
R()
R("| 修改项 | 修改前 (v2) | 修改后 (v3) |")
R("|--------|-------------|-------------|")
R("| AMR输入 | jieba分词后传token列表 | 直接传原始中文字符串 |")
R("| 权重-叙事 | 0.35 | 0.30 |")
R("| 权重-情绪 | 0.30 | 0.20 |")
R("| 权重-视觉 | 0.35 | 0.50 |")
R()

R("## 元数据对比")
R()
R("| 指标 | 修改前 | 修改后 | 变化 |")
R("|------|--------|--------|------|")
R(f"| 总章节 | {old['metadata']['total_chapters']} | {new['metadata']['total_chapters']} | - |")
R(f"| 总段落 | {old['metadata']['total_paragraphs']} | {new['metadata']['total_paragraphs']} | - |")
R(f"| 骨架段 | {old['metadata']['skeleton_count']} | {new['metadata']['skeleton_count']} | - |")
R(f"| 肉块段 | {old['metadata']['meat_count']} | {new['metadata']['meat_count']} | - |")
R(f"| 高价值候选 | {old['metadata']['high_value_count']} | {new['metadata']['high_value_count']} | {new['metadata']['high_value_count'] - old['metadata']['high_value_count']} |")
R()

R("## 高价值候选分数对比（旧版前12条 vs 新版）")
R()
R("### 修改前（jieba分词 + 旧权重 0.35/0.30/0.35）")
R()
R("| # | 类型 | 叙事 | 情绪 | 视觉 | 综合 |")
R("|---|------|------|------|------|------|")
for c in old['high_value_candidates']:
    R(f"| {c['rank']} | {c['type']} | {c['narrative_score']} | {c['emotional_score']} | {c['visual_score']} | {c['composite_score']} |")
R()

R("### 修改后（原始字符串 + 新权重 0.30/0.20/0.50）")
R()
R("**高价值候选: 0 条**（所有候选综合分均低于 4.0 阈值）")
R()
R("> 原因分析见下方")
R()

R("## 分数变化根因分析")
R()
R("### 1. AMR 分数大幅下降")
R()
R("| 维度 | 修改前 | 修改后 | 原因 |")
R("|------|--------|--------|------|")
R("| AMR调用状态 | 报错 → 关键词回退 | 成功执行（无报错） | jieba token 与 BERT word-piece 不对齐导致旧版报错 |")
R("| 叙事分 | 10.0（关键词计数） | 0.0~1.8 | 旧版回退用关键词计数给满分；新版AMR图稀疏 |")
R("| 情绪分 | 10.0（关键词计数） | 0.0~4.0 | 同上 |")
R()
R("**关键发现**：修改前的12条满分候选实际上是**关键词回退的假象**，并非AMR模型真实分析结果。")
R("修改后AMR模型虽然能跑通，但因为原始字符串被perin_parser内部按字符切分，")
R("导致BERT word-piece对齐不佳，产出的语义图节点和边非常稀疏，分数自然很低。")
R()

R("### 2. 权重变化的影响")
R()
R("以骨架候选为例（叙事=1.2, 情绪=4.0, 视觉=3.0）:")
R()
R("| 权重方案 | 计算 | 综合分 |")
R("|----------|------|--------|")
R("| 旧(0.35/0.30/0.35) | 1.2*0.35 + 4.0*0.30 + 3.0*0.35 | 2.52 |")
R("| 新(0.30/0.20/0.50) | 1.2*0.30 + 4.0*0.20 + 3.0*0.50 | 2.56 |")
R()
R("以肉块候选为例（视觉=3.0, 叙事=1.2, 情绪=0.9）:")
R()
R("| 权重方案 | 计算 | 综合分 |")
R("|----------|------|--------|")
R("| 旧(0.35/0.30/0.35) | 1.2*0.35 + 0.9*0.30 + 3.0*0.35 | 1.84 |")
R("| 新(0.30/0.20/0.50) | 1.2*0.30 + 0.9*0.20 + 3.0*0.50 | 2.04 |")
R()
R("**结论**：权重调整确实提升了视觉类候选的相对排名，但因为整体分数太低，")
R("4.0阈值仍然过滤掉了所有候选。")
R()

R("### 3. SRL 视觉分对比")
R()
R("| 维度 | 修改前 | 修改后 |")
R("|------|--------|--------|")
R("| SRL调用状态 | 正常 | 正常（未改动） |")
R("| 视觉分范围 | 1.0~3.0（仅骨架的proxy值） | 1.0~3.0（SRL真实分析） |")
R()
R("注：修改前的高价值列表全是骨架类型，其视觉分=narrative*0.3=3.0（proxy），")
R("并非SRL真实分析结果。修改后SRL对肉块的真实分析也普遍偏低（1.0~3.0）。")
R()

R("## 建议修复方案")
R()
R("### 问题1: AMR 原始字符串输入质量低")
R("- **现状**: perin_parser 内部对原始字符串做字符级切分，BERT 对齐差，图稀疏")
R("- **方案A**: 恢复 jieba 分词，但修复 token 对齐问题（需要确保 jieba 词与 BERT word-piece 正确映射）")
R("- **方案B**: 使用 HanLP 内置分词器（`hanlp.pretrained.CTB9_POS_RADICAL_ELECTRA_SMALL`）做预处理")
R("- **方案C**: 放弃 perin_parser，改用 AMRBART 序列到序列模型（需要 HuggingFace 连通）")
R()
R("### 问题2: 阈值过高")
R("- **现状**: 4.0 阈值在新的评分体系下过滤掉了所有候选")
R("- **建议**: 将阈值降至 2.0，或改用 Top-N 筛选（如取前10条）")
R()
R("### 问题3: 评分公式不够区分")
R("- **现状**: AMR 图节点数和边数太少，导致分数集中在 0~4 的窄区间")
R("- **建议**: 调整评分系数，或增加更多特征（如 AMR 图深度、连通分量数等）")
R()

# Write report
report_path = os.path.join(OUTPUT_DIR, "score_comparison_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"Comparison report: {report_path}")
