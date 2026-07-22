# -*- coding: utf-8 -*-
"""Compare v5 (all 10s) vs v6 (reduced coefficients)"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r'C:\Users\genji\WorkBuddy\2026-07-15-17-39-24'
with open(f'{base}\\holmes_analysis_data_v5_all10.json', 'r', encoding='utf-8') as f:
    old = json.load(f)
with open(f'{base}\\holmes_analysis_data.json', 'r', encoding='utf-8') as f:
    new = json.load(f)

lines = []
def P(s=''):
    lines.append(s)

P("# 评分系数调优前后对比报告")
P()
P("## 一、系数变化")
P()
P("| 公式 | 修改前系数 | 修改后系数 |")
P("|------|-----------|-----------|")
P("| AMR叙事 | pred×1.8 + agent×0.7 + edge×0.2 | pred×0.45 + agent×0.35 + edge×0.06 |")
P("| AMR情绪 | node×0.35 + edge×0.25 | node×0.12 + edge×0.10 |")
P("| SRL视觉 | (act×1.5 + spc×2.0 + actr×0.5 + obj×0.3)×2.0 | (act×0.4 + spc×0.5 + actr×0.15 + obj×0.1)×1.2 |")
P()
P("## 二、整体分数分布对比")
P()
P("| 维度 | 修改前范围 | 修改后范围 | 改善 |")
P("|------|-----------|-----------|------|")

old_nar = [c['narrative_score'] for c in old['high_value_candidates']]
old_emo = [c['emotional_score'] for c in old['high_value_candidates']]
old_vis = [c['visual_score'] for c in old['high_value_candidates']]
old_comp = [c['composite_score'] for c in old['high_value_candidates']]

new_nar = [c['narrative_score'] for c in new['high_value_candidates']]
new_emo = [c['emotional_score'] for c in new['high_value_candidates']]
new_vis = [c['visual_score'] for c in new['high_value_candidates']]
new_comp = [c['composite_score'] for c in new['high_value_candidates']]

P(f"| 叙事分 | {min(old_nar)}~{max(old_nar)} | {min(new_nar)}~{max(new_nar)} | {'有区分度' if max(new_nar) < max(old_nar) or min(new_nar) < min(old_nar) else '无变化'} |")
P(f"| 情绪分 | {min(old_emo)}~{max(old_emo)} | {min(new_emo)}~{max(new_emo)} | {'有区分度' if max(new_emo) < max(old_emo) or min(new_emo) < min(old_emo) else '无变化'} |")
P(f"| 视觉分 | {min(old_vis)}~{max(old_vis)} | {min(new_vis)}~{max(new_vis)} | {'有区分度' if max(new_vis) < max(old_vis) or min(new_vis) < min(old_vis) else '无变化'} |")
P(f"| 综合分 | {min(old_comp)}~{max(old_comp)} | {min(new_comp)}~{max(new_comp)} | {'有区分度' if max(new_comp) < max(old_comp) or min(new_comp) < min(old_comp) else '无变化'} |")
P()
P("## 三、撞顶（=10.0）数量对比")
P()
old_10s = sum(1 for c in old['high_value_candidates'] if c['narrative_score'] == 10.0)
old_10e = sum(1 for c in old['high_value_candidates'] if c['emotional_score'] == 10.0)
old_10v = sum(1 for c in old['high_value_candidates'] if c['visual_score'] == 10.0)
old_10c = sum(1 for c in old['high_value_candidates'] if c['composite_score'] == 10.0)
new_10s = sum(1 for c in new['high_value_candidates'] if c['narrative_score'] == 10.0)
new_10e = sum(1 for c in new['high_value_candidates'] if c['emotional_score'] == 10.0)
new_10v = sum(1 for c in new['high_value_candidates'] if c['visual_score'] == 10.0)
new_10c = sum(1 for c in new['high_value_candidates'] if c['composite_score'] == 10.0)
total = len(new['high_value_candidates'])

P(f"| 维度 | 修改前撞顶数 | 修改后撞顶数 | 总数 |")
P(f"|------|------------|------------|------|")
P(f"| 叙事=10 | {old_10s}/{total} | {new_10s}/{total} | {total} |")
P(f"| 情绪=10 | {old_10e}/{total} | {new_10e}/{total} | {total} |")
P(f"| 视觉=10 | {old_10v}/{total} | {new_10v}/{total} | {total} |")
P(f"| 综合=10 | {old_10c}/{total} | {new_10c}/{total} | {total} |")
P()
P("## 四、Top 10 候选对比（按综合分排序）")
P()
P("| # | 类型 | 修改前(叙/情/视/综合) | 修改后(叙/情/视/综合) | 变化 |")
P("|---|------|---------------------|---------------------|------|")

# Match by text prefix (first 40 chars)
old_map = {}
for c in old['high_value_candidates']:
    key = c['text'][:40]
    old_map[key] = c

for i, c in enumerate(new['high_value_candidates'][:10]):
    key = c['text'][:40]
    old_c = old_map.get(key)
    if old_c:
        old_str = f"{old_c['narrative_score']}/{old_c['emotional_score']}/{old_c['visual_score']}/{old_c['composite_score']}"
        new_str = f"{c['narrative_score']}/{c['emotional_score']}/{c['visual_score']}/{c['composite_score']}"
        delta = round(c['composite_score'] - old_c['composite_score'], 1)
        delta_str = f"{delta:+.1f}" if delta != 0 else "0.0"
    else:
        old_str = "N/A (新进入Top10)"
        new_str = f"{c['narrative_score']}/{c['emotional_score']}/{c['visual_score']}/{c['composite_score']}"
        delta_str = "NEW"
    P(f"| {i+1} | {c['type']:8s} | {old_str} | {new_str} | {delta_str} |")

P()
P("## 五、结论")
P()
P(f"- 修改前：{total}条候选的叙事/情绪/视觉分**全部撞顶10.0**，综合分全部=10.0，零区分度")
P(f"- 修改后：叙事撞顶 {new_10s}/{total}，情绪撞顶 {new_10e}/{total}，视觉撞顶 {new_10v}/{total}，综合撞顶 {new_10c}/{total}")
P(f"- 综合分范围从 {min(old_comp)}~{max(old_comp)} 扩展到 {min(new_comp)}~{max(new_comp)}")
P(f"- Top候选不再是千篇一律的满分，而是有清晰的梯度排名")

output = '\n'.join(lines)
print(output)
with open(f'{base}\\coefficient_comparison_report.md', 'w', encoding='utf-8') as f:
    f.write(output)
print("\n报告已保存: coefficient_comparison_report.md")
