# -*- coding: utf-8 -*-
"""v3 vs v4 对比报告"""
import json, os

v3_path = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24\holmes_analysis_data_v3_backup.json"
v4_path = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24\holmes_analysis_data.json"
out_path = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24\v3_v4_comparison_report.md"

with open(v3_path, 'r', encoding='utf-8') as f:
    v3 = json.load(f)
with open(v4_path, 'r', encoding='utf-8') as f:
    v4 = json.load(f)

report = []
def R(msg=""):
    report.append(msg)

R("# v3 vs v4 流水线对比报告")
R()
R("> v3: 统一权重(叙0.30/情0.20/视0.50), 无对话动词区分, 情绪曲线=骨架占比")
R("> v4: 分类型权重 + 对话动词降权 + 真实AMR情绪曲线")
R()

# Metadata comparison
R("## 一、配置变化")
R()
R("| 配置项 | v3 | v4 |")
R("|--------|----|----|")
R("| 骨架权重 | 叙0.30+情0.20+视0.50 | **叙0.55+情0.35+视0.10** |")
R("| 肉块权重 | 叙0.30+情0.20+视0.50 | **视0.80+叙0.10+情0.10** |")
R("| AMR叙事公式 | pred×0.45(统一) | **action×0.60 + dialogue×0.15** |")
R("| 情绪曲线来源 | 骨架占比(proxy) | **真实AMR情绪节点数** |")
R()

# Candidate comparison
v3_hv = v3.get('high_value_candidates', [])
v4_hv = v4.get('high_value_candidates', [])
v3_all = v3.get('all_candidates', [])
v4_all = v4.get('all_candidates', [])

R("## 二、候选数量变化")
R()
R("| 指标 | v3 | v4 | 变化 |")
R("|------|----|----|------|")
R(f"| 总候选 | {len(v3_all)} | {len(v4_all)} | {len(v4_all)-len(v3_all):+d} |")
R(f"| 高价值候选 | {len(v3_hv)} | {len(v4_hv)} | {len(v4_hv)-len(v3_hv):+d} |")

v3_sk = sum(1 for c in v3_all if c['type']=='skeleton')
v4_sk = sum(1 for c in v4_all if c['type']=='skeleton')
v3_mt = sum(1 for c in v3_all if c['type']=='meat')
v4_mt = sum(1 for c in v4_all if c['type']=='meat')
R(f"| 骨架候选 | {v3_sk} | {v4_sk} | {v4_sk-v3_sk:+d} |")
R(f"| 肉块候选 | {v3_mt} | {v4_mt} | {v4_mt-v3_mt:+d} |")

v3_sk_hv = sum(1 for c in v3_hv if c['type']=='skeleton')
v4_sk_hv = sum(1 for c in v4_hv if c['type']=='skeleton')
v3_mt_hv = sum(1 for c in v3_hv if c['type']=='meat')
v4_mt_hv = sum(1 for c in v4_hv if c['type']=='meat')
R(f"| 高价值骨架 | {v3_sk_hv} | {v4_sk_hv} | {v4_sk_hv-v3_sk_hv:+d} |")
R(f"| 高价值肉块 | {v3_mt_hv} | {v4_mt_hv} | {v4_mt_hv-v3_mt_hv:+d} |")
R()

# Score range comparison
R("## 三、分数分布变化")
R()
R("### 综合分范围")
R()
R("| 类型 | v3综合分范围 | v4综合分范围 | 变化 |")
R("|------|-------------|-------------|------|")

for typ, label in [('skeleton', '骨架'), ('meat', '肉块')]:
    v3_cs = [c['composite_score'] for c in v3_all if c['type']==typ]
    v4_cs = [c['composite_score'] for c in v4_all if c['type']==typ]
    if v3_cs and v4_cs:
        v3_range = f"{min(v3_cs):.1f}~{max(v3_cs):.1f}"
        v4_range = f"{min(v4_cs):.1f}~{max(v4_cs):.1f}"
        R(f"| {label} | {v3_range} | {v4_range} | 范围{'扩大' if (max(v4_cs)-min(v4_cs)) > (max(v3_cs)-min(v3_cs)) else '缩小'} |")

R()
R("### 三维分数范围")
R()
R("| 类型 | 维度 | v3范围 | v4范围 |")
R("|------|------|--------|--------|")
for typ, label in [('skeleton', '骨架'), ('meat', '肉块')]:
    for dim, dim_label in [('narrative_score', '叙事'), ('emotional_score', '情绪'), ('visual_score', '视觉')]:
        v3_vals = [c[dim] for c in v3_all if c['type']==typ]
        v4_vals = [c[dim] for c in v4_all if c['type']==typ]
        if v3_vals and v4_vals:
            R(f"| {label} | {dim_label} | {min(v3_vals):.1f}~{max(v3_vals):.1f} | {min(v4_vals):.1f}~{max(v4_vals):.1f} |")
R()

# Top 10 comparison
R("## 四、Top 10 候选对比")
R()
R("### v3 Top 10")
R()
R("| # | 类型 | 叙事 | 情绪 | 视觉 | 综合 | 内容 |")
R("|---|------|------|------|------|------|------|")
for i, c in enumerate(v3_all[:10]):
    snippet = c['text'][:50].replace('|', '/').replace('\n', ' ')
    R(f"| {i+1} | {c['type']} | {c['narrative_score']} | {c['emotional_score']} | {c['visual_score']} | {c['composite_score']} | {snippet} |")

R()
R("### v4 Top 10")
R()
R("| # | 类型 | 叙事 | 情绪 | 视觉 | 综合 | 对话谓词 | 动作谓词 | 情绪节点 | 内容 |")
R("|---|------|------|------|------|------|----------|----------|----------|------|")
for i, c in enumerate(v4_all[:10]):
    snippet = c['text'][:50].replace('|', '/').replace('\n', ' ')
    dlg = c.get('dialogue_preds', 0)
    act = c.get('action_preds', 0)
    emo = c.get('emo_nodes', 0)
    R(f"| {i+1} | {c['type']} | {c['narrative_score']} | {c['emotional_score']} | {c['visual_score']} | {c['composite_score']} | {dlg} | {act} | {emo} | {snippet} |")
R()

# Dialogue analysis
R("## 五、对话动词降权效果")
R()
v4_dlg = sum(c.get('dialogue_preds', 0) for c in v4_all)
v4_act = sum(c.get('action_preds', 0) for c in v4_all)
R(f"- v4 全候选对话谓词: **{v4_dlg}**, 动作谓词: **{v4_act}**")
R(f"- 对话谓词占比: **{v4_dlg/(v4_dlg+v4_act)*100:.0f}%**")
R(f"- 对话谓词权重: x0.15 (v3: x0.45), 动作谓词权重: x0.60 (v3: x0.45)")
R()

# Check if dialogue-heavy paragraphs dropped in ranking
R("### 对话密集段落排名变化")
R()
dialogue_heavy = []
for c in v4_all:
    text = c['text']
    dlg_marks = text.count('"') + text.count('"') + text.count('「') + text.count('」')
    if dlg_marks >= 8:  # dialogue-heavy
        v3_rank = next((i+1 for i, vc in enumerate(v3_all) if vc['text'][:100] == c['text'][:100]), None)
        v4_rank = next((i+1 for i, vc in enumerate(v4_all) if vc['text'][:100] == c['text'][:100]), None)
        if v3_rank and v4_rank:
            dialogue_heavy.append({
                'text': c['text'][:50].replace('\n', ' '),
                'v3_rank': v3_rank,
                'v4_rank': v4_rank,
                'change': v4_rank - v3_rank
            })

if dialogue_heavy:
    R("| 内容 | v3排名 | v4排名 | 变化 |")
    R("|------|--------|--------|------|")
    for d in sorted(dialogue_heavy, key=lambda x: x['change'])[:10]:
        arrow = f"↓{abs(d['change'])}" if d['change'] > 0 else (f"↑{abs(d['change'])}" if d['change'] < 0 else "→")
        R(f"| {d['text']} | {d['v3_rank']} | {d['v4_rank']} | {arrow} |")
R()

# Emotion curve comparison
R("## 六、情绪曲线变化")
R()
R("### v3: 骨架占比(proxy)")
R("```")
for ch in v3.get('chapters', [])[:12]:
    t = max(1, ch['total'])
    sk = ch['skeleton']
    intensity = sk / t
    bar = '█' * int(intensity * 20) + '░' * (20 - int(intensity * 20))
    R(f"  {ch['title']:18s} [{bar}] {intensity:.2f}")
R("```")
R()
R("### v4: 真实AMR情绪节点数")
R("```")
for ce in v4.get('chapter_emotions', []):
    nodes = ce['emo_nodes']
    score = ce['emo_score']
    bar = '█' * min(20, nodes) + '░' * max(0, 20 - nodes)
    R(f"  {ce['title']:18s} [{bar}] nodes={nodes:2d} score={score:4.1f}")
R("```")
R()
R("**发现**: AMR情绪节点数普遍偏低(0~1), 因为AMR是通用语义解析器, 不专门标注情绪节点。")
R("情绪分(scores 3.9~8.9)有区分度, 但情绪节点数几乎没有变化。")
R("如需更精确的情绪曲线, 建议引入专用情感分析模型(如Senta/ERNIE-Sentiment)。")
R()

R("## 七、总结")
R()
R("| 改进项 | 效果 |")
R("|--------|------|")
R("| 分类型权重 | ✅ 骨架筛'好推理'(7.0~9.2), 肉块筛'好画面'(1.7~9.5), Top1从骨架变肉块 |")
R("| 对话动词降权 | ✅ 对话谓词占比仅6%, 降权生效; 但AMR本身对话谓词识别较少, 影响有限 |")
R("| 真实情绪曲线 | ⚠️ 已用真实AMR数据替代骨架占比, 但情绪节点数普遍为0, 区分度不足 |")
R()

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

print(f"对比报告: {out_path}")
