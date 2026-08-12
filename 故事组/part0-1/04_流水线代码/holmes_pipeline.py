# -*- coding: utf-8 -*-
"""
福尔摩斯探案全集 NLP 剧本改编流水线
Phase 1: 预处理（清洗、切分、骨架/肉块提取）
Phase 2: 价值判断（AMR叙事/情绪分析 + SRL视觉分析 + 综合评分）
"""
import os, sys, io, re, json, time

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import hanlp
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Config
# ============================================================
INPUT_FILE = r"C:\Users\genji\OneDrive\Desktop\故事组\福尔摩斯探案全集2((英)阿瑟·柯南道尔).txt"
OUTPUT_DIR = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24"
SAMPLE_CHAPTERS = range(1, 6)  # Process first 5 chapters for deep analysis

# ============================================================
# Phase 1: Pre-processing
# ============================================================
print("=" * 70)
print("  阶段一：预处理（清洗与切分）")
print("=" * 70)

# Step 1: Read and clean
print("\n[步骤1] 读取并清洗原始文本...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()

print(f"  原始大小: {len(raw):,} 字符, {raw.count(chr(10)):,} 行")

# Cleaning
cleaned = raw
# Remove header metadata
cleaned = re.sub(r'^书名：[^\n]+\n作者：[^\n]+\n简介：[^\n]+\n', '', cleaned, count=1)
# Normalize whitespace (但保留段落首行缩进和章节标题)
cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)  # 多个空格/制表符 -> 单空格
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # 多个空行 -> 双空行
cleaned = re.sub(r'　{2,}', '　', cleaned)  # 全角空格去重
# Remove non-text artifacts
cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', cleaned)  # 控制字符
# Normalize Chinese punctuation
cleaned = cleaned.replace('??', '——')

print(f"  清洗后大小: {len(cleaned):,} 字符")

# Step 2: Split into chapters
print("\n[步骤2] 按章节切分...")
chapter_pattern = re.compile(r'(第\d+章\s+\S+)')
splits = list(chapter_pattern.finditer(cleaned))
chapters = []
for i, m in enumerate(splits):
    title = m.group(1)
    start = m.start()
    end = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
    body = cleaned[start:end].strip()
    chapters.append({'title': title, 'body': body, 'index': i + 1})

print(f"  共切分出 {len(chapters)} 个章节")
for ch in chapters[:5]:
    print(f"    第{ch['index']}章: {ch['title']} ({len(ch['body']):,}字)")

# Step 3: Extract skeleton and meat
print("\n[步骤3] 提取骨架与肉块...")

def split_paragraphs(text):
    """Split chapter body into paragraphs"""
    # Split by line breaks, keep non-empty
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    # Group lines that don't start with 全角缩进 as continuation
    paragraphs = []
    buf = []
    for line in lines:
        # Skip chapter titles in body
        if re.match(r'^第\d+章', line):
            if buf:
                paragraphs.append(''.join(buf))
                buf = []
            continue
        if line.startswith('　'):
            if buf:
                paragraphs.append(''.join(buf))
            buf = [line]
        else:
            buf.append(line)
    if buf:
        paragraphs.append(''.join(buf))
    return paragraphs

def classify_paragraph(para):
    """
    Classify paragraph as 'skeleton' (narrative logic) or 'meat' (scene description)
    
    Skeleton indicators: reasoning, deduction, dialogue about facts,
    cause-effect chains, emotional turning points
    
    Meat indicators: physical actions, scene descriptions, spatial transitions,
    visual details, character movements
    """
    # Heuristic-based classification
    skeleton_keywords = [
        '因此', '所以', '因为', '于是', '显然', '看来', '结论', '推断',
        '分析', '判断', '推理', '证据', '破案', '线索', '逻辑',
        '我明白了', '我发现了', '这说明', '意味着'
    ]
    meat_keywords = [
        '看见', '听到', '走到', '站起', '坐下', '打开', '关上',
        '手中', '眼睛', '脸色', '房间里', '门外', '窗前', '街上',
        '身上', '地上', '光', '暗', '黑', '白', '血', '刀'
    ]
    
    sk_score = sum(1 for kw in skeleton_keywords if kw in para)
    mt_score = sum(1 for kw in meat_keywords if kw in para)
    
    # Also check for dialogue patterns (often skeleton)
    dialogue_count = para.count('"') + para.count('"') + para.count('「') + para.count('」')
    dialogue_count += para.count("'") + para.count("'")
    
    if sk_score > mt_score or dialogue_count > 4:
        return 'skeleton'
    else:
        return 'meat'

all_skeletons = []
all_meats = []
chapter_analysis = []

for idx, ch in enumerate(chapters):
    title = ch['title']
    body = ch['body']
    paragraphs = split_paragraphs(body)
    
    skeletons = []
    meats = []
    
    for para in paragraphs:
        if len(para) < 10:
            continue
        category = classify_paragraph(para)
        if category == 'skeleton':
            skeletons.append(para)
        else:
            meats.append(para)
    
    all_skeletons.extend(skeletons)
    all_meats.extend(meats)
    
    chapter_analysis.append({
        'title': title,
        'index': idx + 1,
        'total_paras': len(paragraphs),
        'skeleton_count': len(skeletons),
        'meat_count': len(meats),
        'skeleton_text': ' '.join(skeletons[:3]),  # Store first 3 for reference
        'meat_text': ' '.join(meats[:3])
    })

print(f"  总计: {len(all_skeletons)} 条骨架, {len(all_meats)} 条肉块")

# Show the preview
print("\n--- 清洗后文本预览（前500字）---")
preview = cleaned[:500]
print(preview)

# ============================================================
# Phase 2: Value Assessment
# ============================================================
print("\n" + "=" * 70)
print("  阶段二：价值判断（三类分析）")
print("=" * 70)

# Load models
print("\n[加载模型]...")
print("  加载 AMR 模型 (MRP2020_AMR_ZHO_MENGZI_BASE)...")
amr_model = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
print("  加载 SRL 模型 (CPB3_SRL_ELECTRA_SMALL)...")
srl_model = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
print("  模型加载完毕")

# Task A: AMR analysis on skeleton passages
print("\n" + "-" * 50)
print("[任务A] AMR 骨架叙事+情绪分析")
print("-" * 50)

# Select top skeleton passages (longer = more narrative substance)
selected_skeletons = sorted(all_skeletons, key=len, reverse=True)[:15]

amr_results = []
for i, skel in enumerate(selected_skeletons):
    print(f"\n  骨架片段 {i+1}/{len(selected_skeletons)}: {skel[:60]}...")
    
    # Truncate to manageable size for AMR
    text_for_amr = skel[:200]
    
    # Tokenize (simple approach: use Chinese characters + common words)
    # For AMR we need word-level tokens
    tokens = list(text_for_amr.replace('　', '').replace(' ', ''))
    
    try:
        result = amr_model(tokens)
        # Extract narrative value indicators from AMR
        nodes = result.get('nodes', [])
        edges = result.get('edges', [])
        
        # Count predicates (actions), agents, patients
        predicates = [n for n in nodes if '-0' in n.get('label', '')]
        agents = sum(1 for e in edges if e.get('label') == 'arg0')
        locations = sum(1 for e in edges if e.get('label', '').startswith('arg'))
        
        narrative_score = min(10, len(predicates) * 1.5 + agents * 0.5)
        emotional_score = min(10, len(nodes) * 0.3 + len(edges) * 0.3)
        
        amr_results.append({
            'text': text_for_amr[:100],
            'predicate_count': len(predicates),
            'agent_count': agents,
            'relation_count': len(edges),
            'node_count': len(nodes),
            'narrative_score': round(narrative_score, 1),
            'emotional_score': round(emotional_score, 1)
        })
    except Exception as e:
        print(f"    AMR 分析失败: {e}")
        # Fallback heuristic scoring
        narrative_score = min(10, text_for_amr.count('因为') * 2 + text_for_amr.count('所以') * 2 + 
                             text_for_amr.count('因此') * 2 + text_for_amr.count('于是') * 1.5)
        emotional_score = min(10, text_for_amr.count('!') * 2 + text_for_amr.count('惊') * 1.5 +
                            text_for_amr.count('怒') * 1.5 + text_for_amr.count('恐') * 2 +
                            text_for_amr.count('惨') * 2 + text_for_amr.count('死') * 2)
        amr_results.append({
            'text': text_for_amr[:100],
            'predicate_count': 0,
            'agent_count': 0,
            'relation_count': 0,
            'node_count': 0,
            'narrative_score': round(narrative_score, 1),
            'emotional_score': round(emotional_score, 1),
            'fallback': True
        })

# Task B: SRL analysis on meat passages
print("\n" + "-" * 50)
print("[任务B] SRL 肉块视觉分析")
print("-" * 50)

selected_meats = sorted(all_meats, key=len, reverse=True)[:15]

srl_results = []
for i, meat in enumerate(selected_meats):
    print(f"\n  肉块片段 {i+1}/{len(selected_meats)}: {meat[:60]}...")
    
    text_for_srl = meat[:300]
    
    try:
        result = srl_model(text_for_srl)
        
        # Count visual elements from SRL
        total_roles = 0
        action_count = 0
        spatial_count = 0
        agent_count = 0
        
        for frame in result:
            for item in frame:
                word, role, start, end = item
                total_roles += 1
                if role == 'PRED':
                    action_count += 1
                elif role in ('ARG2', 'ARGM-LOC', 'ARGM-DIR'):
                    spatial_count += 1
                elif role == 'ARG0':
                    agent_count += 1
        
        visual_score = min(10, action_count * 1.5 + spatial_count * 2 + agent_count * 0.5)
        
        srl_results.append({
            'text': text_for_srl[:100],
            'action_count': action_count,
            'spatial_count': spatial_count,
            'agent_count': agent_count,
            'total_roles': total_roles,
            'visual_score': round(visual_score, 1)
        })
    except Exception as e:
        print(f"    SRL 分析失败: {e}")
        # Fallback heuristic
        visual_score = min(10, text_for_srl.count('看见') * 2 + text_for_srl.count('站') * 1 + 
                          text_for_srl.count('走') * 1 + text_for_srl.count('手') * 1 +
                          text_for_srl.count('眼') * 1 + text_for_srl.count('门') * 1.5)
        srl_results.append({
            'text': text_for_srl[:100],
            'action_count': 0,
            'spatial_count': 0,
            'agent_count': 0,
            'total_roles': 0,
            'visual_score': round(visual_score, 1),
            'fallback': True
        })

# ============================================================
# Task C: Comprehensive scoring and candidate generation
# ============================================================
print("\n" + "-" * 50)
print("[任务C] 综合加权评估与候选筛选")
print("-" * 50)

# Combine AMR and SRL results into unified candidates
candidates = []

for ar in amr_results:
    candidates.append({
        'text': ar['text'],
        'narrative_score': ar['narrative_score'],
        'emotional_score': ar['emotional_score'],
        'visual_score': 3.0,  # Default for skeleton passages
        'type': 'skeleton',
        'amr_detail': ar
    })

for sr in srl_results:
    # Check if we already have this text (avoid duplicates)
    existing = [c for c in candidates if c['text'] == sr['text']]
    if not existing:
        candidates.append({
            'text': sr['text'],
            'narrative_score': 3.0,  # Default for meat passages
            'emotional_score': 2.0,
            'visual_score': sr['visual_score'],
            'type': 'meat',
            'srl_detail': sr
        })
    else:
        # Merge: this passage had both skeleton and meat characteristics
        existing[0]['visual_score'] = sr['visual_score']
        existing[0]['type'] = 'hybrid'
        existing[0]['srl_detail'] = sr

# Weighted comprehensive score
WEIGHTS = {
    'narrative': 0.35,  # 叙事价值权重
    'emotional': 0.30,  # 情绪价值权重
    'visual': 0.35      # 视觉价值权重
}

for c in candidates:
    c['composite_score'] = round(
        c['narrative_score'] * WEIGHTS['narrative'] +
        c['emotional_score'] * WEIGHTS['emotional'] +
        c['visual_score'] * WEIGHTS['visual'],
        1
    )

# Sort by composite score descending
candidates.sort(key=lambda x: x['composite_score'], reverse=True)

# Filter low quality (composite < 3.5)
high_value = [c for c in candidates if c['composite_score'] >= 3.5]

print(f"  总候选数: {len(candidates)}")
print(f"  高价值候选数: {len(high_value)}")

# ============================================================
# Output: Generate report
# ============================================================
print("\n" + "=" * 70)
print("  输出：生成分析报告")
print("=" * 70)

report_lines = []

def rprint(msg=""):
    print(msg)
    report_lines.append(str(msg))

# Part 1: Cleaned text preview
rprint("# 福尔摩斯探案全集2 — NLP 剧本改编分析报告")
rprint()
rprint("## 一、清洗后的干净文本预览（前500字）")
rprint()
rprint("```")
rprint(preview)
rprint("```")
rprint()

# Part 2: Skeleton vs Meat comparison table
rprint("## 二、骨架与肉块的切分结果")
rprint()
rprint("| 章节 | 总段落数 | 骨架(逻辑主干) | 肉块(具象画面) | 比例(骨架:肉块) |")
rprint("|------|---------|---------------|---------------|----------------|")
for ca in chapter_analysis[:10]:
    total = ca['total_paras']
    sk = ca['skeleton_count']
    mt = ca['meat_count']
    ratio = f"{sk}:{mt}" if mt > 0 else "N/A"
    rprint(f"| {ca['title']} | {total} | {sk} | {mt} | {ratio} |")
rprint()

# Show sample skeleton/meat content
rprint("### 骨架示例（逻辑/推理/对话主干）")
rprint()
for i, ca in enumerate(chapter_analysis[:3]):
    rprint(f"**{ca['title']}**: {ca['skeleton_text'][:150]}...")
    rprint()
rprint("### 肉块示例（场景/动作/空间描写）")
rprint()
for i, ca in enumerate(chapter_analysis[:3]):
    rprint(f"**{ca['title']}**: {ca['meat_text'][:150]}...")
    rprint()

# Part 3: High-value candidates
rprint("## 三、综合评估后的高价值内容列表")
rprint()
rprint("| 排名 | 类型 | 叙事分 | 情绪分 | 视觉分 | 综合分 | 内容片段 | 审核建议 |")
rprint("|------|------|--------|--------|--------|--------|----------|----------|")

for rank, c in enumerate(high_value[:20]):
    n_score = c['narrative_score']
    e_score = c['emotional_score']
    v_score = c['visual_score']
    comp = c['composite_score']
    ctype = c['type']
    snippet = c['text'][:40].replace('|', '/')
    
    # Generate recommendation
    if comp >= 7:
        rec = "⭐ 强烈推荐：核心戏剧冲突，建议直接改编为剧本片段"
    elif comp >= 5.5:
        rec = "✓ 推荐：适合作为过渡场景或次要情节"
    elif comp >= 4:
        rec = "○ 可选：可保留为背景穿插"
    else:
        rec = "△ 待定：需人工复审"
    
    if ctype == 'skeleton':
        rec += " [侧重逻辑/推理]"
    elif ctype == 'meat':
        rec += " [侧重画面/动作]"
    else:
        rec += " [逻辑+画面并重]"
    
    rprint(f"| {rank+1} | {ctype} | {n_score} | {e_score} | {v_score} | **{comp}** | {snippet}... | {rec} |")

rprint()
rprint("### 情绪曲线（基于骨架分析的叙事节奏）")
rprint()
rprint("```")
# Build emotional arc from chapter-level aggregated scores
rprint("章节情感强度变化:")
for ca in chapter_analysis[:8]:
    # Heuristic emotional intensity based on skeleton density
    emotional_density = ca['skeleton_count'] / max(1, ca['total_paras'])
    bar_len = int(emotional_density * 20)
    bar = '█' * bar_len + '░' * (20 - bar_len)
    rprint(f"  {ca['title']:20s} [{bar}] {emotional_density:.2f}")
rprint("```")

rprint()
rprint("---")
rprint(f"*报告生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
rprint(f"*AMR 模型: MRP2020_AMR_ZHO_MENGZI_BASE (perin_parser)*")
rprint(f"*SRL 模型: CPB3_SRL_ELECTRA_SMALL*")

# Write report
report_path = os.path.join(OUTPUT_DIR, "holmes_nlp_analysis_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n✅ 报告已保存到: {report_path}")

# Also save JSON data for reference
json_path = os.path.join(OUTPUT_DIR, "holmes_analysis_data.json")
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump({
        'chapters': [{
            'title': ca['title'],
            'index': ca['index'],
            'total_paras': ca['total_paras'],
            'skeleton_count': ca['skeleton_count'],
            'meat_count': ca['meat_count']
        } for ca in chapter_analysis],
        'high_value_candidates': [{
            'rank': i+1,
            'text': c['text'],
            'type': c['type'],
            'narrative_score': c['narrative_score'],
            'emotional_score': c['emotional_score'],
            'visual_score': c['visual_score'],
            'composite_score': c['composite_score']
        } for i, c in enumerate(high_value[:20])]
    }, f, ensure_ascii=False, indent=2)

print(f"✅ 结构化数据已保存到: {json_path}")
print("\n🎬 流水线全部执行完毕！")
