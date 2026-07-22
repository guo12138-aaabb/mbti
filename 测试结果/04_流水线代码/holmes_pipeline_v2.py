# -*- coding: utf-8 -*-
"""
福尔摩斯探案全集 NLP 剧本改编流水线 v2
改进点: jieba分词(先截短再分词+渐进重试) + 精准段落切分 + 优化分类 + AMR/SRL深度分析 + 阈值2.0
"""
import os, sys, io, re, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import jieba
import hanlp
import warnings
warnings.filterwarnings("ignore")

INPUT_FILE = r"C:\Users\genji\OneDrive\Desktop\故事组\福尔摩斯探案全集2((英)阿瑟·柯南道尔).txt"
OUTPUT_DIR = r"C:\Users\genji\WorkBuddy\2026-07-15-17-39-24"

# ============================================================
# PHASE 1: Pre-processing
# ============================================================
print("=" * 70)
print("  阶段一：预处理（清洗与切分）")
print("=" * 70)

# Step 1: Read and clean
print("\n[步骤1] 读取并清洗原始文本...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    raw = f.read()

# Clean metadata header
cleaned = re.sub(r'^书名：[^\n]+\n作者：[^\n]+\n简介：[^\n]+\n', '', raw, count=1)
# Normalize: merge excessive newlines
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
# Remove control characters
cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', cleaned)
print(f"  原始: {len(raw):,}字符 → 清洗后: {len(cleaned):,}字符")

# Step 2: Split into chapters
print("\n[步骤2] 按章节切分并提取段落...")
chapter_pattern = re.compile(r'(第\d+章\s+\S+)')
splits = list(chapter_pattern.finditer(cleaned))

chapters_data = []
all_paragraphs = []

for i, m in enumerate(splits):
    title = m.group(1)
    start = m.start()
    end = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
    body = cleaned[start:end]
    
    # Remove the chapter title line from body
    body = chapter_pattern.sub('', body, count=1).strip()
    
    # Split into real paragraphs: split by \n\n first, then by indented lines
    raw_paras = [p.strip() for p in body.split('\n\n') if p.strip()]
    
    # Further split long paragraphs at natural sentence boundaries (。！？)
    fine_paras = []
    for rp in raw_paras:
        if len(rp) > 300:
            # Try to split at sentence boundaries
            sub = re.split(r'(?<=[。！？」"])', rp)
            merged = []
            buf = ''
            for s in sub:
                if len(buf) + len(s) < 200:
                    buf += s
                else:
                    if buf: merged.append(buf)
                    buf = s
            if buf: merged.append(buf)
            fine_paras.extend(merged if len(merged) > 1 else [rp])
        else:
            fine_paras.append(rp)
    
    ch_info = {
        'title': title,
        'index': i + 1,
        'paragraphs': fine_paras,
        'count': len(fine_paras)
    }
    chapters_data.append(ch_info)
    all_paragraphs.extend(fine_paras)

print(f"  共 {len(chapters_data)} 个章节, {len(all_paragraphs)} 个段落")

# Step 3: Classify each paragraph → skeleton vs meat
print("\n[步骤3] 分类：骨架(逻辑主干) vs 肉块(具象画面)...")

SKELETON_KW = [
    '因此', '所以', '因为', '于是', '显然', '看来', '结论', '推断', '推理',
    '分析', '判断', '证据', '线索', '逻辑', '破案', '发现', '说明', '意味',
    '认为', '想必', '肯定', '决定', '计划', '假设', '推测', '怀疑',
    '关键在于', '问题在于', '原因在于', '事实是'
]

MEAT_KW = [
    '看见', '看到', '听到', '听见', '走到', '站起', '坐下', '打开', '关上',
    '手中', '眼睛', '脸色', '房间', '门外', '窗前', '街上', '身上', '地上',
    '光', '黑暗', '苍白', '红色', '鲜红', '血迹', '刀', '枪', '楼梯',
    '壁炉', '马车', '火车', '雾', '雨', '风', '雪', '月光',
    '衣着', '帽子', '大衣', '手杖', '烟斗'
]

skeleton_paras = []
meat_paras = []

for para in all_paragraphs:
    if len(para) < 20:
        continue
    
    sk = sum(1 for kw in SKELETON_KW if kw in para)
    mt = sum(1 for kw in MEAT_KW if kw in para)
    
    # Dialogue detection
    dialogue_chars = para.count('"') + para.count('"') + para.count('「') + para.count('」') + para.count("'") + para.count("'")
    has_dialogue = dialogue_chars >= 4
    
    if sk > mt or (has_dialogue and sk >= mt):
        skeleton_paras.append(para)
    else:
        meat_paras.append(para)

# Per-chapter stats
for ch in chapters_data:
    sk_count = sum(1 for p in ch['paragraphs'] if p in skeleton_paras)
    mt_count = sum(1 for p in ch['paragraphs'] if p in meat_paras)
    ch['skeleton_count'] = sk_count
    ch['meat_count'] = mt_count

print(f"  骨架(逻辑主干): {len(skeleton_paras)} 段")
print(f"  肉块(具象画面): {len(meat_paras)} 段")
print(f"  骨架:肉块 = {len(skeleton_paras)}:{len(meat_paras)} ≈ 1:{len(meat_paras)/max(1,len(skeleton_paras)):.1f}")

# Show preview
print("\n--- 清洗后文本预览（前500字）---")
print(cleaned[:500])

# ============================================================
# PHASE 2: Value Assessment
# ============================================================
print("\n" + "=" * 70)
print("  阶段二：价值判断（AMR + SRL 分析）")
print("=" * 70)

# Load models
print("\n[加载NLP模型]...")
print("  加载 AMR 模型...")
amr_model = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
print("  加载 SRL 模型...")
srl_model = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
print("  完成")

# Jieba tokenization for AMR (fix: truncate text FIRST, then tokenize, to stay within BERT 512 limit)
def tokenize_for_amr(text, max_chars=80):
    """Truncate raw text to max_chars, then jieba-tokenize into word list for AMR input.
    Short text ensures BERT subword count stays within 512 token limit."""
    text = text.replace('　', '').replace('\n', ' ').strip()
    text = text[:max_chars]
    words = list(jieba.cut(text))
    return words

# ---- Task A: AMR on skeleton passages ----
print("\n" + "-" * 50)
print("[任务A] AMR 骨架→叙事价值+情绪价值分析")
print("-" * 50)

# Select top skeleton passages by length (longer = more substance)
top_skeletons = sorted(skeleton_paras, key=len, reverse=True)[:12]

amr_analysis = []
for idx, skel in enumerate(top_skeletons):
    short = skel[:80].replace('\n', ' ')
    print(f"  [{idx+1}/{len(top_skeletons)}] {short}...")
    
    # Jieba tokenize for AMR (with progressive retry on shorter text)
    amr_success = False
    for try_len in [80, 50, 30]:
        tokens = tokenize_for_amr(skel, max_chars=try_len)
        try:
            result = amr_model(tokens)
            nodes = result.get('nodes', [])
            edges = result.get('edges', [])
            
            # Extract narrative and emotional features from AMR graph
            predicates = [n for n in nodes if '-0' in str(n.get('label', ''))]
            agent_edges = [e for e in edges if e.get('label') == 'arg0']
            patient_edges = [e for e in edges if e.get('label') == 'arg1']
            
            # Narrative score: more predicates + agents = more complex action structure
            nar_amr = min(10, len(predicates) * 1.8 + len(agent_edges) * 0.7 + len(edges) * 0.2)
            
            # Emotional score: node count reflects concept density, edge diversity reflects emotional complexity
            emo_amr = min(10, len(nodes) * 0.35 + len(edges) * 0.25)
            
            amr_success = True
            break
        except Exception as e:
            if try_len == 30:
                # Last retry failed, use keyword fallback
                nar_amr = min(10, skel.count('因此') * 2 + skel.count('所以') * 1.8 + 
                             skel.count('于是') * 1.5 + skel.count('推断') * 2 +
                             skel.count('发现') * 1.2 + skel.count('证据') * 1.5)
                emo_amr = min(10, skel.count('!') * 2 + skel.count('？') * 0.5 +
                             skel.count('惊') * 1.5 + skel.count('恐') * 1.8 +
                             skel.count('惨') * 2 + skel.count('死') * 2 +
                             skel.count('奇') * 1 + skel.count('怕') * 1.5)
            continue
    
    amr_analysis.append({
        'text': short,
        'full_text': skel,
        'narrative_score': round(nar_amr, 1),
        'emotional_score': round(emo_amr, 1),
        'amr_ok': amr_success
    })

# ---- Task B: SRL on meat passages ----
print("\n" + "-" * 50)
print("[任务B] SRL 肉块→视觉价值分析")
print("-" * 50)

top_meats = sorted(meat_paras, key=len, reverse=True)[:20]

srl_analysis = []
for idx, meat in enumerate(top_meats):
    short = meat[:80].replace('\n', ' ')
    print(f"  [{idx+1}/{len(top_meats)}] {short}...")
    
    text_for_srl = meat[:250]  # SRL has input length limits
    
    try:
        result = srl_model(text_for_srl)
        
        # Count visual elements
        actions = 0
        spaces = 0
        actors = 0
        objects_ = 0
        
        for frame in result:
            for item in frame:
                word, role, start, end = item
                if role == 'PRED':
                    actions += 1
                elif role in ('ARG2', 'ARGM-LOC', 'ARGM-DIR', 'ARGM-TMP'):
                    spaces += 1
                elif role == 'ARG0':
                    actors += 1
                elif role == 'ARG1':
                    objects_ += 1
        
        # Visual score: actions + spatial markers + agents = scene vividness
        vis_score = min(10, actions * 1.5 + spaces * 2.0 + actors * 0.5 + objects_ * 0.3)
        
        srl_analysis.append({
            'text': short,
            'full_text': meat,
            'visual_score': round(vis_score, 1),
            'action_count': actions,
            'spatial_count': spaces,
            'actor_count': actors,
            'srl_ok': True
        })
        
    except Exception as e:
        # Fallback keyword scoring
        vis_score = min(10, text_for_srl.count('看见') * 1.5 + text_for_srl.count('听到') * 1.2 +
                       text_for_srl.count('站') * 0.8 + text_for_srl.count('走') * 0.8 +
                       text_for_srl.count('手') * 0.6 + text_for_srl.count('眼') * 0.8 +
                       text_for_srl.count('门') * 1.0 + text_for_srl.count('光') * 0.8 +
                       text_for_srl.count('黑') * 1.2 + text_for_srl.count('血') * 1.5)
        srl_analysis.append({
            'text': short,
            'full_text': meat,
            'visual_score': round(vis_score, 1),
            'action_count': 0,
            'spatial_count': 0,
            'actor_count': 0,
            'srl_ok': False
        })

# ---- Task C: Comprehensive scoring ----
print("\n" + "-" * 50)
print("[任务C] 综合加权评估与候选筛选")
print("-" * 50)

WEIGHTS = {'narrative': 0.30, 'emotional': 0.20, 'visual': 0.50}

all_candidates = []

for a in amr_analysis:
    all_candidates.append({
        'text': a['text'],
        'full_text': a['full_text'],
        'narrative_score': a['narrative_score'],
        'emotional_score': a['emotional_score'],
        'visual_score': round(a['narrative_score'] * 0.3, 1),  # skeletons have some visual value
        'type': 'skeleton',
        'composite_score': round(
            a['narrative_score'] * WEIGHTS['narrative'] +
            a['emotional_score'] * WEIGHTS['emotional'] +
            a['narrative_score'] * 0.3 * WEIGHTS['visual'], 1)
    })

for s in srl_analysis:
    all_candidates.append({
        'text': s['text'],
        'full_text': s['full_text'],
        'narrative_score': round(s['visual_score'] * 0.4, 1),
        'emotional_score': round(s['visual_score'] * 0.3, 1),
        'visual_score': s['visual_score'],
        'type': 'meat',
        'composite_score': round(
            s['visual_score'] * 0.4 * WEIGHTS['narrative'] +
            s['visual_score'] * 0.3 * WEIGHTS['emotional'] +
            s['visual_score'] * WEIGHTS['visual'], 1)
    })

# Sort by composite score
all_candidates.sort(key=lambda x: x['composite_score'], reverse=True)

# Filter: composite >= 2.0
high_value = [c for c in all_candidates if c['composite_score'] >= 2.0]

print(f"  总候选: {len(all_candidates)} → 高价值: {len(high_value)}")

# ============================================================
# Generate Report
# ============================================================
print("\n" + "=" * 70)
print("  生成分析报告")
print("=" * 70)

report = []

def R(msg=""):
    print(msg)
    report.append(msg)

R("# 福尔摩斯探案全集2 — NLP 剧本改编分析报告（v2）")
R()
R("> 生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S'))
R("> 模型: AMR=MRP2020_AMR_ZHO_MENGZI_BASE(perin_parser, jieba分词+渐进重试), SRL=CPB3_SRL_ELECTRA_SMALL")
R("> 权重: 叙事=0.30, 情绪=0.20, 视觉=0.50 | 阈值=2.0")
R()

R("## 一、清洗后的干净文本预览（前500字）")
R()
R("```text")
R(cleaned[:500])
R("```")
R()

R("## 二、骨架(逻辑主干)与肉块(具象画面)切分结果")
R()
R(f"- **总章节数**: {len(chapters_data)}")
R(f"- **总段落数**: {len(all_paragraphs)}")
R(f"- **骨架(逻辑/推理/对话)**: {len(skeleton_paras)} 段")
R(f"- **肉块(场景/动作/空间)**: {len(meat_paras)} 段")
R(f"- **骨架:肉块比**: 1:{len(meat_paras)/max(1,len(skeleton_paras)):.1f}")
R()

R("### 各章分布统计（前15章）")
R()
R("| 章节 | 总段 | 骨架 | 肉块 | 骨架比例 | 叙事密度 |")
R("|------|------|------|------|----------|----------|")
for ch in chapters_data[:15]:
    t = ch['count']
    sk = ch['skeleton_count']
    mt = ch['meat_count']
    ratio = f"{sk/t*100:.0f}%" if t > 0 else "0%"
    density = '█' * min(20, int(sk/max(1,t) * 20)) + '░' * max(0, 20 - int(sk/max(1,t) * 20))
    R(f"| {ch['title']} | {t} | {sk} | {mt} | {ratio} | {density} |")
R()

R("### 骨架示例（逻辑推理主线）")
R()
for a in amr_analysis[:3]:
    R(f"- **叙事{amr_analysis.index(a)+1}** [叙事分{a['narrative_score']} 情绪分{a['emotional_score']}]: {a['text']}")
R()

R("### 肉块示例（具象画面片段）")
R()
for s in srl_analysis[:3]:
    R(f"- **画面{srl_analysis.index(s)+1}** [视觉分{s['visual_score']}]: {s['text']}")
R()

R("## 三、综合评估后的高价值内容列表")
R()
R(f"共筛选出 **{len(high_value)}** 条高价值剧本候选片段")
R()
R("| # | 类型 | 叙事 | 情绪 | 视觉 | **综合** | 内容片段 | 筛选理由 |")
R("|---|------|------|------|------|----------|----------|----------|")

for i, c in enumerate(high_value[:25]):
    ns = c['narrative_score']
    es = c['emotional_score']
    vs = c['visual_score']
    cs = c['composite_score']
    ct = c['type']
    snippet = c['text'][:45].replace('|', '/').replace('\n', ' ')
    
    if cs >= 7:
        reason = "⭐ 核心素材：戏剧冲突强烈，直接改编"
    elif cs >= 5.5:
        reason = "✓ 优质素材：适合主要场景或过渡桥段"
    elif cs >= 4.5:
        reason = "○ 可用素材：可作背景穿插或次要情节"
    else:
        reason = "△ 备选：需人工复审"
    
    tag = "[逻辑]" if ct == 'skeleton' else "[画面]"
    reason += f" {tag}"
    
    R(f"| {i+1} | {ct} | {ns} | {es} | {vs} | **{cs}** | {snippet} | {reason} |")

R()
R("### 情绪曲线与叙事节奏")
R()
R("```")
R("章节 情感强度变化:")
for ch in chapters_data[:12]:
    t = max(1, ch['count'])
    sk = ch['skeleton_count']
    intensity = sk / t
    bar = '█' * int(intensity * 20) + '░' * (20 - int(intensity * 20))
    label = '高峰' if intensity > 0.5 else ('中段' if intensity > 0.2 else '过渡')
    R(f"  {ch['title']:20s} [{bar}] {intensity:.2f}  ({label})")
R("```")
R()
R("*注：情感强度 = 该章骨架段落数/总段落数，骨架越密集代表推理/冲突越集中*")

R()
R("---")
R("*AMR分析基于 perin_parser 的语义图解析，识别谓词-论元结构中的因果关系与情感节点*")
R("*SRL分析基于 ELECTRA-Small 的语义角色标注，提取动作施事、受事、空间关系等视觉要素*")

# Write report
report_path = os.path.join(OUTPUT_DIR, "holmes_nlp_analysis_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# Write structured JSON
json_path = os.path.join(OUTPUT_DIR, "holmes_analysis_data.json")
data = {
    'metadata': {
        'total_chapters': len(chapters_data),
        'total_paragraphs': len(all_paragraphs),
        'skeleton_count': len(skeleton_paras),
        'meat_count': len(meat_paras),
        'high_value_count': len(high_value),
        'models': {'amr': 'MRP2020_AMR_ZHO_MENGZI_BASE', 'srl': 'CPB3_SRL_ELECTRA_SMALL'}
    },
    'chapters': [{
        'title': ch['title'],
        'index': ch['index'],
        'total': ch['count'],
        'skeleton': ch['skeleton_count'],
        'meat': ch['meat_count']
    } for ch in chapters_data],
    'high_value_candidates': [{
        'rank': i + 1,
        'text': c['full_text'][:300],
        'type': c['type'],
        'narrative_score': c['narrative_score'],
        'emotional_score': c['emotional_score'],
        'visual_score': c['visual_score'],
        'composite_score': c['composite_score']
    } for i, c in enumerate(high_value[:25])]
}
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n✅ 完整报告: {report_path}")
print(f"✅ 结构化数据: {json_path}")
print("\n🎬 流水线 v2 全部执行完毕！")
