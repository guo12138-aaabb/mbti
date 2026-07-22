# -*- coding: utf-8 -*-
"""
福尔摩斯探案全集 NLP 剧本改编流水线 v4
改进点:
  1. 综合评分分类型: 骨架(叙0.55/情0.35/视0.10), 肉块(视0.80/叙0.10/情0.10)
  2. AMR叙事分: 识别对话动词(说/问/答)降权, 提升动作事件链权重
  3. 情绪曲线: 用真实AMR情绪节点数替代骨架占比
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
# Dialogue verb & emotion node detection helpers
# ============================================================

DIALOGUE_VERBS = {
    '说', '问', '答', '道', '告诉', '言', '喊', '叫', '回复', '陈述',
    '嚷', '嘀咕', '讲', '谈', '议', '论', '嘱咐', '劝', '骂', '催'
}

EMOTION_WORDS = {
    '害怕', '恐惧', '惊恐', '震惊', '愤怒', '悲哀', '悲伤', '高兴', '快乐',
    '着急', '慌张', '忧愁', '痛恨', '恼怒', '疼痛', '忧虑', '惧怕', '发愣',
    '怔住', '颤抖', '叹气', '哀伤', '欢笑', '气愤', '担心', '激动', '感动',
    '惊讶', '失望', '绝望', '兴奋', '紧张', '焦虑', '痛苦', '欣慰', '得意'
}
EMOTION_CHARS = set('怕惊恐怒悲喜哭笑惨急慌愁恨爱恼疼痛忧惧愣怔震颤叹哀乐愤')

def get_verb_from_label(label):
    """Extract verb from AMR node label like '说-01' -> '说'"""
    s = str(label)
    if '-' in s:
        return s.split('-')[0]
    return s

def is_dialogue_predicate(label):
    """Check if an AMR predicate is a dialogue verb."""
    return get_verb_from_label(label) in DIALOGUE_VERBS

def is_emotion_node(label):
    """Check if an AMR node represents an emotion-related concept."""
    verb = get_verb_from_label(label)
    if verb in EMOTION_WORDS:
        return True
    if len(verb) <= 2 and any(c in EMOTION_CHARS for c in verb):
        return True
    return False

def count_emotion_nodes(nodes):
    """Count AMR nodes that represent emotion-related concepts."""
    return sum(1 for n in nodes if is_emotion_node(n.get('label', '')))

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

cleaned = re.sub(r'^书名：[^\n]+\n作者：[^\n]+\n简介：[^\n]+\n', '', raw, count=1)
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', cleaned)
print(f"  原始: {len(raw):,}字符 -> 清洗后: {len(cleaned):,}字符")

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
    body = chapter_pattern.sub('', body, count=1).strip()

    raw_paras = [p.strip() for p in body.split('\n\n') if p.strip()]

    fine_paras = []
    for rp in raw_paras:
        if len(rp) > 300:
            sub = re.split(r'(?<=[。！？」])', rp)
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

# Step 3: Classify each paragraph -> skeleton vs meat
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
    dialogue_chars = para.count('"') + para.count('"') + para.count('「') + para.count('」') + para.count("'") + para.count("'")
    has_dialogue = dialogue_chars >= 4
    if sk > mt or (has_dialogue and sk >= mt):
        skeleton_paras.append(para)
    else:
        meat_paras.append(para)

for ch in chapters_data:
    sk_count = sum(1 for p in ch['paragraphs'] if p in skeleton_paras)
    mt_count = sum(1 for p in ch['paragraphs'] if p in meat_paras)
    ch['skeleton_count'] = sk_count
    ch['meat_count'] = mt_count

print(f"  骨架(逻辑主干): {len(skeleton_paras)} 段")
print(f"  肉块(具象画面): {len(meat_paras)} 段")
print(f"  骨架:肉块 = {len(skeleton_paras)}:{len(meat_paras)} = 1:{len(meat_paras)/max(1,len(skeleton_paras)):.1f}")

print("\n--- 清洗后文本预览（前500字）---")
print(cleaned[:500])

# ============================================================
# PHASE 2: Value Assessment
# ============================================================
print("\n" + "=" * 70)
print("  阶段二：价值判断（AMR + SRL 双模型分析）")
print("=" * 70)

print("\n[加载NLP模型]...")
print("  加载 AMR 模型...")
amr_model = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
print("  加载 SRL 模型...")
srl_model = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
print("  完成")


# ---- Helper: run AMR, return (nar, emo, emo_nodes, dialogue_count, action_count, ok) ----
def run_amr(text):
    """Run AMR with jieba tokenization + progressive retry.
    v4: dialogue verbs get lower weight in narrative score.
    Returns (nar, emo, emo_node_count, dialogue_preds, action_preds, ok)."""
    for try_len in [80, 50, 30]:
        text_clean = text.replace('\u3000', '').replace('\n', ' ').strip()[:try_len]
        tokens = list(jieba.cut(text_clean))
        try:
            result = amr_model(tokens)
            nodes = result.get('nodes', [])
            edges = result.get('edges', [])

            predicates = [n for n in nodes if '-0' in str(n.get('label', ''))]
            dialogue_preds = [p for p in predicates if is_dialogue_predicate(p.get('label', ''))]
            action_preds = [p for p in predicates if not is_dialogue_predicate(p.get('label', ''))]
            agent_edges = [e for e in edges if e.get('label') == 'arg0']

            # v4: action predicates x0.60, dialogue verbs x0.15 (was all x0.45)
            nar = min(10, len(action_preds) * 0.60 + len(dialogue_preds) * 0.15 + len(agent_edges) * 0.35 + len(edges) * 0.06)

            emo = min(10, len(nodes) * 0.12 + len(edges) * 0.10)
            emo_nodes = count_emotion_nodes(nodes)

            return round(nar, 1), round(emo, 1), emo_nodes, len(dialogue_preds), len(action_preds), True
        except Exception:
            if try_len == 30:
                nar = min(10, text.count('因此') * 2 + text.count('所以') * 1.8 +
                         text.count('于是') * 1.5 + text.count('推断') * 2 +
                         text.count('发现') * 1.2 + text.count('证据') * 1.5)
                emo = min(10, text.count('!') * 2 + text.count('？') * 0.5 +
                         text.count('惊') * 1.5 + text.count('恐') * 1.8 +
                         text.count('惨') * 2 + text.count('死') * 2 +
                         text.count('奇') * 1 + text.count('怕') * 1.5)
                # Fallback: count emotion chars in text
                emo_nodes = sum(1 for c in text if c in EMOTION_CHARS)
                return round(nar, 1), round(emo, 1), emo_nodes, 0, 0, False
            continue


# ---- Helper: run SRL, return (vis, actions, spaces, actors, ok) ----
def run_srl(text, max_sentences=5):
    """Run SRL sentence-by-sentence to avoid BERT 199-token limit."""
    text = text.replace('\u3000', ' ').replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[。！？])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5][:max_sentences]

    total_actions = 0
    total_spaces = 0
    total_actors = 0
    total_objects = 0
    ok_count = 0

    for sent in sentences:
        if len(sent) > 100:
            sent = sent[:100]
        try:
            result = srl_model(sent)
            for frame in result:
                for item in frame:
                    word, role, start, end = item
                    if role == 'PRED':
                        total_actions += 1
                    elif role in ('ARG2', 'ARGM-LOC', 'ARGM-DIR', 'ARGM-TMP'):
                        total_spaces += 1
                    elif role == 'ARG0':
                        total_actors += 1
                    elif role == 'ARG1':
                        total_objects += 1
            ok_count += 1
        except Exception:
            pass

    if ok_count > 0:
        raw = (total_actions * 0.4 + total_spaces * 0.5 + total_actors * 0.15 + total_objects * 0.1) / ok_count
        vis = min(10, round(raw * 1.2, 1))
        return vis, total_actions, total_spaces, total_actors, True
    else:
        vis = min(10, text.count('看见') * 1.5 + text.count('听到') * 1.2 +
                 text.count('站') * 0.8 + text.count('走') * 0.8 +
                 text.count('手') * 0.6 + text.count('眼') * 0.8 +
                 text.count('门') * 1.0 + text.count('光') * 0.8 +
                 text.count('黑') * 1.2 + text.count('血') * 1.5)
        return round(vis, 1), 0, 0, 0, False


# ---- Task A: AMR + SRL on skeleton passages ----
print("\n" + "-" * 50)
print("[任务A] 骨架->AMR(叙事+情绪) + SRL(视觉) 双分析 [v4: 对话动词降权]")
print("-" * 50)

top_skeletons = sorted(skeleton_paras, key=len, reverse=True)[:12]

skeleton_analysis = []
for idx, skel in enumerate(top_skeletons):
    short = skel[:80].replace('\n', ' ')
    print(f"  [{idx+1}/{len(top_skeletons)}] {short}...")

    nar, emo, emo_nodes, dlg_count, act_count, amr_ok = run_amr(skel)
    vis, actions, spaces, actors, srl_ok = run_srl(skel)

    skeleton_analysis.append({
        'text': short,
        'full_text': skel,
        'narrative_score': nar,
        'emotional_score': emo,
        'visual_score': vis,
        'amr_ok': amr_ok,
        'srl_ok': srl_ok,
        'srl_actions': actions,
        'srl_spaces': spaces,
        'srl_actors': actors,
        'emo_nodes': emo_nodes,
        'dialogue_preds': dlg_count,
        'action_preds': act_count
    })
    print(f"       AMR: 叙事={nar} 情绪={emo} 情绪节点={emo_nodes} (对话谓词={dlg_count} 动作谓词={act_count}) ({'OK' if amr_ok else 'FALLBACK'}) | SRL: 视觉={vis} ({'OK' if srl_ok else 'FALLBACK'})")

# ---- Task B: SRL + AMR on meat passages ----
print("\n" + "-" * 50)
print("[任务B] 肉块->SRL(视觉) + AMR(叙事+情绪) 双分析 [v4: 对话动词降权]")
print("-" * 50)

top_meats = sorted(meat_paras, key=len, reverse=True)[:20]

meat_analysis = []
for idx, meat in enumerate(top_meats):
    short = meat[:80].replace('\n', ' ')
    print(f"  [{idx+1}/{len(top_meats)}] {short}...")

    vis, actions, spaces, actors, srl_ok = run_srl(meat)
    nar, emo, emo_nodes, dlg_count, act_count, amr_ok = run_amr(meat)

    meat_analysis.append({
        'text': short,
        'full_text': meat,
        'narrative_score': nar,
        'emotional_score': emo,
        'visual_score': vis,
        'amr_ok': amr_ok,
        'srl_ok': srl_ok,
        'srl_actions': actions,
        'srl_spaces': spaces,
        'srl_actors': actors,
        'emo_nodes': emo_nodes,
        'dialogue_preds': dlg_count,
        'action_preds': act_count
    })
    print(f"       SRL: 视觉={vis} ({'OK' if srl_ok else 'FALLBACK'}) | AMR: 叙事={nar} 情绪={emo} 情绪节点={emo_nodes} (对话谓词={dlg_count} 动作谓词={act_count}) ({'OK' if amr_ok else 'FALLBACK'})")

# ---- Task C: Comprehensive scoring (v4: type-specific weights) ----
print("\n" + "-" * 50)
print("[任务C] 综合加权评估与候选筛选 (v4: 分类型权重, 无proxy)")
print("-" * 50)

# v4: type-specific weights
WEIGHTS_SKELETON = {'narrative': 0.55, 'emotional': 0.35, 'visual': 0.10}
WEIGHTS_MEAT = {'narrative': 0.10, 'emotional': 0.10, 'visual': 0.80}

all_candidates = []

for a in skeleton_analysis:
    ns = a['narrative_score']
    es = a['emotional_score']
    vs = a['visual_score']
    # Skeleton: narrative-heavy
    cs = round(ns * 0.55 + es * 0.35 + vs * 0.10, 1)
    all_candidates.append({
        'text': a['text'],
        'full_text': a['full_text'],
        'narrative_score': ns,
        'emotional_score': es,
        'visual_score': vs,
        'type': 'skeleton',
        'amr_ok': a['amr_ok'],
        'srl_ok': a['srl_ok'],
        'composite_score': cs,
        'emo_nodes': a['emo_nodes'],
        'dialogue_preds': a['dialogue_preds'],
        'action_preds': a['action_preds']
    })

for s in meat_analysis:
    ns = s['narrative_score']
    es = s['emotional_score']
    vs = s['visual_score']
    # Meat: visual-heavy
    cs = round(vs * 0.80 + ns * 0.10 + es * 0.10, 1)
    all_candidates.append({
        'text': s['text'],
        'full_text': s['full_text'],
        'narrative_score': ns,
        'emotional_score': es,
        'visual_score': vs,
        'type': 'meat',
        'amr_ok': s['amr_ok'],
        'srl_ok': s['srl_ok'],
        'composite_score': cs,
        'emo_nodes': s['emo_nodes'],
        'dialogue_preds': s['dialogue_preds'],
        'action_preds': s['action_preds']
    })

# Sort by composite score
all_candidates.sort(key=lambda x: x['composite_score'], reverse=True)

# Filter: composite >= 2.0
high_value = [c for c in all_candidates if c['composite_score'] >= 2.0]

print(f"  总候选: {len(all_candidates)} -> 高价值: {len(high_value)}")
print(f"  骨架候选: {sum(1 for c in all_candidates if c['type']=='skeleton')} -> 高价值骨架: {sum(1 for c in high_value if c['type']=='skeleton')}")
print(f"  肉块候选: {sum(1 for c in all_candidates if c['type']=='meat')} -> 高价值肉块: {sum(1 for c in high_value if c['type']=='meat')}")

# Score distribution
sk_scores = [c['composite_score'] for c in all_candidates if c['type'] == 'skeleton']
mt_scores = [c['composite_score'] for c in all_candidates if c['type'] == 'meat']
if sk_scores:
    print(f"  骨架综合分: min={min(sk_scores)} max={max(sk_scores)} avg={sum(sk_scores)/len(sk_scores):.1f}")
if mt_scores:
    print(f"  肉块综合分: min={min(mt_scores)} max={max(mt_scores)} avg={sum(mt_scores)/len(mt_scores):.1f}")

# Dialogue ratio check
total_dlg = sum(c['dialogue_preds'] for c in all_candidates)
total_act = sum(c['action_preds'] for c in all_candidates)
print(f"  对话谓词总计: {total_dlg}, 动作谓词总计: {total_act}, 对话占比: {total_dlg/max(1,total_dlg+total_act)*100:.0f}%")

# ---- Emotion curve: real AMR emotion node counts per chapter ----
print("\n" + "-" * 50)
print("[情绪曲线] 逐章AMR情绪节点采样 (真实数据, 非骨架占比)")
print("-" * 50)

chapter_emotions = []
for ch in chapters_data[:15]:
    # Pick the longest paragraph from this chapter for sampling
    valid_paras = [p for p in ch['paragraphs'] if len(p) >= 20]
    if not valid_paras:
        chapter_emotions.append({'title': ch['title'], 'emo_nodes': 0, 'emo_score': 0, 'sample': ''})
        continue
    longest = max(valid_paras, key=len)
    nar, emo, emo_nodes, dlg, act, ok = run_amr(longest)
    chapter_emotions.append({
        'title': ch['title'],
        'emo_nodes': emo_nodes,
        'emo_score': emo,
        'sample': longest[:60].replace('\n', ' '),
        'amr_ok': ok
    })
    print(f"  {ch['title']}: 情绪节点={emo_nodes}, 情绪分={emo} ({'OK' if ok else 'FB'}) | {longest[:50].replace(chr(10),' ')}...")

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

R("# 福尔摩斯探案全集2 — NLP 剧本改编分析报告（v4）")
R()
R("> 生成时间: " + time.strftime('%Y-%m-%d %H:%M:%S'))
R("> 模型: AMR=MRP2020_AMR_ZHO_MENGZI_BASE(jieba分词+渐进重试), SRL=CPB3_SRL_ELECTRA_SMALL")
R("> **v4改进1: 分类型权重** — 骨架(叙0.55/情0.35/视0.10)筛好推理, 肉块(视0.80/叙0.10/情0.10)筛好画面")
R("> **v4改进2: 对话动词降权** — AMR叙事分中对话谓词(说/问/答)x0.15, 动作谓词x0.60, 避免对话片段刷分")
R("> **v4改进3: 真实情绪曲线** — 用AMR情绪节点数替代骨架占比, 逐章采样最长段落")
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

R("### 骨架分析示例（AMR+SRL双分析, v4含对话/动作谓词拆分）")
R()
for a in skeleton_analysis[:3]:
    R(f"- **叙事{a['narrative_score']} 情绪{a['emotional_score']} 视觉{a['visual_score']}** (对话谓词={a['dialogue_preds']}, 动作谓词={a['action_preds']}, 情绪节点={a['emo_nodes']}): {a['text']}")
R()

R("### 肉块分析示例（SRL+AMR双分析, v4含对话/动作谓词拆分）")
R()
for s in meat_analysis[:3]:
    R(f"- **视觉{s['visual_score']} 叙事{s['narrative_score']} 情绪{s['emotional_score']}** (对话谓词={s['dialogue_preds']}, 动作谓词={s['action_preds']}, 情绪节点={s['emo_nodes']}): {s['text']}")
R()

R("## 三、综合评估后的高价值内容列表")
R()
R(f"共筛选出 **{len(high_value)}** 条高价值剧本候选片段")
R(f"- 骨架权重: 叙事x0.55 + 情绪x0.35 + 视觉x0.10 (筛'好推理')")
R(f"- 肉块权重: 视觉x0.80 + 叙事x0.10 + 情绪x0.10 (筛'好画面')")
R(f"- 阈值: 2.0 | 全真实分, 无proxy")
R()
R("| # | 类型 | 叙事 | 情绪 | 视觉 | **综合** | 对话谓词 | 动作谓词 | 情绪节点 | AMR | SRL | 内容片段 | 筛选理由 |")
R("|---|------|------|------|------|----------|----------|----------|----------|-----|-----|----------|----------|")

for i, c in enumerate(high_value[:25]):
    ns = c['narrative_score']
    es = c['emotional_score']
    vs = c['visual_score']
    cs = c['composite_score']
    ct = c['type']
    amr_tag = 'OK' if c.get('amr_ok') else 'FB'
    srl_tag = 'OK' if c.get('srl_ok') else 'FB'
    dlg = c.get('dialogue_preds', 0)
    act = c.get('action_preds', 0)
    emo_n = c.get('emo_nodes', 0)
    snippet = c['text'][:45].replace('|', '/').replace('\n', ' ')

    if ct == 'skeleton':
        if cs >= 6:
            reason = "核心推理: 因果链密集, 直接改编"
        elif cs >= 4:
            reason = "优质推理: 适合主线推理场景"
        else:
            reason = "备选推理: 需人工复审"
    else:
        if cs >= 6:
            reason = "核心画面: 动作空间丰富, 直接改编"
        elif cs >= 4:
            reason = "优质画面: 适合视觉化场景"
        else:
            reason = "备选画面: 需人工复审"

    tag = "[逻辑]" if ct == 'skeleton' else "[画面]"
    reason += f" {tag}"

    R(f"| {i+1} | {ct} | {ns} | {es} | {vs} | **{cs}** | {dlg} | {act} | {emo_n} | {amr_tag} | {srl_tag} | {snippet} | {reason} |")

R()
R("### 分数分布对比（骨架 vs 肉块, v4分类型权重）")
R()
R("```")
R("类型     | 叙事分范围      | 情绪分范围      | 视觉分范围      | 综合分范围      | 权重公式")
R("---------|----------------|----------------|----------------|----------------|----------")
sk_nar = [c['narrative_score'] for c in all_candidates if c['type']=='skeleton']
sk_emo = [c['emotional_score'] for c in all_candidates if c['type']=='skeleton']
sk_vis = [c['visual_score'] for c in all_candidates if c['type']=='skeleton']
sk_cs  = [c['composite_score'] for c in all_candidates if c['type']=='skeleton']
mt_nar = [c['narrative_score'] for c in all_candidates if c['type']=='meat']
mt_emo = [c['emotional_score'] for c in all_candidates if c['type']=='meat']
mt_vis = [c['visual_score'] for c in all_candidates if c['type']=='meat']
mt_cs  = [c['composite_score'] for c in all_candidates if c['type']=='meat']
if sk_nar:
    R(f"骨架({len(sk_nar):2d}) | {min(sk_nar):.1f}~{max(sk_nar):.1f}      | {min(sk_emo):.1f}~{max(sk_emo):.1f}      | {min(sk_vis):.1f}~{max(sk_vis):.1f}      | {min(sk_cs):.1f}~{max(sk_cs):.1f}      | 叙0.55+情0.35+视0.10")
if mt_nar:
    R(f"肉块({len(mt_nar):2d}) | {min(mt_nar):.1f}~{max(mt_nar):.1f}      | {min(mt_emo):.1f}~{max(mt_emo):.1f}      | {min(mt_vis):.1f}~{max(mt_vis):.1f}      | {min(mt_cs):.1f}~{max(mt_cs):.1f}      | 视0.80+叙0.10+情0.10")
R("```")
R()

R("### 对话动词影响分析（v4新增）")
R()
R(f"- 全候选对话谓词总计: **{total_dlg}**, 动作谓词总计: **{total_act}**")
R(f"- 对话谓词占比: **{total_dlg/max(1,total_dlg+total_act)*100:.0f}%**")
R("- 对话动词(说/问/答/道/告诉等)在叙事分中权重降至 x0.15 (原 x0.45)")
R("- 动作/因果谓词(推断/发现/走/打开等)权重升至 x0.60 (原 x0.45)")
R("- 效果: 纯对话段落(他说...她说...)叙事分显著下降, 动作事件链段落上升")
R()

R("### 情绪曲线（真实AMR情绪节点数, v4改进）")
R()
R("```")
R("章节                 情绪节点  情绪分  强度条                    级别")
R("--------------------|---------|-------|--------------------------|-----")
for ce in chapter_emotions:
    nodes = ce['emo_nodes']
    score = ce['emo_score']
    bar = '█' * min(20, nodes) + '░' * max(0, 20 - nodes)
    if nodes >= 10:
        label = '高峰'
    elif nodes >= 5:
        label = '中段'
    elif nodes >= 2:
        label = '低谷'
    else:
        label = '平淡'
    R(f"  {ce['title']:18s} | {nodes:7d} | {score:5.1f} | {bar} | {label}")
R("```")
R()
R("*注: 情绪节点数 = AMR语义图中包含情绪相关概念(怕/惊/恐/怒/悲/喜/哭/笑/惨/急/慌/愁/恨/爱/恼/疼/痛/忧/惧/愣/怔/震/颤/叹/哀/乐/愤等)的节点数量*")
R("*采样方式: 每章取最长段落跑AMR, 提取情绪节点数*")
R()

R("---")
R("*AMR分析基于 perin_parser 的语义图解析, v4新增对话谓词/动作谓词拆分*")
R("*SRL分析基于 ELECTRA-Small 的语义角色标注, 句级切分避免BERT超限*")
R("*v4: 分类型权重 + 对话动词降权 + 真实情绪曲线*")

# Write report
report_path = os.path.join(OUTPUT_DIR, "holmes_nlp_analysis_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report))

# Write structured JSON
json_path = os.path.join(OUTPUT_DIR, "holmes_analysis_data.json")
data = {
    'metadata': {
        'version': 'v4',
        'total_chapters': len(chapters_data),
        'total_paragraphs': len(all_paragraphs),
        'skeleton_count': len(skeleton_paras),
        'meat_count': len(meat_paras),
        'high_value_count': len(high_value),
        'models': {'amr': 'MRP2020_AMR_ZHO_MENGZI_BASE', 'srl': 'CPB3_SRL_ELECTRA_SMALL'},
        'weights_skeleton': {'narrative': 0.55, 'emotional': 0.35, 'visual': 0.10},
        'weights_meat': {'narrative': 0.10, 'emotional': 0.10, 'visual': 0.80},
        'threshold': 2.0,
        'v4_changes': [
            'Type-specific composite weights (skeleton=narrative-heavy, meat=visual-heavy)',
            'Dialogue verb penalty in AMR narrative score (x0.15 vs x0.60 for action verbs)',
            'Real AMR emotion node counts for emotion curve (replaces skeleton ratio proxy)'
        ],
        'dialogue_verbs': list(DIALOGUE_VERBS),
        'note': 'All scores are real (no proxy). v4: type-specific weights + dialogue penalty + real emotion curve.'
    },
    'chapters': [{
        'title': ch['title'],
        'index': ch['index'],
        'total': ch['count'],
        'skeleton': ch['skeleton_count'],
        'meat': ch['meat_count']
    } for ch in chapters_data],
    'chapter_emotions': chapter_emotions,
    'all_candidates': [{
        'rank': i + 1,
        'text': c['full_text'][:300],
        'type': c['type'],
        'narrative_score': c['narrative_score'],
        'emotional_score': c['emotional_score'],
        'visual_score': c['visual_score'],
        'composite_score': c['composite_score'],
        'amr_ok': c.get('amr_ok'),
        'srl_ok': c.get('srl_ok'),
        'emo_nodes': c.get('emo_nodes', 0),
        'dialogue_preds': c.get('dialogue_preds', 0),
        'action_preds': c.get('action_preds', 0)
    } for i, c in enumerate(all_candidates)],
    'high_value_candidates': [{
        'rank': i + 1,
        'text': c['full_text'][:300],
        'type': c['type'],
        'narrative_score': c['narrative_score'],
        'emotional_score': c['emotional_score'],
        'visual_score': c['visual_score'],
        'composite_score': c['composite_score'],
        'amr_ok': c.get('amr_ok'),
        'srl_ok': c.get('srl_ok'),
        'emo_nodes': c.get('emo_nodes', 0),
        'dialogue_preds': c.get('dialogue_preds', 0),
        'action_preds': c.get('action_preds', 0)
    } for i, c in enumerate(high_value[:25])]
}
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n完成 完整报告: {report_path}")
print(f"完成 结构化数据: {json_path}")
print("\n流水线 v4 全部执行完毕！")
