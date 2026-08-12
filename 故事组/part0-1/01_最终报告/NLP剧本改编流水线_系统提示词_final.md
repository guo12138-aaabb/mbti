# NLP 小说 → 故事生成流水线（系统提示词 final）

> 最后更新: 2026-08-07 | 当前进度: Part 0 + Part 1 + Part 1.5 + Part 2 + Part 3 已完成 | 下一阶段: Part 4 故事生成

---

## 零、项目总览

### 项目目标

对任意中文小说 `.txt` 文件进行自动 NLP 分析，完成以下全流程：

| 阶段 | 内容 | 状态 |
|------|------|------|
| **Part 0 — 预处理层** | 文本清洗 → 按章切分 → 段落提取 → 骨架/肉块分类 | ✅ 已完成 |
| **Part 1 — 价值判断层** | AMR 叙事/情绪分析 + SRL 视觉分析 → 综合评分 → 高价值候选筛选 | ✅ 已完成 |
| **Part 1.5 — 段级标签层** | 全量情绪曲线 + 每个高价值候选打标签（场景类型/三维倾向/改编优先级） | ✅ 已完成 |
| **Part 2 — 切分处理（信息提取）** | 从 25 条候选片段中提取结构化"信息零部件"（骨架提取逻辑/情绪，肉块提取视觉/场景） | ⬜ 待做 |
| **Part 3 — 可复用库建立** | 将 Part 2 提取物分类归档为三库：连接逻辑库 / 最小块库 / 文本元库 | ⬜ 待做 |
| **Part 4 — 故事生成** | 从可复用库调取信息，生成完整、可阅读、可编辑的候选故事文本 | ⬜ 待做 |

### 执行顺序

1. **Part 2** → 输出结构化提取结果 → **等用户确认**
2. **Part 3** → 输出三库索引 → **等用户确认**
3. **Part 4** → 输出新生成的故事文本

> 若有任何不确定的地方（如"这个段落应该归为哪种场景类型"），先暂停并告诉用户，由用户判断后再继续。

### 测试小说

- **书名**: 《福尔摩斯探案全集2》（(英)阿瑟·柯南道尔）
- **文件路径**: `C:\Users\genji\OneDrive\Desktop\故事组\福尔摩斯探案全集2((英)阿瑟·柯南道尔).txt`
- **文件大小**: 约 802 KB（约 27 万字）
- **编码**: UTF-8
- **结构**: 68 章，预处理后切分为 1551 个段落（骨架 526 段 / 肉块 1023 段）

### 产出目录

所有产出统一存放于：`C:\Users\genji\OneDrive\Desktop\故事组\产出\`

```
产出/
├── 01_最终报告/          ← 核心成果（报告 + JSON 数据 + 系统提示词）
├── 02_对比报告/          ← 系数调优 + 版本分数对比
├── 03_模型测试/          ← AMR/SRL 初始验证
├── 04_流水线代码/        ← pipeline v1→v2→v3→v4
├── 05_测试脚本/          ← 模型测试脚本
├── 06_辅助脚本/          ← 对比生成脚本
├── 07_历史备份/          ← 旧版数据存档
└── README.md             ← 目录索引
```

---

## 一、运行环境（换电脑必读）

### 1.1 Python 环境

使用 WorkBuddy 托管的 Python 3.10.11 + venv，所有依赖安装在隔离环境中。

**Python 可执行文件路径**（用绝对路径）：
```
C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe
```

### 1.2 需要安装的 Python 包

以下是 `pip list` 的**完整清单**。换新电脑后必须安装这些包才能运行流水线：

#### 核心依赖（必须安装）

| 包名 | 版本 | 用途 |
|------|------|------|
| `torch` | 2.13.0 | PyTorch，所有神经网络的底层计算引擎 |
| `transformers` | 4.30.0 | HuggingFace 模型库，ELECTRA 和 BERT 的运行框架 |
| `hanlp` | 2.1.3 | NLP 工具包，提供 AMR 和 SRL 的模型加载与调用 |
| `perin-parser` | 0.0.19 | AMR 图解析引擎，hanlp 内部调用 |
| `jieba` | 0.42.1 | 中文分词，将连续文本切分为词序列 |

#### 全部已安装包（完整清单，pip freeze 格式）

```
annotated-doc==0.0.4
anyio==4.14.1
certifi==2026.6.17
charset-normalizer==3.4.9
click==8.4.2
colorama==0.4.6
exceptiongroup==1.3.1
filelock==3.29.4
fsspec==2026.6.0
h11==0.16.0
hanlp==2.1.3
hanlp-common==0.0.23
hanlp-downloader==0.0.25
hanlp-trie==0.0.5
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==0.36.2
idna==3.18
jieba==0.42.1
Jinja2==3.1.6
markdown-it-py==4.2.0
MarkupSafe==3.0.3
mdurl==0.1.2
mpmath==1.3.0
networkx==3.4.2
numpy==2.2.6
nvidia-ml-py==13.610.43
packaging==26.2
Penman==1.3.1
perin-parser==0.0.19
phrasetree==0.0.9
Pygments==2.20.0
pynvml==13.0.1
PyYAML==6.0.3
regex==2026.5.9
requests==2.34.2
rich==15.0.0
safetensors==0.8.0
scipy==1.15.3
sentencepiece==0.2.2
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
termcolor==3.3.0
tokenizers==0.13.3
toposort==1.5
tqdm==4.68.3
typer==0.25.1
typing_extensions==4.15.0
urllib3==2.7.0
```

> 注意：`torch` 需要根据新电脑的 CUDA 版本选择对应版本。当前使用 CPU 版（无 CUDA），推理速度较慢但兼容性好。如果有 NVIDIA GPU 且 CUDA >= 11.7，建议安装 `torch==2.0.1+cu117` 加速。

#### 快速安装命令

```bash
# 核心5个包（最小安装）
pip install torch==2.13.0 transformers==4.30.0 hanlp==2.1.3 perin-parser==0.0.19 jieba==0.42.1

# 或一键安装全部（用 requirements.txt）
# 将上面的完整清单保存为 requirements.txt，然后：
pip install -r requirements.txt
```

### 1.3 模型文件（需要提前下载并缓存）

两个 NLP 模型**不会自动下载**（因为 huggingface.co 被墙），需要从当前电脑复制到新电脑的对应路径。

#### 模型 1：AMR 语义图解析

| 属性 | 值 |
|------|-----|
| HanLP 标识符 | `hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE` |
| 底层框架 | perin_parser (PyTorch) + Mengzi BERT |
| 编码器参数 | 768 维 / 12 层 / 8 注意力头 |
| 本地缓存路径 | `D:\hanlp\amr\amr-zho-mengzi-base\` |
| 大小 | 约 586 MB |
| 输入要求 | jieba 分词后的词列表（不是原始字符串） |
| 最大 token 数 | Mengzi BERT 限制 512 word-piece tokens |

#### 模型 2：SRL 语义角色标注

| 属性 | 值 |
|------|-----|
| HanLP 标识符 | `hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL` |
| 底层框架 | HanLP SpanBIO 组件 + ELECTRA-small |
| 编码器参数 | 256 维 / 12 层 / 4 注意力头 |
| 本地缓存路径 | `D:\hanlp\srl\cpb3_electra_small_crf_has_transform_20220218_135910\` |
| 大小 | 约 80 MB |
| 输入要求 | 原始字符串（不需要分词） |
| 最大 token 数 | ELECTRA 硬限制 199 word-piece tokens |

#### 模型缓存目录结构

```
D:\hanlp\
├── amr\
│   ├── amr-zho-mengzi-base\             ← AMR 主模型（586 MB）
│   └── amr3_graph_pretrain_parser_20221207_153759\  ← AMR 解析器权重
└── srl\
    └── cpb3_electra_small_crf_has_transform_20220218_135910\  ← SRL 模型（80 MB）
```

**迁移方法**：
1. 将 `D:\hanlp\` 整个目录复制到新电脑的 `D:\hanlp\`
2. 如果新电脑没有 D 盘，可以放到其他盘，但需要在代码中设置 `HANLP_HOME` 环境变量指向新路径
3. 或者直接让 hanlp 在新电脑上自动下载（需要配置镜像，见 1.4）

### 1.4 网络配置（中国大陆必须设置）

huggingface.co 在中国大陆无法直连，必须在每个 Python 脚本开头设置镜像：

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
```

### 1.5 Windows 控制台编码（避免中文乱码）

Windows 终端默认 GBK 编码，打印中文会显示乱码。在每个脚本开头加入：

```python
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
```

---

## 二、模型验证命令（换电脑后首先运行）

### 2.1 快速验证：Python 环境 + 所有包

```bash
C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe -c "
import sys
print(f'Python: {sys.version}')
for pkg in ['torch', 'transformers', 'hanlp', 'perin_parser', 'jieba']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f'  {pkg}: {ver}')
    except ImportError:
        print(f'  {pkg}: MISSING!')
"
```

预期输出：
```
Python: 3.10.11
  torch: 2.13.0
  transformers: 4.30.0
  hanlp: 2.1.3
  perin_parser: OK
  jieba: 0.42.1
```

### 2.2 验证 AMR 模型加载 + 推理

```bash
C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe -c "
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import jieba, hanlp
print('Loading AMR model (~586 MB, 1-3 min)...')
amr = hanlp.load(hanlp.pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
print('AMR model loaded successfully!')

text = '小明把书放在桌子上，然后转身走出了房间。'
tokens = list(jieba.cut(text[:80]))
result = amr(tokens)
nodes = len(result.get('nodes', []))
edges = len(result.get('edges', []))
print(f'Text: {text}')
print(f'Tokens: {tokens}')
print(f'AMR graph: {nodes} nodes, {edges} edges')
print('AMR test PASSED!')
"
```

预期输出：
```
Loading AMR model (~586 MB, 1-3 min)...
AMR model loaded successfully!
Text: 小明把书放在桌子上，然后转身走出了房间。
Tokens: ['小明', '把', '书', '放', '在', '桌子', '上', '，', '然后', '转身', '走出', '了', '房间', '。']
AMR graph: XX nodes, XX edges
AMR test PASSED!
```

### 2.3 验证 SRL 模型加载 + 推理

```bash
C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe -c "
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import hanlp
print('Loading SRL model (~80 MB, ~30 sec)...')
srl = hanlp.load(hanlp.pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
print('SRL model loaded successfully!')

text = '小明把书放在桌子上，然后转身走出了房间。'
result = srl(text)
total_preds = sum(1 for frame in result for item in frame if item[1] == 'PRED')
print(f'Text: {text}')
print(f'SRL frames: {len(result)}, predicates: {total_preds}')
for frame in result:
    print(f'  {[(item[0], item[1]) for item in frame]}')
print('SRL test PASSED!')
"
```

预期输出：
```
Loading SRL model (~80 MB, ~30 sec)...
SRL model loaded successfully!
Text: 小明把书放在桌子上，然后转身走出了房间。
SRL frames: 2, predicates: 2
  [('小明', 'ARG0'), ('放', 'PRED'), ('书', 'ARG1'), ('在桌子上', 'ARGM-LOC')]
  [('小明', 'ARG0'), ('转身', 'PRED'), ('走出了', 'PRED'), ('房间', 'ARG1')]
SRL test PASSED!
```

### 2.4 综合验证（一键检测所有）

```bash
C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe -c "
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import warnings; warnings.filterwarnings('ignore')

errors = []

# 1. Check Python
print(f'[1/5] Python version: {sys.version.split()[0]}')

# 2. Check packages
print('[2/5] Checking packages...')
for pkg in ['torch', 'transformers', 'hanlp', 'perin_parser', 'jieba']:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'OK')
        print(f'  {pkg}: {ver} ✓')
    except ImportError as e:
        print(f'  {pkg}: MISSING ✗')
        errors.append(f'{pkg} not installed')

# 3. Check AMR model
print('[3/5] Loading AMR model (may take 1-3 min)...')
try:
    import jieba
    amr = __import__('hanlp').load(__import__('hanlp').pretrained.amr.MRP2020_AMR_ZHO_MENGZI_BASE)
    tokens = list(jieba.cut('测试'))
    result = amr(tokens)
    print(f'  AMR OK: {len(result.get(\"nodes\",[]))} nodes ✓')
except Exception as e:
    print(f'  AMR FAILED: {e} ✗')
    errors.append(f'AMR: {e}')

# 4. Check SRL model
print('[4/5] Loading SRL model (may take 30 sec)...')
try:
    srl = __import__('hanlp').load(__import__('hanlp').pretrained.srl.CPB3_SRL_ELECTRA_SMALL)
    result = srl('测试句子。')
    print(f'  SRL OK: {len(result)} frames ✓')
except Exception as e:
    print(f'  SRL FAILED: {e} ✗')
    errors.append(f'SRL: {e}')

# 5. Summary
print(f'[5/5] Summary: {5-len(errors)}/5 checks passed')
if errors:
    print('  Errors:')
    for e in errors:
        print(f'    - {e}')
    sys.exit(1)
else:
    print('  All checks PASSED! Environment is ready.')
"
```

预期输出：全部 5 项检查通过。如果任何一项失败，会明确报错。

---

## 三、已完成工作总结（Part 0 + Part 1 + Part 1.5）

### Part 0：预处理层

**做了什么**：
1. 读取 802KB 原始 .txt 文件（UTF-8）
2. 去除文件头元信息（书名/作者/简介行）
3. 压缩多余空行、删除控制字符
4. 按章节标题正则 `第\d+章\s+\S+` 切分为 68 章
5. 每章内用 `\n\n` 拆分段落，>300 字大段再按句号边界合并为 100-200 字小段
6. 用两套关键词词典（骨架关键词 / 肉块关键词）对每个段落分类

**成果**：
- 68 章 → 1551 个段落（骨架 526 / 肉块 1023）
- 骨架：肉块 ≈ 1:1.94

### Part 1：价值判断层

**做了什么**：
1. 骨架取最长 12 段 + 肉块取最长 20 段，跑 AMR + SRL 双模型分析
2. AMR（语义图解析）：提取叙事分 N + 情绪分 E + 情绪节点数 + 对话/动作谓词统计
3. SRL（语义角色标注）：提取视觉分 V（动作 + 空间 + 施事 + 受事）
4. 分类型综合评分：骨架侧重叙事(0.55)，肉块侧重视觉(0.80)
5. 阈值 2.0 筛选高价值候选（共 25 个）

**成果**：
- 25 个高价值候选：S 级 9 个 / A 级 9 个 / B 级 7 个
- 综合分范围：骨架 7.0~9.2 / 肉块 4.5~9.5

### Part 1.5：段级标签层（问题D修复）

**做了什么**：
1. 全量情绪曲线：68 章 × 1549 段全部跑 AMR，取情绪节点平均数（替代之前每章只取 1 段的稀疏采样）
2. 给 25 个高价值候选逐段打标签：全文级（字数/三维倾向/世界观/基调/人物/视角/来源）+ 段级（三维倾向/人物涉及/场景类型/改编优先级）

**成果**：
- 全量情绪曲线均值范围 0.0~0.56（AMR 情绪节点检测对翻译小说效果有限，全部为"平淡"级别）
- 场景分布：对话 7 / 动作暴力 6 / 动作 5 / 推理 4 / 室内 2 / 叙述 1

---

## 四、评分体系（v4 最终版）

### 4.1 综合评分公式

**骨架综合分**（侧重叙事逻辑）：
```
综合分 = N × 0.55 + E × 0.35 + V × 0.10
```

**肉块综合分**（侧重视觉画面）：
```
综合分 = V × 0.80 + N × 0.10 + E × 0.10
```

**阈值**：综合分 ≥ 2.0 → 高价值候选

### 4.2 AMR 叙事分 N

```
N = min(10, action_preds × 0.60 + dialogue_preds × 0.15 + agent_edges × 0.35 + all_edges × 0.06)
```

| 特征 | 系数 | 含义 |
|------|------|------|
| 动作谓词数 | 0.60 | 动作事件链是叙事推进的主力 |
| 对话谓词数 | 0.15 | 降权：纯对话不等于叙事推进 |
| arg0 边数 | 0.35 | 施事-动作关系密度 |
| 总边数 | 0.06 | 语义图复杂度微调项 |

**对话谓词集**：说/问/答/道/告诉/言/喊/叫/回复/陈述/嚷/嘀咕/讲/谈/议/论/嘱咐/劝/骂/催

### 4.3 AMR 情绪分 E

```
E = min(10, nodes × 0.12 + edges × 0.10)
```

> **注意**：情绪分 E 本质是 AMR 图复杂度 proxy，不是真实情绪浓度。情绪节点数（emo_nodes）才是真实情绪指标，但 AMR 不专门标注情绪，情绪节点数通常为 0-1。

### 4.4 SRL 视觉分 V

```
V = min(10, (total_actions × 0.4 + total_spaces × 0.5 + total_actors × 0.15 + total_objects × 0.1) / ok_count × 1.2)
```

| 特征 | 系数 | 含义 |
|------|------|------|
| PRED 总数 | 0.4 | 每个动作 = 一个可拍画面 |
| ARG2/LOC/DIR/TMP 总数 | 0.5 | 空间信息是场景设计关键输入 |
| ARG0 总数 | 0.15 | 人物密度 |
| ARG1 总数 | 0.1 | 道具/物体密度 |

### 4.5 AMR 输入处理

```python
# 必须 jieba 分词 + 渐进重试
for try_len in [80, 50, 30]:
    text_clean = text.replace('\u3000', '').replace('\n', ' ').strip()[:try_len]
    tokens = list(jieba.cut(text_clean))
    try:
        result = amr_model(tokens)
        break  # 成功则跳出
    except Exception:
        if try_len == 30:
            # 回退：字符级情绪统计
            emo_nodes = sum(1 for c in text if c in EMOTION_CHARS)
        continue
```

**为什么需要 jieba**：AMR 底层 perin_parser 的 `predict()` 要求输入是词级别 token 列表。如果传原始字符串，每个中文字被当成独立 token，导致 BERT word-piece 对齐失败。

**为什么渐进重试**：Mengzi BERT 有 512 token 上限。80 字中文经 jieba + BERT word-piece 约 50-80 token，通常在限额内。但含英文/数字/长实体的文本可能超限，此时依次降级到 50→30 字。30 字在任何情况下都不会超限。

### 4.6 SRL 输入处理

```python
# 句级切分 + 逐句运行（最关键！）
sentences = re.split(r'(?<=[。！？])', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 5][:5]

for sent in sentences:
    if len(sent) > 100:
        sent = sent[:100]
    try:
        result = srl_model(sent)
        # 统计每句的 PRED/ARG0/ARG1/ARG2/LOC/DIR/TMP
    except Exception:
        pass  # 个别句子失败不阻塞
```

**为什么必须句级切分**：ELECTRA-small 有硬限制 199 word-piece token。250 字中文经 BERT 子词切分可达 300-400 token，必定超限。解决方案：按句号/"！"/"？"切分，每句截取前 100 字，逐句跑 SRL 后汇总统计。

---

## 五、已知问题与注意事项

### 5.1 情绪分与情绪节点数的矛盾

**问题**：报告显示情绪节点数普遍为 0-1，但情绪分 E 却高达 6.5~9.2。

**原因**：情绪分 E 的公式 `nodes × 0.12 + edges × 0.10` 用的是**全部节点和边**，不专门针对情绪。一个推理段落即使情绪节点为 0，只要 AMR 图有 25 个节点 + 20 条边，情绪分就是 `25×0.12+20×0.10 = 5.0`。

**结论**：情绪分 E 本质是 AMR 图复杂度 proxy，不是真实情绪浓度。情绪节点数才是真实情绪指标。

### 5.2 "动作谓词"名不副实

代码中的"动作谓词"实际是"非对话谓词"——只要谓词不在对话动词集合里就算"动作谓词"。这包含了真正的动作（走/打开/发现）但也包含因果推理（推断/说明/意味）和状态/存在（有/是/在）。建议更名为"非对话谓词"或在后续 Part 2 中进一步细分。

### 5.3 SRL 句级切分的副作用

长句截断到 100 字后，后半句的动作会丢失。这是 trade-off：要么超限报错得 0 分，要么牺牲部分精度换取可用结果。

### 5.4 AMR 情绪节点检测的局限性

AMR 是通用语义解析器，不专门标注情绪。即使段落中有"恐惧""震惊"等词，AMR 可能将它们解析为普通概念节点。全量情绪曲线（1549 段）的结果也印证了这一点：均值范围仅 0.0~0.56，全部章节为"平淡"级别。如需改善情绪检测，可引入专用情感分析模型。

### 5.5 词典和采样的通用性

- 骨架/肉块关键词词典当前针对侦探推理小说，换其他类型需替换
- Top-N 采样（骨架取 12 段 + 肉块取 20 段）适用于推理小说，其他类型可能需要调整

---

## 六、关键代码片段速查

### 文本清洗

```python
import re
cleaned = re.sub(r'^书名：[^\n]+\n作者：[^\n]+\n简介：[^\n]+\n', '', raw, count=1)
cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
cleaned = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', cleaned)
```

### 章节切分

```python
chapter_pattern = re.compile(r'(第\d+章\s+\S+)')
splits = list(chapter_pattern.finditer(cleaned))
for i, m in enumerate(splits):
    title = m.group(1)
    start = m.start()
    end = splits[i + 1].start() if i + 1 < len(splits) else len(cleaned)
    body = cleaned[start:end]
    body = chapter_pattern.sub('', body, count=1).strip()
```

### 骨架/肉块分类

```python
SKELETON_KW = ['因此', '所以', '因为', '于是', '显然', '看来', '结论', '推断', '推理',
               '分析', '判断', '证据', '线索', '逻辑', '破案', '发现', '说明', '意味',
               '认为', '想必', '肯定', '决定', '计划', '假设', '推测', '怀疑',
               '关键在于', '问题在于', '原因在于', '事实是']

MEAT_KW = ['看见', '看到', '听到', '听见', '走到', '站起', '坐下', '打开', '关上',
           '手中', '眼睛', '脸色', '房间', '门外', '窗前', '街上', '身上', '地上',
           '光', '黑暗', '苍白', '红色', '鲜红', '血迹', '刀', '枪', '楼梯',
           '壁炉', '马车', '火车', '雾', '雨', '风', '雪', '月光',
           '衣着', '帽子', '大衣', '手杖', '烟斗']

sk = sum(1 for kw in SKELETON_KW if kw in para)
mt = sum(1 for kw in MEAT_KW if kw in para)
dialogue_chars = para.count('"') + para.count('"') + para.count('「') + para.count('」')
has_dialogue = dialogue_chars >= 4
if sk > mt or (has_dialogue and sk >= mt):
    skeleton_paras.append(para)
else:
    meat_paras.append(para)
```

### AMR 情绪词集

```python
EMOTION_WORDS = {
    '害怕', '恐惧', '惊恐', '震惊', '愤怒', '悲哀', '悲伤', '高兴', '快乐',
    '着急', '慌张', '忧愁', '痛恨', '恼怒', '疼痛', '忧虑', '惧怕', '发愣',
    '怔住', '颤抖', '叹气', '哀伤', '欢笑', '气愤', '担心', '激动', '感动',
    '惊讶', '失望', '绝望', '兴奋', '紧张', '焦虑', '痛苦', '欣慰', '得意'
}
EMOTION_CHARS = set('怕惊恐怒悲喜哭笑惨急慌愁恨爱恼疼痛忧惧愣怔震颤叹哀乐愤')
```

### 标签体系（Part 1.5 使用）

```python
FULLTEXT_TAGS = {
    '字数': '长篇',
    '三维倾向': '叙事',
    '世界观/背景': '都市',
    '基调': '正剧',
    '主要人物': '福尔摩斯,华生',
    '视角': '第一人称',
    '来源': '人工原创'
}

# 段级标签维度
# 三维倾向: 取 N/E/V 最高分对应的维度
# 场景类型: 推理场景/对话场景/叙述场景/动作暴力场景/室内场景/室外环境场景/动作场景
# 改编优先级: S级(≥8.5骨架/≥8.0肉块) / A级 / B级 / C级
```

---

## 七、产出文件速查表

| 需求 | 看哪个文件 |
|------|-----------|
| 🆕 在新对话中复现流水线 | `01_最终报告/NLP故事生成流水线_系统提示词_final.md`（本文件） |
| 查看最终分析结果 | `01_最终报告/福尔摩斯NLP故事生成分析报告.md` |
| 编程读取候选数据 | `01_最终报告/福尔摩斯NLP分析数据.json` |
| 查看全量情绪曲线 + 标签 | `01_最终报告/问题D修复_全量情绪曲线与标签.md` |
| 了解 v3→v4 改进 | `02_对比报告/v3_v4_comparison_report.md` |
| 了解评分为什么调了 | `02_对比报告/coefficient_comparison_report.md` |
| 重新跑流水线 | `04_流水线代码/holmes_pipeline_v4.py` |
| 模型验证脚本 | `03_模型测试/test_hanlp_models.py` |

---

## 八、下一阶段任务（Part 2 → Part 3 → Part 4）

> 核心原则：Part 2 **不是把文本拆成更短的句子**，而是从优秀片段中提取结构化的"信息零部件"，供 Part 3 建库和 Part 4 故事生成使用。

### Part 2：切分处理（信息提取）

对 25 条候选片段按类型分别提取结构化信息。

#### 一、骨架类（type: skeleton）——提取逻辑和情绪信息

| 提取项 | 说明 |
|--------|------|
| **人设清单** | 每个片段涉及的人物、性格特征、人物关系、核心动机 |
| **因果连接** | 事件 A → 导致事件 B → 导致事件 C 的因果链，以及时序先后关系 |
| **情绪转折点** | 情绪变化的关键节点（从 X 情绪转为 Y 情绪）、触发原因、转折强度 |
| **叙事密度** | 该片段的叙事节奏（密集推理 vs 平缓过渡） |

#### 二、肉块类（type: meat）——提取视觉和场景信息

| 提取项 | 说明 |
|--------|------|
| **场景清单** | 地点名称、时间（昼夜/季节）、环境氛围（色调/温度/声音/气味） |
| **动作单元** | 连续动作序列（谁 + 做了什么 + 用什么做），按时间顺序排列 |
| **空间关系** | 人物/物体在场景中的位置、相对距离、移动轨迹（从 A 到 B） |
| **视觉元素** | 道具（具体物品）、服饰、环境细节（光线/天气/植被/建筑） |

**输出格式**：为每条候选片段生成一个结构化条目，包含：来源片段编号（#1-#25）、类型（骨架/肉块）、提取出的所有信息项，以清晰的 Markdown 或 JSON 格式呈现。

**跨组交接要求**：每条提取结果必须携带以下追溯信息，供下游（群像、视频合成一组）通过路径和版本引用：
| 字段 | 说明 |
|------|------|
| **稳定 ID** | 每个条目唯一标识（如 CH-01、SC-03、AC-05），可用于跨组引用 |
| **来源段落映射** | 指向原始小说文件的章节号和段落索引，可反向定位原文 |
| **模型/提示词版本** | 记录提取时使用的 AMR/SRL 模型版本和系统提示词版本 |
| **评审记录** | 预留人工审核状态字段（未审核 / 已确认 / 有争议） |

---

### Part 3：可复用库（将 Part 2 提取物分类归档）

将 Part 2 提取出的所有信息零部件，按以下三类分别建库索引。

#### 第一类：连接逻辑库（记录"块与块之间怎么连"）

| 维度 | 说明 |
|------|------|
| 入库内容 | 因果链、时序关系、人物跨片段关联 |
| 索引方式 | 按人物名、按章节顺序、按因果关系类型 |
| 用途 | 供 Part 4 故事生成时维持叙事连贯性 |

#### 第二类：最小块库（存所有提取出的独立单元）

| 维度 | 说明 |
|------|------|
| 入库内容 | 人设条目（每个人物一条）、场景条目（每个场景一条）、动作单元条目（每个动作序列一条） |
| 标签 | 条目来源（#1-#25）、类型（人设/场景/动作/情绪转折）、关联人物、关联场景 |
| 用途 | 供 Part 4 故事生成时按需调取组合 |

#### 第三类：文本元库（存"块的元信息"）

| 维度 | 说明 |
|------|------|
| 入库内容 | 每一条目的叙事分/情绪分/视觉分、情绪节点数、动作谓词数、对话谓词数 |
| 标签 | 改编优先级（S/A/B级）、场景类型标签 |
| 用途 | 供 Part 4 故事生成时做筛选/排序/过滤 |

**输出格式**：三库索引表，每个库以 Markdown 表格列出所有条目及其标签、来源、元信息。

**跨组交接要求**：三库中的所有条目必须沿用 Part 2 的稳定 ID，并额外补充：
| 字段 | 说明 |
|------|------|
| **来源段落映射** | 每个库条目反向标注其对应的 Part 2 原始片段编号（#1-#25）及原文章节位置 |
| **实体引用** | 如实体组已提供人物/地点/道具的结构化资产 ID，在库条目中标注对应 entity_id |
| **模型/提示词版本** | 记录建库时的系统提示词版本、AMR/SRL 模型版本、评分公式版本 |
| **评审记录** | 三库整体版本号 + 人工审核状态（未审核 / 已确认 / 有争议） |

---

### Part 4：故事生成（从可复用库调取信息，生成新故事）

#### 目标
从三库中调取 Part 2/3 提取的结构化"故事基因"（人设、因果链、场景、动作、情绪、叙事密度），合成一篇**完整、可阅读、可编辑的新故事文本**。

> 注意：故事组不负责改编成剧本或分镜脚本，那是视频合成一组的职责。Part 4 的输出物是**故事**，不是剧本。

#### 生成逻辑
1. **选基因**：从三库中按星级评分筛选高价值元素（人设 ★★★★★、动作 ★★★★★、情绪转折 ★★★★★ 优先调取）
2. **搭骨架**：用因果连接链 + 人物关系网构建故事主干（谁 → 遇到什么冲突 → 如何解决）
3. **填肉**：将动作单元 + 场景条目嵌入骨架中，形成完整的叙事段落
4. **调节奏**：根据叙事密度和情绪转折条目的强度分布，安排故事的起伏节奏（紧张 → 释放 → 再紧张）

#### 输入来源
- 可以是**单本小说**的提取库（如当前福尔摩斯 25 条候选），生成同风格的新故事
- 也可以是**多本小说**的提取库融合，生成跨风格的混合故事

#### 输出格式
- **输出文件**：`生成故事.md`
- **内容要求**：完整的故事文本，有明确的开头（引入冲突）、发展（因果推进）、结尾（解决/反转）
- **可追溯性**：每条生成内容需标注来源（调用了哪个库的哪些条目）

---


---

## 九、暂记事项（记住，暂不执行）

以下内容已确认方向，但因当前仅做单本测试，暂不纳入流水线执行：

| # | 事项 | 说明 | 触发条件 |
|---|------|------|----------|
| 1 | **总结经典故事架构** | 输入多本小说后，从可复用库中抽象出跨文本的故事架构模式（如"密室杀人型""宿命对决型""层层反转型"），作为 Part 4 故事生成的结构模板 | 多本小说 Part 0-3 跑完后启用 |
| 2 | **输入契约定义** | Part 0 需要明确：什么样的 .txt 可以进入流水线、实体组（YingHe-entity）的结构化资产如何对接 | 实体组数据就绪后正式定义 |
| 3 | **多本入 / 单本出模式** | 真实场景是 N 本小说通过 Part 0-3 提取基因汇入同一个可复用库，再由 Part 4 生成一篇新故事。当前单本入单本出仅为测试验证 | 多本小说素材到位后切换 |

---

*本文件是完整、可迁移的项目说明书。新设备上按照"一、运行环境"配置后，运行"二、模型验证命令"确认一切就绪，即可从 Part 2 继续推进。*
