# 福尔摩斯NLP剧本改编分析 — 产出文件索引

> 更新: 2026-07-21 | 最终版本: v4

## 目录结构

```
产出/
├── 01_最终报告/          ← 核心成果，优先看这里
│   ├── NLP剧本改编流水线_系统提示词.md         ★ 系统提示词：完整公式/原理/代码/已知问题，可复用到新对话
│   ├── 福尔摩斯NLP剧本改编分析报告.md          v4最终分析报告（清洗/切分/双模型评分/情绪曲线/高价值候选Top15）
│   └── 福尔摩斯NLP分析数据.json               v4结构化数据（32条候选的完整分数+文本+AMR/SRL原始输出）
│
├── 02_对比报告/          ← 版本迭代过程的分数对比
│   ├── v3_v4_comparison_report.md             v3→v4：分类型权重 + 对话动词降权 + 真实情绪曲线 的影响
│   ├── coefficient_comparison_report.md        系数调优记录：v5(全撞顶)→v6(有区分度) 的前后对比
│   ├── score_comparison_report.md              评分体系版本间的分数对比与根因分析
│   ├── 评分系数调优对比.md                     系数调优对比（中文版摘要）
│   └── 流水线版本分数对比.md                   v2(旧)→v3(新) 分数对比
│
├── 03_模型测试/          ← AMR/SRL 模型的初始验证
│   ├── AMR_SRL模型测试报告.txt                 perin-parser + HanLP 可用性测试
│   └── test_hanlp_models.py                    HanLP 模型加载与推理测试脚本
│
├── 04_流水线代码/        ← 主流水线脚本（版本演进）
│   ├── holmes_pipeline_v4.py        ✅ 最终版：分类型权重+对话降权+双模型全跑+渐进重试+句级SRL
│   ├── holmes_pipeline_v3.py             中间版：双模型全跑+无proxy（系数过大导致分数无区分度）
│   ├── holmes_pipeline_v2.py             中间版：jieba分词+带proxy公式（假分问题）
│   └── holmes_pipeline.py                初始版：粗粒度切分+基础评分
│
├── 05_测试脚本/          ← AMR/SRL 单独测试用
│   ├── test_final.py                     AMR+SRL 联合测试（UTF-8输出）
│   ├── test_amr_srl.py                   初步测试
│   ├── test_amr_only.py                  AMR 单独测试
│   └── test_amr_v2.py                    AMR 分词格式测试
│
├── 06_辅助脚本/          ← 对比/分析工具
│   ├── compare_v3_v4.py                  v3→v4 对比生成脚本
│   ├── compare_v5_v6.py                  系数调优前后对比生成脚本
│   └── compare_scores.py                 版本间分数对比生成脚本
│
└── 07_历史备份/          ← 旧版数据存档
    ├── holmes_analysis_data_v5_all10.json     	v5版 JSON（全撞顶 10.0）
    ├── holmes_analysis_data_v3_backup.json    	v3版 JSON（系数过大）
    ├── holmes_analysis_data_v2_old.json       	v2版 JSON（有 proxy）
    ├── holmes_nlp_analysis_report_v3_backup.md	v3版分析报告
    └── holmes_nlp_analysis_report_v2_old.md   	v2版分析报告
```

## 快速导航

| 需求 | 看哪个文件 |
|------|-----------|
| 🆕 在新对话中复现流水线 | `01_最终报告/NLP剧本改编流水线_系统提示词.md` |
| 查看最终分析结果 | `01_最终报告/福尔摩斯NLP剧本改编分析报告.md` |
| 编程读取候选数据 | `01_最终报告/福尔摩斯NLP分析数据.json` |
| 了解 v3→v4 改进 | `02_对比报告/v3_v4_comparison_report.md` |
| 了解评分为什么调了 | `02_对比报告/coefficient_comparison_report.md` |
| 重新跑流水线 | `04_流水线代码/holmes_pipeline_v4.py` |

## v4 评分体系摘要

| 维度 | 骨架权重 | 肉块权重 |
|------|---------|---------|
| 叙事 (AMR) | 0.55 | 0.10 |
| 情绪 (AMR) | 0.35 | 0.10 |
| 视觉 (SRL) | 0.10 | 0.80 |
| **阈值** | | **2.0** |

- AMR 输入: jieba 分词后截 80 字，渐进重试 80→50→30
- SRL 输入: 句级切分（每句 ≤100 字），避免 ELECTRA 199 token 超限
- 对话动词降权: 说/问/答等谓词权重 0.45→0.15
- 情绪曲线: 逐章采样最长段落，AMR 提取情绪节点数

## 运行环境

- Python: `C:\Users\genji\.workbuddy\binaries\python\envs\default\Scripts\python.exe`
- 依赖: hanlp 2.1.3, perin-parser 0.0.19, transformers 4.30.0, torch 2.13.0, jieba 0.42.1
- AMR 模型: `MRP2020_AMR_ZHO_MENGZI_BASE` (perin_parser + Mengzi BERT)
- SRL 模型: `CPB3_SRL_ELECTRA_SMALL` (HanLP SpanBIO + ELECTRA-small)
