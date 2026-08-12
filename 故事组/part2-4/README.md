# part2-4 — 目录索引

> Part 2-4 产出（切分处理 → 可复用库 → 故事生成）
> 生成日期: 2026-08-05 ~ 2026-08-07

```
part2-4/
├── README.md                    ← 本文件
├── 01_最终报告/
│   ├── NLP剧本改编流水线_系统提示词_final.md   ← 系统提示词（生效版本, 2026-08-07）
│   ├── Part2_信息提取结果.md                 ← Part 2 输出: 25候选结构化提取
│   ├── part2_candidates.json                ← Part 2 中间数据: 25候选完整原文
│   ├── Part3_可复用库.md                     ← Part 3 输出: 三库（连接逻辑/最小块/文本元）
│   └── 生成故事.md                           ← Part 4 输出: 《黑沼宅邸》(测试生成)
└── 04_流水线代码/
    ├── download_models.py        ← 模型下载脚本 v1 (huggingface_hub)
    ├── download_models_v2.py     ← 模型下载脚本 v2 (local_dir_use_symlinks=False)
    └── download_models_v3.py     ← 模型下载脚本 v3 (HTTP直下, 绕过sandbox)
```
