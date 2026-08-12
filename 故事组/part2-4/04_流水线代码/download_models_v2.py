"""下载 3 个 HuggingFace 模型 — 绕过 symlink + safe-delete 问题"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 关键：禁用 symlink，直接复制文件
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import snapshot_download

MODELS = [
    "Langboat/mengzi-bert-base",
    "hfl/chinese-electra-180g-small-discriminator",
    "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
]

for model_id in MODELS:
    print(f"\nDownloading {model_id}...")
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir_use_symlinks=False,  # 关键：不用 symlink
            resume_download=True,
            max_workers=2,
        )
        print(f"OK -> {path}")
    except Exception as e:
        print(f"FAILED: {e}")

print("\nAll done!")
