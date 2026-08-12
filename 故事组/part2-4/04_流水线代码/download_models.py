"""下载 3 个 HuggingFace 模型到本地缓存"""
import os, io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 设置 HF 镜像（中国大陆必须）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

MODELS = [
    ("Langboat/mengzi-bert-base", "AMR 编码器 (Mengzi BERT)", "~400 MB"),
    ("hfl/chinese-electra-180g-small-discriminator", "SRL 编码器 (ELECTRA-small)", "~80 MB"),
    ("xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2", "AMR 解析器 (BART-large)", "~1.6 GB"),
]

for model_id, name, size in MODELS:
    print(f"\n{'='*60}")
    print(f"Downloading: {name}")
    print(f"Model: {model_id} ({size})")
    print(f"{'='*60}")
    try:
        local_path = snapshot_download(
            repo_id=model_id,
            resume_download=True,
            max_workers=4,
        )
        print(f"OK -> {local_path}")
    except Exception as e:
        print(f"FAILED: {e}")

print("\n\nAll downloads completed!")
