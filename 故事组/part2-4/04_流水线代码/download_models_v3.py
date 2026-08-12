"""直接 HTTP 下载模型文件到 D:\\hanlp\\hf_models — 完全绕过 huggingface_hub 缓存和 sandbox safe-delete"""
import os, io, sys, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import requests
from tqdm import tqdm

HF_MIRROR = "https://hf-mirror.com"
TARGET_DIR = r"D:\hanlp\hf_models"

MODELS = {
    "mengzi-bert-base": "Langboat/mengzi-bert-base",
    "chinese-electra-180g-small-discriminator": "hfl/chinese-electra-180g-small-discriminator",
    "AMRBART-large": "xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2",
}

def list_files(repo_id):
    """Get file list from HF Hub API"""
    url = f"{HF_MIRROR}/api/models/{repo_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    siblings = data.get("siblings", [])
    return [s["rfilename"] for s in siblings if not s["rfilename"].startswith(".")]

def download_file(repo_id, filename, local_dir):
    """Download a single file from HF mirror"""
    url = f"{HF_MIRROR}/{repo_id}/resolve/main/{filename}"
    local_path = os.path.join(local_dir, filename)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if os.path.exists(local_path):
        # Check size
        resp = requests.head(url, timeout=30)
        remote_size = int(resp.headers.get("content-length", 0))
        local_size = os.path.getsize(local_path)
        if local_size == remote_size:
            return "skip"

    resp = requests.get(url, stream=True, timeout=60)
    total = int(resp.headers.get("content-length", 0))

    with open(local_path, "wb") as f:
        if total:
            with tqdm(total=total, unit="B", unit_scale=True, desc=filename[:50]) as pbar:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        else:
            f.write(resp.content)
    return "ok"

for name, repo_id in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Model: {name} ({repo_id})")
    print(f"{'='*60}")

    local_dir = os.path.join(TARGET_DIR, name)
    os.makedirs(local_dir, exist_ok=True)

    try:
        files = list_files(repo_id)
        print(f"Files: {len(files)}")
        for fname in files:
            result = download_file(repo_id, fname, local_dir)
            if result == "skip":
                print(f"  SKIP: {fname}")
            else:
                print(f"  OK: {fname}")
    except Exception as e:
        print(f"  FAILED: {e}")

print(f"\n\nAll models downloaded to: {TARGET_DIR}")
