# -*- coding: utf-8 -*-
"""
HanLP AMR / SRL 模型测试脚本
=============================
测试 D:/hanlp/ 下的 AMR 和 SRL 模型能否正常加载和推理。

用法：
    1. 先装依赖（建议 Python 3.10）：
         pip install hanlp
       （会自动装 CPU 版 torch；若有 NVIDIA GPU 想用 CUDA 加速，
         先按 https://pytorch.org/ 装好 GPU 版 torch，再 pip install hanlp）
    2. 运行：
         python test_hanlp_models.py
    3. 看输出里每个模型的 ✅ / ❌ 结果。

说明：
    - 中文 AMR (amr-zho-mengzi-base)：encoder 是 mengzi-bert-base，
      已在 HF 缓存中，可完全离线加载。
    - SRL (cpb3)：encoder 是 hfl/chinese-electra-180g-small-discriminator，
      本地 electra 目录只有配置没有权重，首次加载会联网下载约 50MB。
    - AMR3 (amr3_graph_pretrain_parser)：只有 config.json 没有 model.pt，
      无法加载，本脚本跳过。
"""
import os
import sys
import traceback

# ============ 配置区（按需修改路径）============
HANLP_ROOT = r'D:/hanlp'

SRL_MODEL_DIR = os.path.join(
    HANLP_ROOT, 'srl',
    'cpb3_electra_small_crf_has_transform_20220218_135910'
)
AMR_ZHO_MODEL_DIR = os.path.join(
    HANLP_ROOT, 'amr', 'amr-zho-mengzi-base'
)
# AMR3 缺 model.pt，无法测试，如需测试请先补齐权重
# AMR3_MODEL_DIR = os.path.join(
#     HANLP_ROOT, 'amr', 'amr3_graph_pretrain_parser_20221207_153759'
# )
# =============================================


def check_env():
    """检查 Python / torch / hanlp 环境"""
    print('=' * 60)
    print('[1/3] 环境检查')
    print('=' * 60)
    print(f'Python : {sys.version.split()[0]}  ({sys.executable})')

    try:
        import torch
        print(f'PyTorch: {torch.__version__}  (CUDA available: {torch.cuda.is_available()})')
    except ImportError:
        print('PyTorch: 未安装  <- 请先 pip install hanlp（会自动带 torch）')
        return False

    try:
        import hanlp
        print(f'HanLP  : {hanlp.__version__}')
    except ImportError:
        print('HanLP  : 未安装  <- 请先  pip install hanlp')
        return False

    # 模型版本提示
    print('\n提示：模型是用 hanlp 2.1.0-beta 系列训练的。')
    print('     若新版 hanlp 加载报错，可装对应版本：')
    print('     pip install hanlp==2.1.0-beta.15')
    return True


def test_amr_zho():
    """测试中文 AMR（encoder 走 HF 缓存，可离线）"""
    print('\n' + '=' * 60)
    print('[2/3] 测试 AMR 中文  (amr-zho-mengzi-base)')
    print('=' * 60)
    print(f'模型目录: {AMR_ZHO_MODEL_DIR}')
    if not os.path.isdir(AMR_ZHO_MODEL_DIR):
        print('❌ 目录不存在，跳过')
        return False

    import hanlp
    try:
        print('加载中（encoder = Langboat/mengzi-bert-base，走 HF 缓存）...')
        amr = hanlp.load(AMR_ZHO_MODEL_DIR)
        print('✅ 模型加载成功')

        # AMR 输入需要是已分词的 token list
        test_tokens = ['男孩', '想', '让', '女孩', '相信', '他', '。']
        print(f'测试输入（已分词）: {test_tokens}')
        result = amr(test_tokens)
        print('✅ 推理成功，AMR 结果：')
        print(result)
        return True
    except Exception as e:
        print(f'❌ AMR 中文测试失败: {e}')
        traceback.print_exc()
        return False


def test_srl():
    """测试 SRL（首次会联网下载 electra 权重 ~50MB）"""
    print('\n' + '=' * 60)
    print('[3/3] 测试 SRL  (cpb3_electra_small_crf_has_transform)')
    print('=' * 60)
    print(f'模型目录: {SRL_MODEL_DIR}')
    if not os.path.isdir(SRL_MODEL_DIR):
        print('❌ 目录不存在，跳过')
        return False

    import hanlp
    try:
        print('加载中（encoder = hfl/chinese-electra-180g-small-discriminator）')
        print('      首次会联网下载 electra 权重约 50MB，请耐心等待...')
        srl = hanlp.load(SRL_MODEL_DIR)
        print('✅ 模型加载成功')

        # SRL 输入是已分词的 token list（CPB3 语料风格）
        test_tokens = ['上海', '浦东', '开发', '与', '法制', '建设', '同步']
        print(f'测试输入（已分词）: {test_tokens}')
        result = srl(test_tokens)
        print('✅ 推理成功，SRL 结果：')
        print(result)
        return True
    except Exception as e:
        print(f'❌ SRL 测试失败: {e}')
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print('HanLP AMR / SRL 模型测试')
    print(f'模型根目录: {HANLP_ROOT}\n')

    if not check_env():
        print('\n环境未就绪，请先安装依赖后重试。')
        sys.exit(1)

    amr_ok = test_amr_zho()
    srl_ok = test_srl()

    print('\n' + '=' * 60)
    print('测试总结')
    print('=' * 60)
    print(f'  AMR 中文 : {"✅ 可用" if amr_ok else "❌ 不可用"}')
    print(f'  SRL      : {"✅ 可用" if srl_ok else "❌ 不可用"}')
    print(f'  AMR3     : ⏭  跳过（缺 model.pt 权重）')
    print('=' * 60)
