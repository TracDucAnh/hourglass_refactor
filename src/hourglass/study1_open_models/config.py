# Config & đường dẫn lưu trữ (Cell A1)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 1 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell A1 — Config & đường dẫn lưu trữ
# ============================================================
import os, gc, json, time, math, re, itertools, unicodedata, urllib.request, urllib.error, warnings
from datetime import datetime
import numpy as np

from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", "MY_KEY")      # hoặc dán token trực tiếp ở đây
if HF_TOKEN and HF_TOKEN != "MY_KEY":
    login(token=HF_TOKEN)


# ---- Storage: mọi kết quả nằm dưới /mnt/cl ----
ROOT         = param("ROOT", os.getcwd())
DATA_DIR     = param("DATA_DIR", f"{ROOT}/data")
STUDY_DIR    = param("STUDY_DIR", f"{ROOT}/study1")
FIG_DIR      = param("FIG_DIR", f"{STUDY_DIR}/figures")
ANALYSIS_DIR = param("ANALYSIS_DIR", f"{STUDY_DIR}/analysis")
HF_CACHE_DIR = param("HF_CACHE_DIR", None)            # vd. f"{ROOT}/hf_cache" nếu ổ home không đủ chỗ cho ~150GB trọng số
for _d in (f"{DATA_DIR}/raw/ntrex128", f"{DATA_DIR}/processed/parallel", STUDY_DIR, FIG_DIR, ANALYSIS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- Siêu tham số thí nghiệm ----
SEED         = param("SEED", 42)
N_SENT       = param("N_SENT", None)      # None = dùng toàn bộ (FLORES devtest 1012, NTREX 1997). Đặt 200 để test nhanh
                         # (kết quả test được lưu ở thư mục riêng *_n200, không lẫn với kết quả thật)
N_SHUFFLES   = param("N_SHUFFLES", 1000)      # shuffled-pair baseline (Eq. 18)
N_BOOT       = param("N_BOOT", 1000)      # bootstrap theo content (95% CI)
CSLS_K       = param("CSLS_K", 10)        # số láng giềng cho hubness correction của CSLS
D_PCA        = param("D_PCA", 64)        # số chiều PCA trước khi khớp phép xoay Procrustes (fit nửa này, chấm nửa kia)
PROBE_SEEDS  = param("PROBE_SEEDS", 3)         # số cách chia content train/test cho probe M4
MASK_LEVELS  = param("MASK_LEVELS", [0.0, 0.25, 0.5, 0.75])   # diffusion: tỉ lệ token bị che
N_MASK_DRAWS = param("N_MASK_DRAWS", 8)                         # số lần che ngẫu nhiên / câu / mức che (>0)

# ---- Điều khiển chạy ----
DATASETS_TO_RUN      = param("DATASETS_TO_RUN", ["flores", "ntrex"])
RUN_MODELS           = param("RUN_MODELS", None)     # None = cả 15 model; hoặc list key, vd. ["xlmr_base", "qwen3_1.7b", "olmoe"]
NTREX_MODEL_KEYS     = param("NTREX_MODEL_KEYS", ["xlmr_base", "qwen3_1.7b", "olmoe", "llada_8b"])   # NTREX lặp ở 3 mô hình đại diện + 2 diffusion. None = tất cả
RUN_RANDOM_BASELINES = param("RUN_RANDOM_BASELINES", True)
FORCE_RERUN          = param("FORCE_RERUN", False)
SANITY_CHECK         = param("SANITY_CHECK", True)     # kiểm tra batch-vs-single (padding) 1 lần cho mỗi model

# 12 ngôn ngữ của Study 1 (Table 8 / kế hoạch). Tiếng Anh đứng đầu (dùng cho J_k).
LANGS = param("LANGS", ["eng_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "vie_Latn",
         "tur_Latn", "arb_Arab", "jpn_Jpan", "fin_Latn", "swh_Latn", "tha_Thai"])

def ds_tag(ds):
    return ds if N_SENT is None else f"{ds}_n{N_SENT}"
