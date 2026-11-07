# Config, đường dẫn, ngôn ngữ, giao thức huấn luyện (Study 2b)
# Study 2b — Ngôn ngữ tự nhiên: căn chỉnh theo capacity và số ngôn ngữ (P3, S2, S4; P4, S5 phụ).
# LIB (định nghĩa): luôn được nạp khi gọi load(). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
import os, gc, json, math, time, random, itertools, hashlib, traceback
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm

SMOKE = param("SMOKE", False)        # True: dữ liệu toy + mô hình tí hon, chạy được trên CPU trong vài phút

# ---- Đường dẫn ----
ROOT         = param("ROOT", os.getcwd())
DATA_DIR     = param("DATA_DIR", f"{ROOT}/data")
HF_CACHE_DIR = param("HF_CACHE_DIR", None)
STUDY_DIR    = param("STUDY_DIR", f"{ROOT}/outputs/study2b") + ("_smoke" if SMOKE else "")
# decoder (joint + twin) lưu ở đây theo định dạng HF để Study 4 tự đăng ký (thư mục twin_* bị Study 4 bỏ qua);
# MoE / diffusion và mọi checkpoint của SMOKE lưu trong STUDY_DIR để không lẫn vào registry của Study 4.
CKPT_DIR     = param("CKPT_DIR", f"{ROOT}/study3_natural/checkpoints")
OTHER_CKPT_DIR = f"{STUDY_DIR}/checkpoints"
if SMOKE:
    CKPT_DIR = OTHER_CKPT_DIR
ANALYSIS_DIR = param("ANALYSIS_DIR", f"{STUDY_DIR}/analysis")
FIG_DIR      = param("FIG_DIR", f"{STUDY_DIR}/figures")
CORPUS_DIR   = f"{STUDY_DIR}/corpus"
TOK_DIR      = f"{STUDY_DIR}/tokenizers"
RUN_DIR      = f"{STUDY_DIR}/runs"
for _d in (CORPUS_DIR, TOK_DIR, RUN_DIR, ANALYSIS_DIR, FIG_DIR, CKPT_DIR, OTHER_CKPT_DIR):
    os.makedirs(_d, exist_ok=True)
SEED = param("SEED", 42)

# ---- Ngôn ngữ ----
# Hai thứ tự lồng 2 ⊂ 4 ⊂ 8 (|L| = k lấy k ngôn ngữ đầu). A cân bằng trật tự từ cơ bản ở mọi cỡ; B cố ý dồn các
# ngôn ngữ khác trật tự (SOV, VSO) về cuối, nên |L| lớn = khoảng cách typology lớn. So A với B tách hiệu ứng |L|
# khỏi hiệu ứng khoảng cách typology (Study 2a: trật tự từ khác nhau làm eo của decoder nông đi).
NESTINGS = param("NESTINGS", {
    "A": ["eng_Latn", "hin_Deva", "rus_Cyrl", "tur_Latn", "zho_Hans", "jpn_Jpan", "fra_Latn", "arb_Arab"],
    "B": ["eng_Latn", "fra_Latn", "rus_Cyrl", "zho_Hans", "hin_Deva", "tur_Latn", "jpn_Jpan", "arb_Arab"],
})
ALL_LANGS = sorted({l for v in NESTINGS.values() for l in v})
L_LIST = param("L_LIST", [2, 4, 8])

# ---- Dữ liệu ----
DATA_SOURCE      = param("DATA_SOURCE", "fineweb")       # "fineweb" (FineWeb-2 + FineWeb cho tiếng Anh) | "toy"
CORPUS_BYTES_ENG = param("CORPUS_BYTES_ENG", 800_000_000) # corpus mỗi ngôn ngữ = số byte này x byte premium (≈ 200M token tiếng Anh)
DATA_FRAC        = param("DATA_FRAC", 0.5)                # phần corpus một run dùng (0.5 ≈ 100M token / ngôn ngữ); cố định khi đổi |L|
VAL_FRAC         = param("VAL_FRAC", 0.002)               # đuôi corpus giữ ra làm tập val (không bao giờ huấn luyện)
N_EVAL           = param("N_EVAL", None)                  # số câu FLORES devtest dùng để đo (None = 1012)
TOY_N_DOCS       = param("TOY_N_DOCS", 3000)              # SMOKE / toy: số văn bản mỗi ngôn ngữ
TOY_N_EVAL       = param("TOY_N_EVAL", 300)

# ---- Tokenizer ----
VOCAB        = param("VOCAB", 64000)                      # < 65536 -> token lưu uint16
TOK_TRAIN_BYTES_ENG = param("TOK_TRAIN_BYTES_ENG", 50_000_000)  # văn bản train BPE mỗi ngôn ngữ (x premium x trọng số)
SEQ_LEN      = param("SEQ_LEN", 512)

# ---- Mô hình (kiểu Llama, 12 tầng; MoE: 8 expert nửa cỡ, top-2) ----
N_LAYERS     = param("N_LAYERS", 12)
D_HEAD       = param("D_HEAD", 64)
MOE_EXPERTS  = param("MOE_EXPERTS", 8)
MOE_TOPK     = param("MOE_TOPK", 2)
MOE_AUX      = param("MOE_AUX", 0.01)

# ---- Huấn luyện ----
BATCH_SEQS   = param("BATCH_SEQS", 128)                   # 128 x 512 = 65k token / bước
LR_BY_D      = param("LR_BY_D", {128: 2e-3, 192: 1.5e-3, 256: 1.2e-3, 320: 1e-3, 384: 1e-3, 448: 8e-4, 512: 7e-4})
WARMUP_FRAC  = param("WARMUP_FRAC", 0.02)
MIN_LR_FRAC  = param("MIN_LR_FRAC", 0.1)
WEIGHT_DECAY = param("WEIGHT_DECAY", 0.1)
JOINT_EPOCHS = param("JOINT_EPOCHS", 1)                   # mô hình chung thấy mỗi token một lần
TWIN_EPOCHS  = param("TWIN_EPOCHS", 2)                    # twin lặp dữ liệu của mình (<= ~4 epoch gần như dữ liệu mới)
EPOCH_MULT_FAMILY = param("EPOCH_MULT_FAMILY", {"decoder": 1.0, "moe": 1.0, "diffusion": 2.0})  # 2a: diffusion cần lâu hơn
DIFF_T_MIN   = param("DIFF_T_MIN", 0.05)
N_EVALS      = param("N_EVALS", 10)                       # số lần đo val trong một run
VAL_CHUNKS   = param("VAL_CHUNKS", 64)                    # số chuỗi val mỗi ngôn ngữ
CONV_TOL     = param("CONV_TOL", 0.02)                    # nats: val còn giảm hơn mức này trong ~15% bước cuối -> chưa hội tụ
TWIN_EXCESS_MAX = param("TWIN_EXCESS_MAX", 0.10)          # nats: twin kém mô hình chung quá mức này -> ΔA có thể bị thổi phồng
TRAIN_VERSION = param("TRAIN_VERSION", 1)

# ---- Đo ----
READOUT      = param("READOUT", "family")                 # "family": causal = token cuối, hai chiều = trung bình | "last" | "mean"
D_PCA        = param("D_PCA", 64)
DELTA_A_MARGIN = param("DELTA_A_MARGIN", 0.05)            # biên "ΔA ≈ 0" (TOST của paper: max(0.05, 1 SD seed))
METRIC_VERSION = param("METRIC_VERSION", 1)

# ---- Lưới (Study 2b) ----
D_P3        = param("D_P3", [256, 384, 512])
D_S2        = param("D_S2", [128, 192, 256, 320, 384, 448, 512])
D_HOLDOUT   = param("D_HOLDOUT", [320])
D_FAMILY    = param("D_FAMILY", 384)                      # cỡ duy nhất của MoE / diffusion
N_SEEDS     = param("N_SEEDS", 3)
N_SEEDS_S2  = param("N_SEEDS_S2", 10)
S2_FRAC     = param("S2_FRAC", 0.25)                      # S2 chạy ở nửa dữ liệu để vừa ngân sách
PILOT_FRACS = param("PILOT_FRACS", [0.125, 0.25, 0.5, 1.0])
S4_LANG     = param("S4_LANG", "rus_Cyrl")                # ngôn ngữ X của thiết kế 2 x 2 (S4); phải thuộc 4 ngôn ngữ đầu của NESTINGS["A"], không phải tiếng Anh
D_S4        = param("D_S4", 384)
S4_TOK_W    = param("S4_TOK_W", [0.25, 4.0])              # trọng số của X khi train BPE: 0.25 = X tốn token, 4 = X rẻ
S4_SHARE_MULT = param("S4_SHARE_MULT", 2.0)               # "X nhiều dữ liệu": X dùng gấp đôi phần corpus của mỗi ngôn ngữ khác

# ---- Điều khiển chạy ----
RUN_GRIDS   = param("RUN_GRIDS", ["pilot_data"])          # pilot_data -> p3 -> p3_families -> s4 -> s2 -> holdout_p4
FAMILY      = param("FAMILY", None)                       # None = mọi họ; hoặc "decoder" | "moe" | "diffusion"
SHARD       = param("SHARD", None)                        # None | "k/n": chỉ chạy run thứ k, k+n, ... của mỗi lưới
FORCE_RERUN = param("FORCE_RERUN", False)
print(f"[config] STUDY_DIR={STUDY_DIR}  SMOKE={SMOKE}  DATA_SOURCE={DATA_SOURCE}  ngôn ngữ={len(ALL_LANGS)}  "
      f"DATA_FRAC={DATA_FRAC}  BATCH={BATCH_SEQS}x{SEQ_LEN}  RUN_GRIDS={RUN_GRIDS}")
