# Config, đường dẫn, kích thước lưới, giao thức huấn luyện v2 (Cell A1)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 3 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell A1 — Config & đường dẫn lưu trữ
# ============================================================
import os, gc, json, time, math, re, itertools, random, bisect, heapq, warnings, collections, traceback
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm

SMOKE = param("SMOKE", False)      # True: chạy thử cực nhỏ (vài phút, lưu ở study2a_smoke/) để kiểm tra đường ống trước khi chạy thật
ROOT         = param("ROOT", os.environ.get("LED_ROOT", os.getcwd()))
STUDY_DIR    = param("STUDY_DIR", f"{ROOT}/study2a") + ("_smoke" if SMOKE else "")
WORLD_DIR    = f"{STUDY_DIR}/worlds"
RUN_DIR      = f"{STUDY_DIR}/runs"
ANALYSIS_DIR = param("ANALYSIS_DIR", f"{STUDY_DIR}/analysis")
FIG_DIR      = param("FIG_DIR", f"{STUDY_DIR}/figures")
for _d in (WORLD_DIR, RUN_DIR, ANALYSIS_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

def _S(full, small):
    return small if SMOKE else full

SEED = param("SEED", 42)

# ---- Ngôn ngữ tổng hợp / dữ liệu ----
N_POOL       = param("N_POOL", _S(1_600_000, 12_000))   # số câu trong pool huấn luyện (mỗi câu = một nội dung mới, một ngôn ngữ)
N_EVAL       = param("N_EVAL", _S(3000, 300))         # nội dung giữ ra để đo (x L ngôn ngữ)
N_PROBE      = param("N_PROBE", _S(2000, 300))         # tập dò S3: danh từ chủ ngữ lấy đều theo hạng
BPE_VOCAB    = param("BPE_VOCAB", _S(8000, 1200))        # số ký hiệu BPE chung (ký tự + merge)
BPE_MAX_TYPES = param("BPE_MAX_TYPES", _S(80_000, 20_000))   # số loại từ (theo tần suất) dùng để huấn luyện BPE
MAX_LEN      = param("MAX_LEN", 64)                    # chiều dài chuỗi (token); câu <= MAX_SENT_TOK + 2 (bos/eos), chừa chỗ cho hậu tố nhãn
MAX_SENT_TOK = param("MAX_SENT_TOK", 60)
MIN_WORDS    = param("MIN_WORDS", 3)                     # R27: câu ngắn hơn bị loại (giảm mơ hồ chéo ngôn ngữ ở giao diện cân bằng)
MAX_DEPTH    = param("MAX_DEPTH", 4)                     # R25
HELDOUT_PCT  = param("HELDOUT_PCT", 5)                     # R28: % tổ hợp (vị ngữ, danh từ arg0, danh từ arg1) bị giữ ra
N_SEM_CLS    = param("N_SEM_CLS", 50)

# ---- Mô hình / huấn luyện (Study 2a, kế hoạch mục 4) ----
K_LAYERS = param("K_LAYERS", 8)
D_HEAD   = param("D_HEAD", 32)
D_MAIN   = param("D_MAIN", 192)                              # d dùng cho các lưới không quét capacity

# ---- Batch / số bước (GIAO THỨC HUẤN LUYỆN v2) ----
# Chẩn đoán từ cổng G2 cũ (chỉ 750 bước cập nhật với batch rất lớn, 7,7 epoch): loss LM ~5,3 (eval ~5,05) so với SÀN entropy của ngôn ngữ
# ~4,3 nats/token  =>  CHƯA HỘI TỤ. Nguyên nhân: giữ nguyên TỔNG SỐ CHUỖI nhưng giảm 16x SỐ LẦN CẬP NHẬT (12000 -> 750).
# Mô hình 3,5M tham số cần nhiều lần cập nhật chứ không cần batch lớn (batch quá lớn vượt critical batch size).
# v2: batch nhỏ hơn (nhiều bước hơn) + nhân tổng số chuỗi bằng SEQ_MULT.
#   steps  = STEPS_REF * (BATCH_REF / BATCH) * SEQ_MULT        (tổng chuỗi = BATCH_REF * STEPS_REF * SEQ_MULT)
#   warmup = WARMUP_REF * (BATCH_REF / BATCH) * SEQ_MULT       (tỉ lệ theo số bước)
#   lr đỉnh nhân (BATCH / BATCH_REF) ** LR_BATCH_EXP           (0.5 = căn bậc hai)
# Mặc định BATCH=1024, SEQ_MULT=2 (6000 bước, ~3,8 epoch trên pool 1,6M) là ƯỚC LƯỢNG. CHỐT bằng lưới `pilot_conv` (Cell C5b) rồi đóng băng.
BATCH_REF  = param("BATCH_REF", _S(256, 64))
STEPS_REF  = param("STEPS_REF", _S(12000, 150))
WARMUP_REF = param("WARMUP_REF", _S(500, 15))
BATCH        = param("BATCH", _S(1024, 64))       # A100 80GB: logits fp32 là phần tốn nhất (~2,4 GB / 1024 chuỗi)
SEQ_MULT     = param("SEQ_MULT", _S(2.0, 1.0))       # hệ số nhân tổng số chuỗi so với cấu hình tham chiếu (256 x 12000)
SEQ_MULT_FAMILY = param("SEQ_MULT_FAMILY", None)   # dict họ -> SEQ_MULT cho run của họ đó (configs/experiment/study2a.yaml); None = mọi họ dùng SEQ_MULT
LR_BATCH_EXP = param("LR_BATCH_EXP", 0.5)
TRAIN_VERSION = param("TRAIN_VERSION", 3)                 # run có train_version khác sẽ bị chạy lại / bị loại khỏi phân tích (v1 = 750 bước, chưa hội tụ; v2 = mọi họ cùng ngân sách)

def train_hparams(bs=None, seq_mult=None):
    """Siêu tham số huấn luyện cho một run (bs/seq_mult = None -> mặc định toàn cục). Dùng cho lưới pilot_conv."""
    bs = int(bs or BATCH); m = float(SEQ_MULT if seq_mult is None else seq_mult)
    scale = BATCH_REF / bs
    return dict(batch=bs, seq_mult=m, steps=max(1, round(STEPS_REF * scale * m)),
                warmup=max(1, round(WARMUP_REF * scale * m)), lr_scale=(bs / BATCH_REF) ** LR_BATCH_EXP)

HP = train_hparams()
STEPS_SCALE = BATCH_REF / BATCH
STEPS, WARMUP, LR_SCALE = HP["steps"], HP["warmup"], HP["lr_scale"]
EVAL_BS  = param("EVAL_BS", _S(8192, 128))          # batch cho các forward KHÔNG gradient: trích biểu diễn, logit-lens, eval loss (mỗi chuỗi độc lập)

# ---- Giám sát hội tụ ----
VAL_N      = param("VAL_N", _S(2048, 128))        # số chuỗi giữ ra (tổ hợp R28) và số chuỗi huấn luyện dùng để đo val/trn trong lúc train
VAL_BS     = param("VAL_BS", _S(512, 64))          # batch khi đo val (logits fp32: 512 x 63 x V ~ 1,2 GB)
CONV_TOL   = param("CONV_TOL", 0.02)                 # nats: val còn giảm hơn mức này trong ~1/6 số bước cuối -> chưa hội tụ
OVERFIT_TOL = param("OVERFIT_TOL", 0.10)                # nats: val - trn lớn hơn mức này -> nghi nhớ thuộc lòng (cần thêm mẫu mới, không thêm epoch)
GATE_MIN_K = param("GATE_MIN_K", 2)                    # cổng G2: đỉnh G ở khối đầu tiên (k*=1) không được coi là waist
AUX_FRAC = param("AUX_FRAC", 0.20)          # tỉ lệ chuỗi trong batch có nhãn phụ (SEM / SURF)
AUX_W    = param("AUX_W", 1.0)           # trọng số loss nhãn phụ so với loss LM trung bình
MOE_AUX  = param("MOE_AUX", 0.01)          # hệ số load-balancing của MoE

# ---- Đo ----
PROBE_SEEDS  = param("PROBE_SEEDS", _S(2, 1))
D_PCA        = param("D_PCA", 64)
N_BOOT       = param("N_BOOT", _S(500, 100))
N_RANK_BINS  = param("N_RANK_BINS", 12)                         # S3: bin theo floor(log2(hạng)) của danh từ (hạng 1..2500 -> 0..11)

# ---- Điều khiển chạy ----
RUN_GRIDS   = param("RUN_GRIDS", ["pilot_conv"])  # CHẠY pilot_conv TRƯỚC (chốt BATCH/SEQ_MULT) -> gate_g2 -> các lưới còn lại. Xem Cell C1:
                              # pilot_conv | gate_g2 | smoke | p1p2 | moe_diff
FORCE_RERUN = param("FORCE_RERUN", False)
FAMILY      = param("FAMILY", None)   # None = mọi họ | "decoder" | "encoder" | "moe" | "diffusion": run_grid chỉ chạy run của họ này (mỗi họ một tiến trình/GPU)
SHARD       = param("SHARD", None)    # None | "k/n": sau khi lọc FAMILY, chỉ chạy run thứ k, k+n, k+2n... của mỗi lưới (chia một họ cho n GPU)
print(f"[config] STUDY_DIR={STUDY_DIR}  SMOKE={SMOKE}  N_POOL={N_POOL}\n"
      f"[config] giao thức v{TRAIN_VERSION}: BATCH={BATCH} (ref {BATCH_REF})  STEPS={STEPS} (ref {STEPS_REF})  WARMUP={WARMUP}  "
      f"lr_scale={LR_SCALE:.2f}  SEQ_MULT={SEQ_MULT:g}  EVAL_BS={EVAL_BS}  -> {BATCH * STEPS / N_POOL:.1f} epoch")
