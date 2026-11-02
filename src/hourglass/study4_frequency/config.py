# Config & đường dẫn lưu trữ (Cell A1)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 2 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A1 — Config & đường dẫn lưu trữ (Study 4)
# ============================================================
import os, gc, json, time, math, re, itertools, unicodedata, urllib.request, urllib.error, warnings, pickle, glob, subprocess, shutil
from collections import Counter, defaultdict
from datetime import datetime
import numpy as np

from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", "MY_KEY")      # hoặc dán token trực tiếp ở đây
if HF_TOKEN and HF_TOKEN != "MY_KEY":
    login(token=HF_TOKEN)

# ---- Nguồn dữ liệu ĐO (v3: dùng cả 3 nguồn song song, trước đây chỉ FLORES devtest được đo) ----
MEASURE_SETS = param("MEASURE_SETS", ["flores", "flores_dev", "ntrex"])   # ["flores"] = hành vi cũ (1012 câu). Ba nguồn = ~4000 câu x 12 ngôn ngữ
MEAS         = param("MEAS", "all")                                # tên split của corpus gộp (cache words/, align/)
DATA_TAG     = "_full" if len(MEASURE_SETS) > 1 else ""
FW_TAG       = param("FW_TAG", "_200M")                              # đổi khi đổi FW_MAX_WORDS_*: cache tần suất tách thư mục, không lẫn bản cũ

# ---- Storage ----
# data/ dùng chung với Study 1 (cùng ROOT thì FLORES/NTREX đã xử lý sẽ được đọc lại, không tải lại)
ROOT         = param("ROOT", "/mnt/cl")
DATA_DIR     = param("DATA_DIR", f"{ROOT}/data")
STUDY_DIR    = param("STUDY_DIR", f"{ROOT}/study4")
N_SENT       = param("N_SENT", None)      # None = toàn bộ FLORES devtest (1012 câu). Đặt 200 để test nhanh: kết quả vào thư mục riêng work_n200
RUN_TAG      = ("" if N_SENT is None else f"_n{N_SENT}") + DATA_TAG
WORK_DIR     = f"{STUDY_DIR}/work{RUN_TAG}"          # tách từ, gióng hàng, cache biểu diễn, kết quả truy hồi
FREQ_DIR     = f"{STUDY_DIR}/freq{FW_TAG}"                   # đếm từ FineWeb-2 (một lần, dùng chung mọi lần chạy)
FIG_DIR      = param("FIG_DIR", f"{STUDY_DIR}/figures")
ANALYSIS_DIR = param("ANALYSIS_DIR", f"{STUDY_DIR}/analysis")
HF_CACHE_DIR = param("HF_CACHE_DIR", None)
for _d in (f"{DATA_DIR}/raw/muse", f"{DATA_DIR}/raw/ntrex128", f"{DATA_DIR}/processed/parallel",
           f"{WORK_DIR}/words", f"{WORK_DIR}/align", f"{WORK_DIR}/cache", f"{WORK_DIR}/runs",
           FREQ_DIR, FIG_DIR, ANALYSIS_DIR):
    os.makedirs(_d, exist_ok=True)

# ---- Ngôn ngữ: giống Study 1 (Table 8); tiếng Anh đứng đầu ----
LANGS = param("LANGS", ["eng_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "zho_Hans", "vie_Latn",
         "tur_Latn", "arb_Arab", "jpn_Jpan", "fin_Latn", "swh_Latn", "tha_Thai"])
SHORT = {l: l.split("_")[0] for l in LANGS}
PAIRS = list(itertools.combinations(range(len(LANGS)), 2))          # 66 cặp
PAIR_NAME = [f"{SHORT[LANGS[a]]}-{SHORT[LANGS[b]]}" for a, b in PAIRS]
SEED = param("SEED", 42)

# ---- Siêu tham số: tách từ / tần suất ----
FW_MAX_DOCS_PER_LANG  = param("FW_MAX_DOCS_PER_LANG", 2_000_000)     # v3: trước 100_000 (thực tế bị chặn bởi 20M từ)
FW_MAX_WORDS_PER_LANG = param("FW_MAX_WORDS_PER_LANG", 200_000_000)   # v3: 10x (trước 20M). Mục tiêu: tỉ lệ dạng từ FLORES/NTREX 'unseen' < ~1% và hạng đuôi bớt nhiễu
FW_WORDS_OVERRIDE     = param("FW_WORDS_OVERRIDE", {"zho_Hans": 100_000_000, "jpn_Jpan": 100_000_000, "tha_Thai": 100_000_000, "vie_Latn": 100_000_000})   # bộ tách từ chậm (jieba/fugashi/newmm/pyvi)
FW_WORKERS            = max(1, min(8, (os.cpu_count() or 2) - 1))

# ---- Siêu tham số: gióng hàng từ (Eflomal) + lọc MUSE ----
ALIGN_EXTRA     = [d for d in ["flores_dev", "ntrex"] if d not in MEASURE_SETS]   # nguồn CHỈ để học gióng hàng (rỗng khi đã nằm trong MEASURE_SETS)
EFLOMAL_MODEL   = param("EFLOMAL_MODEL", 3)                         # 1 = IBM1, 2 = +HMM, 3 = +fertility
EFLOMAL_WORKERS = max(1, min(8, (os.cpu_count() or 2)))
MAX_WORDS_SENT  = param("MAX_WORDS_SENT", 120)                       # bỏ câu quá dài khi gióng hàng
CONFIRM_MODE    = param("CONFIRM_MODE", "muse")                    # "muse" (kế hoạch: giữ cặp được MUSE xác nhận) | "none"
FALLBACK_DICE   = param("FALLBACK_DICE", True)                      # cặp ngôn ngữ ít hơn MIN_PAIR_ITEMS cặp từ được MUSE xác nhận -> xác nhận bằng Dice thống kê (ghi rõ trong bảng)
DICE_MIN, DICE_MIN_COUNT = param("DICE_MIN", 0.4), param("DICE_MIN_COUNT", 2)
MIN_PAIR_ITEMS  = param("MIN_PAIR_ITEMS", 200)
DROP_SAME_FORM  = param("DROP_SAME_FORM", True)                      # bỏ cặp từ cùng dạng chữ (tên riêng, vay mượn): dễ khớp nhờ trùng subword, không phản ánh chia sẻ nghĩa
DROP_NUMERIC    = param("DROP_NUMERIC", True)

# ---- Siêu tham số: sửa lỗi phương pháp (v2) ----
RETR_VERSION = param("RETR_VERSION", 4)          # tăng số này khi đổi cách tính truy hồi -> kết quả cũ tự bị coi là "stale"
HUB_MODE     = param("HUB_MODE", "type")     # CSLS: "type" = hubness tính trên LOẠI từ (mỗi loại đếm một lần) | "instance" = cách cũ (phạt từ phổ biến) | "cos" = cosine thường
MAIN_PAIRS   = param("MAIN_PAIRS", "muse")     # "muse": phân tích chính chỉ dùng cặp thuộc MAIN_SRCS (tiêu chí xác nhận đồng nhất) | "all": mọi cặp
DROP_UNSEEN  = param("DROP_UNSEEN", True)       # bỏ từ không thấy trong FineWeb-2 (hạng giả V+1) khỏi phân tích chính
AXIS         = param("AXIS", "logrank")  # trục x chính: "logrank" (hạng trung bình) | "zipf" (-log10 tần suất/triệu từ, so sánh được giữa ngôn ngữ)
USE_PMIX     = param("USE_PMIX", True)       # thêm covariate: độ dễ trung bình của các cặp ngôn ngữ mà từ tham gia
assert HUB_MODE in ("type", "instance", "cos") and MAIN_PAIRS in ("muse", "all") and AXIS in ("logrank", "zipf")

# ---- Siêu tham số: làm sạch từ điển / tầng phân tích / độ bền theo ngôn ngữ (v4) ----
MUSE_ZH_T2S   = param("MUSE_ZH_T2S", True)                 # MUSE zh-en là phồn thể (體, 球隊) còn corpus là giản thể -> đổi MUSE zh sang giản thể khi nạp
ALT_KEY_LANGS = param("ALT_KEY_LANGS", {"jpn_Jpan"})         # ngôn ngữ tra từ điển bằng THÊM dạng gốc (orthBase/lemma). CHỈ ảnh hưởng bước xác nhận bằng từ điển;
                                     # đơn vị từ, tần suất, gióng hàng Eflomal, biểu diễn mô hình vẫn dùng dạng bề mặt (không phải đếm lại 100M từ)
ALT_DICT      = param("ALT_DICT", True)                 # ngôn ngữ MUSE không có (swh): thử từ điển thay thế (tệp người dùng, hoặc Wiktionary qua kaikki.org) [XM]
MAIN_SRCS     = param("MAIN_SRCS", ("muse",))            # nguồn xác nhận tính vào PHÂN TÍCH CHÍNH. Cặp có nguồn khác ("alt", "dice") = TẦNG MỞ RỘNG, báo cáo riêng
UPOS_PROJECT  = param("UPOS_PROJECT", True)                 # ngôn ngữ không có tagger (swh): chiếu UPOS từ các ngôn ngữ có tagger qua liên kết Eflomal (Cell A6b)
UPOS_SUBSETS_EXCLUDE_PROJECTED = param("UPOS_SUBSETS_EXCLUDE_PROJECTED", True)    # kiểm tra độ bền theo UPOS (content/function) bỏ ngôn ngữ chỉ có UPOS chiếu (không phải gold)
UPOS_PROJECTED = set()               # được Cell A6b điền: các ngôn ngữ đang dùng UPOS chiếu
PROBLEM_LANGS = param("PROBLEM_LANGS", ("jpn_Jpan", "zho_Hans", "swh_Latn"))   # nhóm ngôn ngữ nghi gây thiên lệch phương pháp: bỏ CÙNG LÚC trong kiểm tra độ bền
LOO_LANGS     = param("LOO_LANGS", None)                 # None = leave-one-language-out cho MỌI ngôn ngữ trong tầng chính; hoặc list, vd. ["jpn_Jpan", "zho_Hans"]
LOO_MIN_WORDS = param("LOO_MIN_WORDS", 500)                  # bỏ qua tập con có ít dạng từ hơn ngưỡng này
MANTEL_SCOPE  = param("MANTEL_SCOPE", "main")               # "main": Mantel chỉ dùng cặp tầng chính (ngôn ngữ thiếu cặp bị loại) | "all": mọi cặp. Cell C6 luôn chạy thêm bản "all" (hậu tố _ext)

# ---- Siêu tham số: bản v5 (sửa theo rà soát độ tin cậy kết quả) ----
MASK_VALID_TRANSLATIONS = param("MASK_VALID_TRANSLATIONS", True)   # B2: che mọi bản dịch HỢP LỆ đã quan sát của từ nguồn (the->le thì la, les không còn là "đối thủ"); False = chỉ che cùng dạng đích (cách cũ)
RR_VARIANT        = param("RR_VARIANT", "valid")      # biến thể RR dùng cho phân tích CHÍNH: "valid" | "strict". Cả hai đều được lưu; biến thể còn lại chạy thành tập con đối chiếu (rr_strict / rr_valid)
USE_TRANS_ENT     = param("USE_TRANS_ENT", True)         # covariate entropy bản dịch (đa dạng bản dịch của từ) trong GAM
EDGE_Q            = param("EDGE_Q", (0.10, 0.90)) # (v5) đầu/đuôi theo PHÂN VỊ DỮ LIỆU của trục x (tiêu chí *_q), thay vì 20% ngoài cùng của lưới x
CLUSTER_BOOT      = param("CLUSTER_BOOT", True)         # bootstrap theo CỤM KHÁI NIỆM (cùng bản dịch tiếng Anh vào/ra mẫu cùng nhau) thay vì từng dòng độc lập
RUN_COVAR_CHECKS  = param("RUN_COVAR_CHECKS", True)         # chạy thêm GAM chỉ-spline, GAM không nsub/độ dài/ngôn ngữ, độ nhạy λ (LAM_SENS) ở tầng waist
LAM_SENS          = param("LAM_SENS", (0.1, 10.0))  # nhân λ (GCV) với các hệ số này để xem kết luận có đổi không (λ cố định trong bootstrap)
SANITY_ON_FAIL    = param("SANITY_ON_FAIL", "fp32_check") # "warn" | "fp32_check" (lệch ở bf16 -> chạy lại sanity bằng fp32) | "abort" (dừng model đó)
SANITY_LANGS      = param("SANITY_LANGS", ["eng_Latn", "zho_Hans"])   # ngôn ngữ dùng cho sanity batch-vs-single (trước đây chỉ eng)
EXTRACT_DTYPE     = param("EXTRACT_DTYPE", "float32")         # None = bf16 trên GPU; "float32" nếu sanity chỉ qua ở fp32 (cần nhiều VRAM hơn; hạ tok_budget trong A2 nếu OOM)
assert RR_VARIANT in ("valid", "strict") and SANITY_ON_FAIL in ("warn", "fp32_check", "abort") and EXTRACT_DTYPE in (None, "bfloat16", "float16", "float32")

# ---- Siêu tham số: mở rộng từ điển xác nhận (v7) — giảm mất từ hiếm ở đuôi ----
LEMMA_LANGS        = param("LEMMA_LANGS", {"eng_Latn", "fra_Latn", "rus_Cyrl", "hin_Deva", "tur_Latn", "arb_Arab", "jpn_Jpan", "fin_Latn"})   # Bước 1: khoá tra phụ = lemma Stanza (Cell A6a). zho/vie không biến cách (lemma = dạng bề mặt); swh/tha không có Stanza
WIKT_DICT          = param("WIKT_DICT", True)        # Bước 2: hợp thêm Wiktionary (kaikki.org) vào từ điển của MỌI ngôn ngữ có MUSE (swh đã dùng Wiktionary làm từ điển chính, nhãn 'alt')
WIKT_CODES         = param("WIKT_CODES", None)        # None = mọi mã MUSE_CODE; hoặc tập con, vd {"fi", "tr", "ru"}
WIKT_KEEP_RAW      = param("WIKT_KEEP_RAW", False)       # giữ tệp JSONL thô sau khi trích (mỗi ngôn ngữ vài trăm MB – vài GB)
WIKT_FORCE_REBUILD = param("WIKT_FORCE_REBUILD", False)       # True = dựng lại mọi từ điển Wiktionary (kể cả swh bản cũ chưa có dạng biến cách)
ITEM_MAX_TIER      = param("ITEM_MAX_TIER", 2)           # bậc xác nhận cao nhất được đưa vào ITEMS: 0 = từ điển gốc (hành vi v5) | 1 = +lemma | 2 = +Wiktionary. Đổi -> chạy lại build_items + C2

# ---- Siêu tham số: truy hồi (M3) ----
CSLS_K       = param("CSLS_K", 10)        # giống Study 1
RETR_CHUNK   = param("RETR_CHUNK", 4096)      # chia khối khi đếm hạng (giới hạn bộ nhớ GPU)
POOL_N       = param("POOL_N", 1000)      # kích thước tập ứng viên cố định để so sánh GIỮA CÁC CẶP (Mantel); thực tế = min(POOL_N, số cặp từ ít nhất)
POOL_REPS    = param("POOL_REPS", 20)        # số lần rút mẫu con cho mỗi cặp

# ---- Siêu tham số: GAM (P-spline), bootstrap, shape, Mantel ----
N_KNOTS      = param("N_KNOTS", 12)        # số nút trong của B-spline bậc 3
LAMBDA_GRID  = 10.0 ** np.arange(-2, 4.01, 0.5)
X_QUANT      = param("X_QUANT", (0.01, 0.99))   # đoạn log-hạng dùng để ước lượng đỉnh (cắt 1% hai đuôi cho spline ổn định)
GAM_QUANT    = param("GAM_QUANT", (0.0, 1.0))     # GAM dùng toàn bộ đoạn x sau khi đã cắt đuôi bằng prep_main (không clip -> không có khối điểm dồn ở biên)
N_BOOT       = param("N_BOOT", 1000)      # bootstrap theo TỪ (đơn vị = một dạng từ trong một ngôn ngữ)
BOOT_CHUNK   = param("BOOT_CHUNK", 250)
CV_FOLDS     = param("CV_FOLDS", 5)
MANTEL_PERMS = param("MANTEL_PERMS", 100_000)
MIN_INST_PER_TYPE = param("MIN_INST_PER_TYPE", 1)

# ---- Điều khiển chạy ----
RUN_MODELS      = param("RUN_MODELS", None)      # None = mọi model đã đăng ký (core + sweep + 2b nếu có); hoặc list key, vd. ["xlmr_large", "qwen3_8b"]
FORCE_RERUN     = param("FORCE_RERUN", False)
KEEP_WORD_CACHE = param("KEEP_WORD_CACHE", False)     # False = xoá cache biểu diễn mức từ sau khi đã lưu kết quả truy hồi (Qwen3-8B ~50GB; trích lại chỉ vài phút)
SANITY_CHECK    = param("SANITY_CHECK", True)      # kiểm tra batch-vs-single 1 lần cho mỗi model
WORD_BUF_GPU_GB = param("WORD_BUF_GPU_GB", 12.0)      # buffer biểu diễn của MỘT ngôn ngữ nằm trên GPU nếu nhỏ hơn ngưỡng này, ngược lại nằm trên CPU

def short_pair(p):
    return PAIR_NAME[p]
