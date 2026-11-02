# STATE: đếm tần suất + nạp RANKS
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 10 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

RANKS.clear()   # bắt buộc sau khi đổi RankTable
# ---- Đếm tần suất (một lần/ngôn ngữ; 200M từ nên chạy nền, có thể mất vài giờ/ngôn ngữ ở bộ tách từ chậm) ----
FW_KEEP.update({_l: set(WT[MEAS][_l]["norm"]) for _l in LANGS})
for _l in LANGS:
    build_freq(_l)
load_ranks()
print({SHORT[l]: f"V={RANKS[l].V:,}" for l in LANGS})
# Chẩn đoán độ phủ: tỉ lệ DẠNG TỪ của corpus đo không thấy trong mẫu tần suất (bị DROP_UNSEEN loại). Cao (>2-3%) -> tăng FW_MAX_WORDS_PER_LANG cho ngôn ngữ đó.
_un = {}
for _l in LANGS:
    _ty = [w for w in set(WT[MEAS][_l]["norm"]) if not _ALL_NUM.match(w)]
    _un[SHORT[_l]] = f"{np.mean([w not in RANKS[_l].rank for w in _ty]):.1%}"
print("[freq] unseen:", _un)
