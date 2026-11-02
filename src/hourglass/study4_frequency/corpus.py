# STATE: gộp các nguồn đo thành một corpus song song -> TEXTS, WT
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 7 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ---- Gộp các nguồn ĐO thành MỘT corpus song song (devtest + dev + NTREX), tách từ + UPOS một lần ----
# Câu i của mọi ngôn ngữ vẫn là bản dịch của nhau (mỗi nguồn đã được clean_parallel cùng chỉ số cho mọi ngôn ngữ, nối cùng thứ tự).
# SENT_SRC[i] = chỉ số nguồn của câu i (để kiểm tra lặp lại theo miền: FLORES wiki/du lịch vs NTREX tin tức).
_src = {ds: get_texts(ds, N_SENT) for ds in MEASURE_SETS + ALIGN_EXTRA}
TEXTS = {MEAS: {l: [s for ds in MEASURE_SETS for s in _src[ds][l]] for l in LANGS}}
SENT_SRC = np.concatenate([np.full(len(_src[ds][LANGS[0]]), i, np.int8) for i, ds in enumerate(MEASURE_SETS)])
assert len(SENT_SRC) == len(TEXTS[MEAS][LANGS[0]])
print(f"[data] corpus đo '{MEAS}': {len(SENT_SRC)} câu x {len(LANGS)} ngôn ngữ ({ {ds: int((SENT_SRC == i).sum()) for i, ds in enumerate(MEASURE_SETS)} })")
SPLIT_OF = {ds: f"x_{ds}" for ds in ALIGN_EXTRA}          # nguồn chỉ-gióng-hàng (nếu còn)
WT = {MEAS: {}}
for _l in LANGS:
    WT[MEAS][_l] = get_wordtab(MEAS, _l, TEXTS[MEAS][_l], with_upos=True)
    wt = WT[MEAS][_l]
    n_w = len(wt["norm"]); n_s = len(wt["ptr"]) - 1
    unk = np.mean([u == "UNK" for u in wt["upos"]]) if n_w else 1.0
    print(f"  [words] {SHORT[_l]}: {n_w:,} từ ({n_w / n_s:.1f} từ/câu) | UPOS UNK {unk:.0%} | tách từ: {wt['seg']}")
for _ds in ALIGN_EXTRA:
    TEXTS[_ds] = _src[_ds]; WT[SPLIT_OF[_ds]] = {_l: get_wordtab(SPLIT_OF[_ds], _l, _src[_ds][_l]) for _l in LANGS}
