# STATE: ITEMS, PAIR_STATS, NEED, PAIR_MAIN, ITEMS_SIG
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 13 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

ensure_lemmas()          # (v7, bước 1) lemma Stanza -> khoá tra từ điển (cache vào words/*.pkl; chạy một lần)
prefetch_wikt()          # (v7, bước 2) Wiktionary (kaikki.org) cho mọi ngôn ngữ (cache ở raw/dict_alt/; chạy một lần)
ITEMS, PAIR_STATS = build_items()
NEED = need_masks(ITEMS)
PAIR_MAIN = np.array([s in MAIN_SRCS for s in PAIR_STATS.confirm_src], bool)      # cặp thuộc TẦNG CHÍNH (tiêu chí xác nhận đồng nhất)
ITEMS_SIG = items_signature(ITEMS)
print(f"[tiers] chính {int(PAIR_MAIN.sum())}/{len(PAIR_MAIN)} cặp (MAIN_SRCS={MAIN_SRCS}) | mở rộng: {PAIR_STATS[~PAIR_MAIN].pair.tolist()} | nguồn từ điển: {({SHORT[l]: dict_origin(l) for l in LANGS})} | items_sig={ITEMS_SIG}")
_n = PAIR_STATS.n_items.values
print(f"[items] {int(_n.sum()):,} cặp từ trên {len(PAIRS)} cặp ngôn ngữ | trung vị {int(np.median(_n))}, min {int(_n.min())}, max {int(_n.max())}")
print("[items] xác nhận:", PAIR_STATS.confirm_src.value_counts().to_dict(), "| cặp < MIN_PAIR_ITEMS:", int(PAIR_STATS.low_items.sum()))
_t = ITEMS["tier"]
print(f"[tiers v7] số mục theo bậc xác nhận: 0 (từ điển gốc, = v5) {int((_t == 0).sum()):,} | 1 (+lemma) {int((_t == 1).sum()):,} | 2 (+Wiktionary) {int((_t == 2).sum()):,} | 3 (Dice) {int((_t == 3).sum()):,}  (ITEM_MAX_TIER={ITEM_MAX_TIER})")
print("[items] số từ cần trích mỗi ngôn ngữ:", {SHORT[l]: int(NEED[l].sum()) for l in LANGS})
PAIR_STATS.head(10)
