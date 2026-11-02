# Chiếu UPOS cho ngôn ngữ không có tagger (Cell A6b)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 14 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A6b — Chiếu UPOS cho ngôn ngữ không có tagger (swh) qua liên kết Eflomal (chạy sau build_items; chạy lại mỗi phiên vì WT nạp lại từ cache UNK)
#   Vì sao: UPOS swh = UNK 100% -> cột upos:UNK trong GAM trùng hệt cột lang:swh (cộng tuyến), và swh không vào được các kiểm tra theo UPOS.
#   Cách làm: với mỗi từ swh, bỏ phiếu theo UPOS (gold: Stanza/PyThaiNLP) của từ được gióng cùng ở các ngôn ngữ có tagger (liên kết giao hai chiều, mọi cặp
#   chứa swh, KHÔNG phụ thuộc từ điển); tiếng Anh nặng 1.5, ngôn ngữ khác nặng 1. Từ không có liên kết nào giữ UNK. UPOS chiếu kém chính xác hơn gold:
#   được ghi vào UPOS_PROJECTED và (UPOS_SUBSETS_EXCLUDE_PROJECTED) bị loại khỏi kiểm tra content/function; vẫn dùng làm covariate trong GAM.
# ============================================================
UPOS_PIVOT_ORDER = ["eng_Latn", "fra_Latn", "rus_Cyrl", "tur_Latn", "fin_Latn", "arb_Arab", "hin_Deva", "vie_Latn", "zho_Hans", "jpn_Jpan", "tha_Thai"]
import pandas as pd
def _has_gold_upos(l):
    u = WT[MEAS][l]["upos"]
    return u is not None and len(u) > 0 and np.mean([x != "UNK" for x in u]) > 0.5

def project_upos(lang, pivots=None, w_eng=1.5):
    wt = WT[MEAS][lang]; N = len(wt["norm"]); tid = {u: i for i, u in enumerate(UPOS_TAGS)}
    votes = np.zeros((N, len(UPOS_TAGS)), np.float32); li = LANGS.index(lang); used = []
    for pv in (pivots or UPOS_PIVOT_ORDER):
        if pv == lang or not _has_gold_upos(pv):
            continue
        pj = LANGS.index(pv); pair = (min(li, pj), max(li, pj)); pi = PAIRS.index(pair)
        z = np.load(f"{WORK_DIR}/align/{PAIR_NAME[pi]}/links.npz")
        g_l, g_p = (z["wa"], z["wb"]) if li == pair[0] else (z["wb"], z["wa"])
        tag = np.array([tid.get(u, tid["X"]) for u in WT[MEAS][pv]["upos"]])[g_p]
        ok = tag != tid["UNK"]
        np.add.at(votes, (g_l[ok], tag[ok]), w_eng if pv == "eng_Latn" else 1.0); used.append(SHORT[pv])
    has = votes.sum(1) > 0
    up = np.where(has, votes.argmax(1), tid["UNK"])
    wt["upos"] = [UPOS_TAGS[i] for i in up]
    return dict(lang=SHORT[lang], pivots=used, coverage=float(has.mean()))

if UPOS_PROJECT:
    UPOS_PROJECTED.clear()
    for _l in LANGS:
        if _l != "eng_Latn" and WT[MEAS][_l]["upos"] is not None and np.all(np.asarray(WT[MEAS][_l]["upos"], dtype=object) == "UNK"):
            _r = project_upos(_l); UPOS_PROJECTED.add(_l)
            _d = pd.Series(WT[MEAS][_l]["upos"]).value_counts(normalize=True).round(3).head(8).to_dict()
            print(f"[upos] {_r['lang']}: chiếu từ {_r['pivots']} | có UPOS cho {_r['coverage']:.0%} số token | phân bố: {_d}")
    if not UPOS_PROJECTED:
        print("[upos] không có ngôn ngữ nào cần chiếu UPOS")
