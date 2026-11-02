# Bảng cấp từ và cấp cặp (Cell C3)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 25 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C3 — Bảng cấp TỪ và cấp CẶP từ kết quả truy hồi (không cần GPU / không cần model)
#   word_table(key) : mỗi dòng = một dạng từ trong một ngôn ngữ; R[:,k] = RR trung bình (qua mọi cặp ngôn ngữ và mọi lần xuất hiện)
#   pair_matrices   : ma trận căn chỉnh 12x12 theo tầng (RR trên tập ứng viên cố định n_pool -> so sánh được giữa các cặp)
# ============================================================
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau

P_ = len(PAIRS)
ITEM_PAIR = np.repeat(np.arange(P_), np.diff(ITEMS["ptr"]))
ITEM_LA = np.array([p[0] for p in PAIRS])[ITEM_PAIR]
ITEM_LB = np.array([p[1] for p in PAIRS])[ITEM_PAIR]
WAIST_SOURCE = "retrieval"       # "retrieval": tầng có R_pool trung bình lớn nhất | "study1": k* của Study 1 (study1/analysis/features_flores.csv)
FUNC_UPOS    = ["ADP", "AUX", "CCONJ", "DET", "PART", "PRON", "SCONJ"]
CONTENT_UPOS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]

def load_run(key):
    rd = run_dir(key)
    z = np.load(f"{rd}/retrieval.npz")
    return {k: z[k] for k in z.files}, read_json(f"{rd}/meta.json", {})

def done_models():
    return [k for k in MODELS if _retr_current(run_dir(k))]

def waist_layer(key, run=None):
    """(k_waist, K1). Mặc định: tầng có R_pool trung bình qua các cặp lớn nhất; hoặc k* của Study 1 nếu có."""
    run = run or load_run(key)[0]
    K1 = int(run["K1"])
    if WAIST_SOURCE == "study1":
        p = f"{ROOT}/study1/analysis/features_flores.csv"
        if os.path.exists(p):
            df = pd.read_csv(p); r = df[df.run == key]
            if len(r):
                return int(r.k_star.iloc[0]), K1
        print(f"  [warn] không có k* của Study 1 cho {key} -> dùng tầng theo truy hồi")
    ok = np.diff(run["pair_ptr"]) >= MIN_PAIR_ITEMS                  # chỉ cặp đủ lớn (cặp Dice nhỏ làm R_pool nhiễu)
    if len(PAIR_MAIN) == len(ok) and PAIR_MAIN.any():
        ok = ok & PAIR_MAIN                                          # v4: chỉ cặp TẦNG CHÍNH (tiêu chí xác nhận đồng nhất)
    P_k = run["POOL_R"][ok] if ok.any() else run["POOL_R"]
    return int(np.argmax(P_k.mean(0))), K1

CONCEPT = None
def _concepts():
    """(v5) Cụm KHÁI NIỆM cho bootstrap: dạng từ ở ngôn ngữ khác được gắn vào dạng tiếng Anh mà nó được gióng/xác nhận NHIỀU NHẤT (trên toàn bộ ITEMS, độc lập với tập con đang phân tích);
    tiếng Anh gắn vào chính nó. -1 = không có đối ứng tiếng Anh (tự thành cụm riêng). Cùng một khái niệm xuất hiện ở nhiều ngôn ngữ nên các dòng của một cụm tương quan với nhau."""
    global CONCEPT
    if CONCEPT is not None:
        return CONCEPT
    conc = {l: np.full(int(TID[l].max()) + 1, -1, np.int64) for l in LANGS}
    conc[LANGS[0]] = np.arange(int(TID[LANGS[0]].max()) + 1, dtype=np.int64)
    for j in range(1, len(LANGS)):
        pi = PAIRS.index((0, j)); s, e = ITEMS["ptr"][pi], ITEMS["ptr"][pi + 1]
        if e == s:
            continue
        en, xx = TID[LANGS[0]][ITEMS["wa"][s:e]], TID[LANGS[j]][ITEMS["wb"][s:e]]
        d = pd.DataFrame({"x": xx, "e": en}).groupby(["x", "e"]).size().rename("c").reset_index()
        d = d.sort_values(["x", "c", "e"], ascending=[True, False, True]).drop_duplicates("x")
        conc[LANGS[j]][d.x.values] = d.e.values
    CONCEPT = conc
    return conc

def word_table(key, pairs=None, src=None, variant=None, max_tier=None):
    """Trả về (df, R [n_rows, K1]). (v7) max_tier: chỉ giữ mục có bậc xác nhận từ điển <= max_tier (0 = từ điển gốc như v5; 1 = +lemma; 2 = +Wiktionary); RR vẫn tính trên toàn bộ ITEMS (cùng tập ứng viên), chỉ đổi từ nào vào phân tích. df: lang, type, n_inst, log_rank, seen, zipf, upos, len, nsub, + (v5) fw_count, trans_ent, n_trans, concept.
    pairs: chỉ dùng các cặp ngôn ngữ có chỉ số trong danh sách này (None = tất cả).
    variant (v5): "valid" = RR với mặt nạ mọi bản dịch hợp lệ (RR_AB_V) | "strict" = chỉ che cùng dạng đích (cách cũ). Mặc định RR_VARIANT."""
    run, meta = load_run(key)
    vi = run["valid_idx"]
    keep = np.ones(len(vi), bool) if pairs is None else np.isin(ITEM_PAIR[vi], list(pairs))
    if src is not None:                                  # chỉ các câu thuộc nguồn MEASURE_SETS[src] (kiểm tra lặp lại theo miền)
        keep &= (SENT_SRC[ITEMS["sent"][vi]] == src)
    if max_tier is not None and "tier" in ITEMS:
        keep &= (ITEMS["tier"][vi] <= max_tier)
    vi = vi[keep]
    lang = np.concatenate([ITEM_LA[vi], ITEM_LB[vi]]); g = np.concatenate([ITEMS["wa"][vi], ITEMS["wb"][vi]])
    variant = variant or RR_VARIANT
    ab, ba = run["RR_AB"], run["RR_BA"]
    if variant == "valid" and "RR_AB_V" in run:
        ab, ba = run["RR_AB_V"], run["RR_BA_V"]
    RR = np.concatenate([ab[:, keep], ba[:, keep]], axis=1).astype(np.float32)          # [K1, 2n]
    K1 = RR.shape[0]
    tid = np.zeros(len(g), np.int64); ntok = np.zeros(len(g), np.float32); upos = np.empty(len(g), dtype=object)
    for li, l in enumerate(LANGS):
        m = lang == li
        tid[m] = TID[l][g[m]]; ntok[m] = run[f"ntok_{SHORT[l]}"][g[m]]
        up = np.asarray(WT[MEAS][l]["upos"], dtype=object); upos[m] = up[g[m]]
    keyv = lang.astype(np.int64) * 10_000_000 + tid
    uniq, first, inv, cnt = np.unique(keyv, return_index=True, return_inverse=True, return_counts=True)
    R = np.stack([np.bincount(inv, weights=RR[k], minlength=len(uniq)) / cnt for k in range(K1)], 1).astype(np.float32)
    nsub = np.bincount(inv, weights=ntok, minlength=len(uniq)) / cnt
    # --- đếm đúng: instance token phân biệt, số cặp ngôn ngữ, độ dễ trung bình của các cặp mà từ tham gia (pmix) ---
    pr = np.concatenate([ITEM_PAIR[vi], ITEM_PAIR[vi]])
    _, fi = np.unique(lang.astype(np.int64) * 100_000_000 + g, return_index=True)
    n_inst = np.bincount(inv[fi], minlength=len(uniq))
    n_pairs = np.bincount(np.unique(inv.astype(np.int64) * 100 + pr) // 100, minlength=len(uniq))
    pool_p = run["POOL_R"].mean(1)                                       # [P] độ dễ của từng cặp (trung bình theo tầng)
    pmix = np.bincount(inv, weights=pool_p[pr], minlength=len(uniq)) / cnt
    # --- (v5) đa dạng bản dịch: entropy (bit) của phân bố từ đối ứng, tính riêng theo từng ngôn ngữ đối tác rồi lấy trung bình ---
    pg = np.concatenate([ITEMS["wb"][vi], ITEMS["wa"][vi]]); pl = np.concatenate([ITEM_LB[vi], ITEM_LA[vi]])
    ptid = np.zeros(len(pg), np.int64)
    for lj, l2 in enumerate(LANGS):
        mm = pl == lj
        if mm.any():
            ptid[mm] = TID[l2][pg[mm]]
    q = pd.DataFrame({"w": inv, "pl": pl, "pt": ptid}).groupby(["w", "pl", "pt"]).size().rename("c").reset_index()
    q["p"] = q.c / q.groupby(["w", "pl"]).c.transform("sum")
    q["h"] = -q.p * np.log2(q.p)
    qq = q.groupby(["w", "pl"]).agg(h=("h", "sum"), nt=("pt", "size")).reset_index()
    qw = qq.groupby("w").agg(h=("h", "mean"), nt=("nt", "mean"))
    trans_ent = qw.h.reindex(np.arange(len(uniq))).fillna(0.0).values
    n_trans = qw.nt.reindex(np.arange(len(uniq))).fillna(1.0).values
    um = pd.DataFrame({"k": inv, "u": upos}).groupby(["k", "u"]).size().reset_index(name="n")
    um = um.sort_values(["k", "n"], ascending=[True, False]).drop_duplicates("k").sort_values("k")
    rows = []
    for u, f in enumerate(first):
        li, gg = int(lang[f]), int(g[f]); l = LANGS[li]
        w = WT[MEAS][l]["norm"][gg]; lr, seen = RANKS[l].log_rank(w)
        rows.append((li, w, int(cnt[u]), lr, seen, RANKS[l].zipf(w), len(w.replace("_", " ")), int(RANKS[l].count.get(w, 0))))
    df = pd.DataFrame(rows, columns=["li", "type", "n_inst", "log_rank", "seen", "zipf", "len", "fw_count"])
    df["lang"] = [SHORT[LANGS[i]] for i in df.li]; df["upos"] = um.u.values; df["nsub"] = nsub
    df["n_inst"] = n_inst; df["n_rows"] = cnt; df["n_pairs"] = n_pairs; df["pmix"] = pmix
    df["trans_ent"] = trans_ent; df["n_trans"] = n_trans
    cm = _concepts(); lang_u, tid_u = lang[first], tid[first]; concept = np.full(len(first), -1, np.int64)
    for li, l in enumerate(LANGS):
        mk = lang_u == li
        if mk.any():
            concept[mk] = cm[l][tid_u[mk]]
    df["concept"] = concept
    df["log_rank_raw"] = df["log_rank"]
    if AXIS == "zipf":
        df["log_rank"] = -df["zipf"]                                     # càng lớn càng hiếm (cùng hướng với hạng)
    m = (df.n_inst >= MIN_INST_PER_TYPE).values
    return df[m].reset_index(drop=True), R[m]

def pair_matrices(run, pairs_ok=None):
    """[K1, L, L] ma trận căn chỉnh đối xứng (NaN ở đường chéo và ở cặp có < MIN_PAIR_ITEMS cặp từ hợp lệ) từ R_pool."""
    L, K1 = len(LANGS), run["POOL_R"].shape[1]
    nv = np.diff(run["pair_ptr"])
    M = np.full((K1, L, L), np.nan, np.float32)
    for pi, (a, b) in enumerate(PAIRS):
        if nv[pi] >= MIN_PAIR_ITEMS and (pairs_ok is None or pairs_ok[pi]):
            M[:, a, b] = M[:, b, a] = run["POOL_R"][pi]
    return M

def pairs_without(pairs, exclude_langs):
    """Các cặp (chỉ số) trong `pairs` KHÔNG chứa ngôn ngữ nào trong exclude_langs (leave-one-language-out đúng nghĩa: loại cả từ của ngôn ngữ đó lẫn vai trò đối tác)."""
    ex = {LANGS.index(l) for l in exclude_langs}
    return [pi for pi in pairs if PAIRS[pi][0] not in ex and PAIRS[pi][1] not in ex]
