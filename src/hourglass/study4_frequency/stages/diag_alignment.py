# Chẩn đoán gióng hàng / từ điển (Cell A7, A7b, A7c, A7d + cell chẩn đoán độ phủ)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 15, 16, 17, 18, 23 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["diag_alignment"]) hoặc `python -m hourglass run <experiment> diag_alignment`.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A7 — CHẨN ĐOÁN gióng hàng / MUSE (chạy sau Cell A6): MUSE phủ bao nhiêu từ của FLORES, theo ngôn ngữ và theo log-hạng?
# Hàng = ngôn ngữ, cột = khoảng log10-hạng; giá trị = tỉ lệ LOẠI từ của FLORES devtest có ít nhất một bản dịch trong MUSE.
# has_dict=False -> không tải được từ điển -> mọi cặp có ngôn ngữ đó rơi xuống Dice. Độ phủ giảm mạnh ở đuôi -> MUSE cắt cụt đuôi tần suất.
# ============================================================
import pandas as pd
def _type_keys(l):
    """{dạng chuẩn: khoá phụ} của một ngôn ngữ (hợp các khoá phụ qua mọi lần xuất hiện)."""
    wt, d = WT[MEAS][l], {}
    if l not in ALT_KEY_LANGS or wt.get("alt") is None:
        return d
    for g, n in enumerate(wt["norm"]):
        if wt["alt"][g]:
            d.setdefault(n, set()).update(wt["alt"][g])
    return {n: tuple(sorted(v - {n})) for n, v in d.items()}

def muse_coverage(bins=(0, 2, 3, 4, 5, 9), use_alt=True):
    rows = []
    for l in LANGS:
        if l == "eng_Latn":
            continue
        norms = list(dict.fromkeys(WT[MEAS][l]["norm"]))
        tk = _type_keys(l) if use_alt else {}
        tr = [to_en_keys(l, (w,) + tuple(tk.get(w, ()))) for w in norms]
        has_dict = any(t is not None for t in tr)
        hit = np.array([bool(t) for t in tr])
        lr = np.array([RANKS[l].log_rank(w)[0] for w in norms])
        b = np.digitize(lr, bins[1:-1])
        for i in range(len(bins) - 1):
            m = b == i
            rows.append(dict(lang=SHORT[l], has_dict=has_dict, bin=f"{bins[i]}-{bins[i + 1]}", n_types=int(m.sum()),
                             muse_hit=float(hit[m].mean()) if m.sum() else np.nan))
    d = pd.DataFrame(rows)
    order = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    cov = d.pivot(index="lang", columns="bin", values="muse_hit")[order].round(2)
    cov.insert(0, "has_dict", d.groupby("lang").has_dict.first())
    return cov.loc[[SHORT[l] for l in LANGS if l != "eng_Latn"]], d

MUSE_COV, _ = muse_coverage()
print(MUSE_COV.to_string())
print("\nCặp theo nguồn xác nhận:", PAIR_STATS.confirm_src.value_counts().to_dict())
print(PAIR_STATS.sort_values("n_items")[["pair", "n_links", "n_after_filters", "n_muse_checked", "n_muse_ok", "n_items", "confirm_src"]].head(15).to_string(index=False))

# ============================================================
# Cell A7b — CHẨN ĐOÁN từ điển theo ngôn ngữ (zho / jpn / swh). Chạy sau A7. Chỉ để KIỂM CHỨNG; kết quả thật nằm ở ITEMS (cell build_items).
#   zho: độ phủ MUSE trước/sau khi đổi phồn -> giản thể | jpn: độ phủ khi chỉ tra dạng bề mặt vs thêm orthBase/lemma + các dạng phổ biến vẫn thiếu
#   swh: nguồn từ điển đang dùng (muse / alt / None) và độ phủ theo hạng. Nếu số liệu ở đây khác kỳ vọng, đổi cấu hình ở A1 rồi chạy lại A6 + build_items.
# ============================================================
def _coverage_by_bin(l, use_alt, bins=(0, 2, 3, 4, 5, 9)):
    norms = list(dict.fromkeys(WT[MEAS][l]["norm"])); tk = _type_keys(l) if use_alt else {}
    tr = [to_en_keys(l, (w,) + tuple(tk.get(w, ()))) for w in norms]
    hit = np.array([bool(t) for t in tr]); lr = np.array([RANKS[l].log_rank(w)[0] for w in norms]); bi = np.digitize(lr, bins[1:-1])
    cov = {f"{bins[i]}-{bins[i + 1]}": (round(float(hit[bi == i].mean()), 2) if (bi == i).any() else np.nan) for i in range(len(bins) - 1)}
    return cov, norms, hit, lr

def diag_dict_keys(l="jpn_Jpan", n_missing=40):
    """Độ phủ từ điển theo log10-hạng: tra dạng bề mặt vs thêm khoá phụ; liệt kê các dạng phổ biến nhất vẫn KHÔNG có trong từ điển."""
    c0, *_ = _coverage_by_bin(l, False); c1, norms, hit, lr = _coverage_by_bin(l, True)
    print(f"[{SHORT[l]}] nguồn từ điển: {dict_origin(l)}"); print(pd.DataFrame({"bề mặt": c0, "bề mặt + khoá phụ": c1}).T.to_string())
    order = np.argsort(lr); print(f"  {n_missing} dạng phổ biến nhất vẫn thiếu:", [norms[i] for i in order if not hit[i]][:n_missing])

def diag_zh_t2s():
    """Độ phủ zho khi nạp MUSE zh nguyên bản vs đã đổi giản thể (tạm đổi cờ, rồi khôi phục + xoá cache từ điển)."""
    global MUSE_ZH_T2S
    orig, out = MUSE_ZH_T2S, {}
    try:
        for flag in (False, True):
            MUSE_ZH_T2S = flag; _MUSE.clear(); _EN_INV.clear(); _DICTS.clear()
            out["đã đổi giản thể" if flag else "nguyên bản"] = _coverage_by_bin("zho_Hans", False)[0]
    finally:
        MUSE_ZH_T2S = orig; _MUSE.clear(); _EN_INV.clear(); _DICTS.clear()
    print("[zho] độ phủ MUSE theo log10-hạng"); print(pd.DataFrame(out).T.to_string())

diag_zh_t2s()
diag_dict_keys("jpn_Jpan")
print("[swh] nguồn từ điển:", dict_origin("swh_Latn"))
if dict_origin("swh_Latn"):
    print(pd.DataFrame({"swh": _coverage_by_bin("swh_Latn", False)[0]}).T.to_string())

# ============================================================
# Cell A7c — (v5) CHẨN ĐOÁN thành phần theo hạng sau các bộ lọc (chạy sau build_items / A6, cần WT, RANKS, ITEMS)
#   MUSE phủ giảm mạnh theo hạng (cell A7) nên các từ hiếm còn lại có thể là loại dịch "sạch"; DROP_SAME_FORM thì bỏ các mục dễ (tên riêng, từ mượn) vốn tập trung ở đuôi.
#   Hai bộ lọc kéo đuôi về hai hướng ngược nhau -> hình dạng đuôi phụ thuộc cách lọc. Bảng dưới cho thấy mỗi khoảng log10-hạng giữ lại bao nhiêu DẠNG TỪ:
#     keep_filters = tỉ lệ dạng từ có liên kết Eflomal còn sau DROP_NUMERIC / DROP_SAME_FORM ; keep_final = tỉ lệ còn trong ITEMS (sau xác nhận từ điển).
# ============================================================
def selection_diag(bins=(0, 2, 3, 4, 5, 9)):
    rows = []
    for li, l in enumerate(LANGS):
        cand, kept, conf = set(), set(), set()
        for pi, (a, b) in enumerate(PAIRS):
            if li not in (a, b):
                continue
            z = np.load(f"{WORK_DIR}/align/{PAIR_NAME[pi]}/links.npz"); ga, gb = z["wa"], z["wb"]
            if not len(ga):
                continue
            wta, wtb = WT[MEAS][LANGS[a]], WT[MEAS][LANGS[b]]
            na = np.asarray(wta["norm"], dtype=object)[ga]; nb = np.asarray(wtb["norm"], dtype=object)[gb]
            ok = np.ones(len(ga), bool)
            if DROP_NUMERIC:
                ok &= ~(wta["is_num"][ga] | wtb["is_num"][gb])
            if DROP_SAME_FORM:
                ok &= (na != nb)
            mine = na if li == a else nb
            cand.update(mine.tolist()); kept.update(mine[ok].tolist())
            s, e = ITEMS["ptr"][pi], ITEMS["ptr"][pi + 1]
            g_it = ITEMS["wa"][s:e] if li == a else ITEMS["wb"][s:e]
            conf.update(np.asarray(WT[MEAS][l]["norm"], dtype=object)[g_it].tolist())
        ws = [w for w in cand if not _ALL_NUM.match(w) and RANKS[l].log_rank(w)[1]]
        lr = np.array([RANKS[l].log_rank(w)[0] for w in ws]); bi = np.digitize(lr, bins[1:-1])
        kf = np.array([w in kept for w in ws]); kc = np.array([w in conf for w in ws])
        for i in range(len(bins) - 1):
            m = bi == i
            rows.append(dict(lang=SHORT[l], bin=f"{bins[i]}-{bins[i + 1]}", n_cand=int(m.sum()),
                             keep_filters=float(kf[m].mean()) if m.any() else np.nan, keep_final=float(kc[m].mean()) if m.any() else np.nan))
    d = pd.DataFrame(rows); order = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    for col in ("n_cand", "keep_filters", "keep_final"):
        print(f"\n[{col}] theo ngôn ngữ x log10-hạng"); print(d.pivot(index="lang", columns="bin", values=col)[order].round(2).to_string())
    return d

SELECTION = selection_diag()

# ============================================================
# Cell A7d — (v7, bước 3) KIỂM CHỨNG mở rộng từ điển (lemma + Wiktionary): thêm được bao nhiêu từ hiếm, và các mục thêm có ĐÚNG không?
# Chạy sau build_items (A6) + A6b. Không cần GPU. Chỉ đọc WT, RANKS, ITEMS, links.npz.
#   (1) tail_gain()    : tỉ lệ DẠNG TỪ có liên kết Eflomal (sau DROP_*) còn trong ITEMS theo ngôn ngữ x khoảng log10-hạng — chỉ bậc 0 (= v5) vs mọi bậc.
#   (2) tier_quality() : ĐỘ ĐÚNG của từng nhóm liên kết theo khoảng hạng (hạng = của từ HIẾM hơn trong cặp). Không có gold nên dùng tín hiệu ĐỘC LẬP với từ điển:
#        tỉ lệ khớp LỚP TỪ thô (N / V / A / R / NUM / chức năng) giữa hai từ được gióng. UPOS do Stanza/PyThaiNLP gán riêng từng ngôn ngữ;
#        ngôn ngữ có UPOS CHIẾU qua liên kết (swh) bị loại khỏi phép đo này vì vòng tròn. `chance` = tỉ lệ khớp khi hoán vị ngẫu nhiên đối tác trong cùng ô.
#        Nhóm: t0 = từ điển gốc (= v5) | t1 = chỉ khi thêm lemma | t2 = chỉ khi thêm Wiktionary | new = t1 + t2 | rej = cả hai từ có trong từ điển nhưng KHÔNG chung bản dịch | unk = có từ không có trong từ điển.
#   Cách đọc:  new ≈ t0 (trong dung sai) và >> chance  -> phần mở rộng thêm cặp đúng, không chỉ thêm nhiễu.
#              unk ≈ t0                                -> liên kết Eflomal ngoài từ điển ở khoảng hạng đó cũng đáng tin (cơ sở để cân nhắc kéo vào đuôi, quyết định để người dùng).
#              rej >> chance                           -> từ điển còn thiếu bản dịch: nhiều liên kết bị loại vẫn đúng.
#   Giới hạn: lớp từ thô chỉ là bằng chứng gián tiếp (N<->V khi dịch thoáng, PROPN/NOUN đã gộp); không thay cho kiểm tay một mẫu nhỏ ở đuôi (xem sample_links()).
# ============================================================
import pandas as pd

_RB  = (0, 2, 3, 4, 5, 9)                                    # khoảng log10-hạng (giống A7/A7c)
_CC  = {"NOUN": 0, "PROPN": 0, "VERB": 1, "AUX": 1, "ADJ": 2, "ADV": 3, "NUM": 4,
        "ADP": 5, "CCONJ": 5, "SCONJ": 5, "PART": 5, "DET": 5, "PRON": 5}        # lớp từ thô; INTJ/SYM/X/UNK -> -1 (không dùng)
_GRP = ["t0", "t1", "t2", "rej", "unk"]
_LR_CACHE = {}

def _lr_tokens(l):
    """(log10-hạng, đã thấy trong FineWeb-2) cho từng token của bảng từ ngôn ngữ l."""
    if l not in _LR_CACHE:
        u, inv = np.unique(np.asarray(WT[MEAS][l]["norm"], dtype=object).astype(str), return_inverse=True)
        lr, sn = np.empty(len(u)), np.empty(len(u), bool)
        for i, w in enumerate(u):
            lr[i], sn[i] = RANKS[l].log_rank(w)
        _LR_CACHE[l] = (lr[inv], sn[inv])
    return _LR_CACHE[l]

def _upos_codes(l):
    up = WT[MEAS][l]["upos"]
    if up is None or l in UPOS_PROJECTED:
        return None
    return np.array([_CC.get(u, -1) for u in up], np.int8)

def _main_ok(pi):
    return not (len(PAIR_MAIN) == len(PAIRS) and PAIR_MAIN.any() and not PAIR_MAIN[pi])

def link_frame(main_only=True):
    """DataFrame mỗi dòng = một liên kết Eflomal (sau DROP_*, cả hai từ đã thấy trong FineWeb-2): pair, rare_lang, rmax, bin, grp, ua, ub (lớp từ; -1 = không dùng)."""
    if CONFIRM_MODE != "muse":
        print("[A7d] CONFIRM_MODE != 'muse' -> không có bậc xác nhận để kiểm"); return None
    LR = {l: _lr_tokens(l) for l in LANGS}; UC = {l: _upos_codes(l) for l in LANGS}; parts = []
    for pi, (a, b) in enumerate(PAIRS):
        if main_only and not _main_ok(pi):
            continue
        la, lb = LANGS[a], LANGS[b]
        L = link_tiers(pi, WT[MEAS][la], WT[MEAS][lb]); k = L["keep"]
        if not k.any():
            continue
        ga, gb = L["wa"][k], L["wb"][k]
        ra, sa, rb, sb = LR[la][0][ga], LR[la][1][ga], LR[lb][0][gb], LR[lb][1][gb]
        m = sa & sb
        t, kn = L["tier"][k][m], L["known"][k][m]
        grp = np.where(t == 0, 0, np.where(t == 1, 1, np.where(t == 2, 2, np.where(kn, 3, 4)))).astype(np.int8)
        rmax = np.maximum(ra, rb)[m].astype(np.float32)
        ua = UC[la][ga][m] if UC[la] is not None else np.full(int(m.sum()), -1, np.int8)
        ub = UC[lb][gb][m] if UC[lb] is not None else np.full(int(m.sum()), -1, np.int8)
        parts.append(pd.DataFrame(dict(pair=np.full(int(m.sum()), pi, np.int16), rare_lang=np.where(ra >= rb, a, b)[m].astype(np.int8), rmax=rmax,
                                       bin=np.digitize(rmax, _RB[1:-1]).astype(np.int8), grp=grp, ua=ua, ub=ub)))
    return pd.concat(parts, ignore_index=True) if parts else None

def tail_gain(main_only=True):
    """Tỉ lệ DẠNG TỪ (có liên kết Eflomal sau DROP_*) còn trong ITEMS, theo ngôn ngữ x khoảng log10-hạng: chỉ bậc 0 (= v5) vs mọi bậc. Cùng định nghĩa với A7c (selection_diag)."""
    if "tier" not in ITEMS:
        print("[A7d] ITEMS chưa có 'tier' -> chạy lại cell build_items"); return None
    rows = []
    for li, l in enumerate(LANGS):
        norms = np.asarray(WT[MEAS][l]["norm"], dtype=object); cand, k_all, k_t0 = set(), set(), set()
        for pi, (a, b) in enumerate(PAIRS):
            if li not in (a, b) or (main_only and not _main_ok(pi)):
                continue
            L = link_tiers(pi, WT[MEAS][LANGS[a]], WT[MEAS][LANGS[b]]); k = L["keep"]
            cand.update(norms[(L["wa"] if li == a else L["wb"])[k]].tolist())
            s, e = ITEMS["ptr"][pi], ITEMS["ptr"][pi + 1]
            gi = (ITEMS["wa"][s:e] if li == a else ITEMS["wb"][s:e]); ti = ITEMS["tier"][s:e]
            k_all.update(norms[gi].tolist()); k_t0.update(norms[gi[ti == 0]].tolist())
        ws = [w for w in cand if not _ALL_NUM.match(w) and RANKS[l].log_rank(w)[1]]
        lr = np.array([RANKS[l].log_rank(w)[0] for w in ws]); bi = np.digitize(lr, _RB[1:-1])
        f0 = np.array([w in k_t0 for w in ws]); fa = np.array([w in k_all for w in ws])
        for i in range(len(_RB) - 1):
            m = bi == i
            rows.append(dict(lang=SHORT[l], bin=f"{_RB[i]}-{_RB[i + 1]}", n_cand=int(m.sum()),
                             keep_t0=float(f0[m].mean()) if m.any() else np.nan, keep_all=float(fa[m].mean()) if m.any() else np.nan))
    d = pd.DataFrame(rows); d["gain"] = d.keep_all - d.keep_t0
    order = [f"{_RB[i]}-{_RB[i + 1]}" for i in range(len(_RB) - 1)]; langs = [SHORT[l] for l in LANGS]
    for col, ttl in (("n_cand", "số dạng từ ứng viên"), ("keep_t0", "tỉ lệ còn lại, chỉ bậc 0 (= v5)"), ("keep_all", "tỉ lệ còn lại, mọi bậc (v7)"), ("gain", "tăng (v7 − v5)")):
        print(f"\n[{col}] {ttl} — ngôn ngữ x log10-hạng"); print(d.pivot(index="lang", columns="bin", values=col)[order].reindex(langs).round(3).to_string())
    return d

def tier_quality(fr, tol=0.05, min_n=200, seed=SEED):
    """Độ nhất quán lớp từ giữa hai từ được gióng, theo khoảng hạng (của từ hiếm hơn) x nhóm. Trả về DataFrame dài."""
    rng = np.random.default_rng(seed)
    b, g, ua, ub = fr.bin.values, fr.grp.values, fr.ua.values, fr.ub.values
    views = [("t0", (0,)), ("t1", (1,)), ("t2", (2,)), ("new", (1, 2)), ("rej", (3,)), ("unk", (4,))]
    cells_ = [(f"{_RB[i]}-{_RB[i + 1]}", b == i) for i in range(len(_RB) - 1)] + [("tail>=4", b >= 3)]
    rows = []
    for lab, mb in cells_:
        for vn, gs in views:
            m = mb & np.isin(g, gs); inf = m & (ua >= 0) & (ub >= 0); ni = int(inf.sum())
            if ni:
                x, y = ua[inf], ub[inf]; ag, ch = float((x == y).mean()), float((x == rng.permutation(y)).mean())
            else:
                ag = ch = np.nan
            rows.append(dict(bin=lab, grp=vn, n_links=int(m.sum()), n_upos=ni, upos_agree=ag, chance=ch))
    q = pd.DataFrame(rows); order = [c[0] for c in cells_]; cols = [v[0] for v in views]
    pv = lambda v: q.pivot(index="bin", columns="grp", values=v)[cols].reindex(order)
    print("\n[số liên kết] theo khoảng log10-hạng của từ hiếm hơn x nhóm (new = t1 + t2 nên chồng lên hai cột đó)"); print(pv("n_links").astype(int).to_string())
    print("\n[tỉ lệ khớp lớp từ] (chỉ liên kết có UPOS gold ở cả hai phía)"); print(pv("upos_agree").round(3).to_string())
    ch = q.groupby("bin").chance.mean().reindex(order)
    print("\n[chance] tỉ lệ khớp khi hoán vị ngẫu nhiên đối tác (mức nền):", {k: round(float(v), 3) for k, v in ch.items()})
    n = pv("n_links"); known = n[["t0", "t1", "t2", "rej"]].sum(axis=1)
    print("\n[phủ của từ điển trên các liên kết nó PHÁN XÉT được (cả hai từ có trong từ điển)] t0 = v5, v7 = t0+t1+t2")
    print(pd.DataFrame({"known": known.astype(int), "confirmed_v5": (n.t0 / known).round(3), "confirmed_v7": ((n.t0 + n.t1 + n.t2) / known).round(3)}).to_string())
    ag, nu = pv("upos_agree"), pv("n_upos")
    inf0 = (g == 0) & (ua >= 0) & (ub >= 0); pooled_t0 = float((ua[inf0] == ub[inf0]).mean()) if inf0.any() else np.nan       # t0 gộp mọi khoảng hạng: tham chiếu dự phòng khi t0 ở khoảng đó quá thưa
    print(f"\n[kết luận gợi ý] dung sai {tol}; cần n_upos >= {min_n} ở nhóm được so; khi t0 thưa ở khoảng đó thì tham chiếu = t0 gộp mọi khoảng ({pooled_t0:.3f}), ghi chú '*'")
    for lab in order:
        r0, nw, un, rj = ag.loc[lab, "t0"], ag.loc[lab, "new"], ag.loc[lab, "unk"], ag.loc[lab, "rej"]
        thin = not (nu.loc[lab, "t0"] >= min_n) or np.isnan(r0); r = pooled_t0 if thin else r0
        def _cmp(x, nx):
            if nx < min_n or np.isnan(x) or np.isnan(r):
                return "n nhỏ -> chưa kết luận"
            return f"Δ={x - r:+.3f}{'*' if thin else ''} " + ("(đạt)" if x >= r - tol else "(THẤP hơn bậc gốc: kiểm lại)")
        print(f"  [{lab}] t0={r0:.3f} | new={nw:.3f} {_cmp(nw, nu.loc[lab, 'new'])} | unk={un:.3f} {_cmp(un, nu.loc[lab, 'unk'])} | rej={rj:.3f} | chance={ch[lab]:.3f}")
    return q

def tail_by_lang(fr, min_bin=3):
    """Đuôi (log10-hạng của từ hiếm hơn >= _RB[min_bin]) theo ngôn ngữ của từ hiếm hơn: số liên kết và tỉ lệ khớp lớp từ của t0 / new / unk."""
    rows = []
    for li, l in enumerate(LANGS):
        m0 = (fr.rare_lang.values == li) & (fr.bin.values >= min_bin)
        for vn, gs in (("t0", (0,)), ("new", (1, 2)), ("unk", (4,))):
            m = m0 & np.isin(fr.grp.values, gs); inf = m & (fr.ua.values >= 0) & (fr.ub.values >= 0)
            rows.append(dict(lang=SHORT[l], grp=vn, n_links=int(m.sum()), agree=float((fr.ua.values[inf] == fr.ub.values[inf]).mean()) if inf.any() else np.nan))
    d = pd.DataFrame(rows); langs = [SHORT[l] for l in LANGS]
    print(f"\n[đuôi theo ngôn ngữ] log10-hạng >= {_RB[min_bin]}: số liên kết"); print(d.pivot(index="lang", columns="grp", values="n_links")[["t0", "new", "unk"]].reindex(langs).to_string())
    print(f"[đuôi theo ngôn ngữ] tỉ lệ khớp lớp từ (NaN = không có UPOS gold, vd swh)"); print(d.pivot(index="lang", columns="grp", values="agree")[["t0", "new", "unk"]].reindex(langs).round(3).to_string())
    return d

def sample_links(grp="new", lang="fin_Latn", min_log_rank=4.0, n=30, seed=SEED):
    """Mẫu liên kết để KIỂM TAY (nhìn bằng mắt): n cặp (từ hiếm, đối tác, bậc) ở đuôi cho nhóm `grp` ('t0','t1','t2','new'=t1+t2,'rej','unk'), từ hiếm hơn thuộc `lang`. In ra và trả về DataFrame."""
    want = {"t0": (0,), "t1": (1,), "t2": (2,), "new": (1, 2), "rej": (3,), "unk": (4,)}[grp]
    li, rows = LANGS.index(lang), []
    for pi, (a, b) in enumerate(PAIRS):
        if li not in (a, b) or not _main_ok(pi):
            continue
        la, lb = LANGS[a], LANGS[b]; wa_, wb_ = WT[MEAS][la], WT[MEAS][lb]; L = link_tiers(pi, wa_, wb_); k = np.where(L["keep"])[0]
        ra, sa = _lr_tokens(la); rb, sb = _lr_tokens(lb)
        for q in k:
            ga, gb = L["wa"][q], L["wb"][q]
            if not (sa[ga] and sb[gb]):
                continue
            t = int(L["tier"][q]); g_ = 0 if t == 0 else 1 if t == 1 else 2 if t == 2 else (3 if L["known"][q] else 4)
            if g_ not in want:
                continue
            rare_is_a = ra[ga] >= rb[gb]
            if (la if rare_is_a else lb) != lang or max(ra[ga], rb[gb]) < min_log_rank:
                continue
            rows.append(dict(pair=PAIR_NAME[pi], rare=wa_["norm"][ga] if rare_is_a else wb_["norm"][gb], partner=wb_["norm"][gb] if rare_is_a else wa_["norm"][ga], tier=t))
    d = pd.DataFrame(rows)
    if len(d):
        d = d.sample(min(n, len(d)), random_state=seed).reset_index(drop=True)
    print(d.to_string(index=False) if len(d) else "(không có liên kết thoả điều kiện)")
    return d

FR = link_frame()
TIER_GAIN = tail_gain()
TIER_QUALITY = tier_quality(FR) if FR is not None else None
TIER_TAIL_LANG = tail_by_lang(FR) if FR is not None else None

# số dạng từ trong mỗi khoảng (để biết đuôi còn bao nhiêu từ)
_, d = muse_coverage(); print(d.pivot(index="lang", columns="bin", values="n_types"))

# tiếng Việt: từ ghép (có "_") vs từ đơn
l = "vie_Latn"; ws = list(dict.fromkeys(WT[MEAS][l]["norm"]))
for nm, f in (("từ ghép", lambda w: "_" in w), ("từ đơn", lambda w: "_" not in w)):
    sub = [w for w in ws if f(w)]
    print(nm, len(sub), np.mean([bool(to_en(l, w)) for w in sub]))

# tiếng Trung: MUSE giản thể hay phồn thể?
print(open(f"{DATA_DIR}/raw/muse/zh-en.txt", encoding="utf-8").read().split("\n")[1000:1015])
