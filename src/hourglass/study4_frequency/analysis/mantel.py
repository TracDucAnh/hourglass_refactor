# Mantel: căn chỉnh vs khoảng cách loại hình (Cell C6)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 33 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C6 — Ngôn ngữ nào được chia sẻ? Mantel giữa ma trận căn chỉnh và khoảng cách loại hình (lang2vec/URIEL)
#   Căn chỉnh A_ab(k) = R_pool (RR trên tập ứng viên cố định) -> kỳ vọng ÂM so với khoảng cách loại hình.
#   Hai phép hoán vị: "full" (mọi nhãn ngôn ngữ) và "block" (giữ nguyên phân hoạch họ ngôn ngữ: khối cùng cỡ hoán vị cho nhau,
#   trong khối xáo trộn thành viên) -> kiểm tra khoảng cách loại hình có giải thích căn chỉnh NGOÀI cấu trúc họ ngôn ngữ hay không.
# ============================================================
FAMILY = {"eng_Latn": "Indo-European", "fra_Latn": "Indo-European", "rus_Cyrl": "Indo-European", "hin_Deva": "Indo-European",
          "zho_Hans": "Sino-Tibetan", "vie_Latn": "Austroasiatic", "tur_Latn": "Turkic", "arb_Arab": "Afro-Asiatic",
          "jpn_Jpan": "Japonic", "fin_Latn": "Uralic", "swh_Latn": "Niger-Congo", "tha_Thai": "Kra-Dai"}
L2V_CODE = {"eng_Latn": "eng", "fra_Latn": "fra", "rus_Cyrl": "rus", "hin_Deva": "hin", "zho_Hans": "cmn", "vie_Latn": "vie",
            "tur_Latn": "tur", "arb_Arab": "arb", "jpn_Jpan": "jpn", "fin_Latn": "fin", "swh_Latn": "swh", "tha_Thai": "tha"}   # [XM]
TYPO_SETS = {"syntactic": "syntax_knn", "genetic": "fam", "geographic": "geo",
             "featural": "syntax_knn+phonology_knn+inventory_knn"}

def typology_distances(langs=LANGS):
    """{loại: ma trận khoảng cách [L,L]} từ vector đặc trưng URIEL (khoảng cách cosine; địa lý: Euclid trên vector 'geo')."""
    import lang2vec.lang2vec as l2v
    out = {}
    for kind, fs in TYPO_SETS.items():
        try:
            f = l2v.get_features([L2V_CODE[l] for l in langs], fs)
            V = np.array([np.asarray(f[L2V_CODE[l]], dtype=float) for l in langs])
            if kind == "geographic":
                D = np.linalg.norm(V[:, None] - V[None], axis=-1)
            else:
                Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
                D = 1.0 - Vn @ Vn.T
            np.fill_diagonal(D, 0.0); out[kind] = D
        except Exception as e:
            print(f"  [warn] khoảng cách loại hình '{kind}' lỗi: {repr(e)[:120]}")
    return out

def full_perms(L, n, rng):
    return np.argsort(rng.random((n, L)), axis=1).astype(np.int16)

def block_perms(fam, n, rng):
    """fam: list nhãn họ cho L ngôn ngữ. perm[:, i] = ngôn ngữ đặt ở vị trí i; giữ nguyên phân hoạch họ."""
    L = len(fam); blocks = defaultdict(list)
    for i, f in enumerate(fam):
        blocks[f].append(i)
    by_size = defaultdict(list)
    for f, m in blocks.items():
        by_size[len(m)].append(m)
    perm = np.zeros((n, L), np.int16)
    for s, bl in by_size.items():
        M = np.array(bl)                                           # [m, s]
        bp = np.argsort(rng.random((n, len(bl))), axis=1)          # khối i nhận thành viên của khối bp[:, i]
        for i, m in enumerate(bl):
            tgt = M[bp[:, i]]                                      # [n, s]
            sh = np.argsort(rng.random((n, s)), axis=1)
            perm[:, m] = np.take_along_axis(tgt, sh, axis=1)
    return perm

def _rank_mat(A, iu):
    r = rankdata(A[iu]); R = np.zeros_like(A); R[iu] = r; R = R + R.T
    return R, r

def mantel(A, T, perms):
    """Spearman giữa phần tam giác trên của A và T; p hai phía và một phía (kỳ vọng âm) bằng hoán vị nhãn `perms` [n,L]."""
    L = A.shape[0]; iu = np.triu_indices(L, 1)
    Rm, ra = _rank_mat(A, iu); tc = rankdata(T[iu]) - (len(iu[0]) + 1) / 2
    rc = ra - (len(ra) + 1) / 2
    r_obs = float((rc * tc).sum() / np.sqrt((rc ** 2).sum() * (tc ** 2).sum()))
    ia, ib = perms[:, iu[0]], perms[:, iu[1]]
    vals = Rm[ia, ib] - (len(ra) + 1) / 2
    rp = (vals * tc[None]).sum(1) / np.sqrt((rc ** 2).sum() * (tc ** 2).sum())
    n = len(perms)
    return dict(r=r_obs, p_two=float((1 + (np.abs(rp) >= abs(r_obs) - 1e-12).sum()) / (n + 1)),
                p_neg=float((1 + (rp <= r_obs + 1e-12).sum()) / (n + 1)))

def mantel_r_layers(M, T):
    """r Spearman theo tầng (không hoán vị) giữa M[k] và T; M [K1,L,L] có thể chứa NaN (bỏ ô thiếu)."""
    L = T.shape[0]; iu = np.triu_indices(L, 1); t = T[iu]
    out = np.full(M.shape[0], np.nan)
    for k in range(M.shape[0]):
        a = M[k][iu]; ok = np.isfinite(a)
        if ok.sum() >= 5:
            out[k] = spearmanr(a[ok], t[ok])[0]
    return out

def drop_nan_langs(A):
    """Bỏ lần lượt ngôn ngữ có nhiều ô NaN nhất (ngoài đường chéo) cho tới khi ma trận đầy đủ. Trả về chỉ số giữ lại."""
    keep = list(range(A.shape[0]))
    while True:
        sub = A[np.ix_(keep, keep)].copy(); np.fill_diagonal(sub, 0.0)
        bad = np.isnan(sub).sum(1)
        if bad.max() == 0:
            return keep
        keep.pop(int(bad.argmax()))

def analyze_mantel(key, n_perm=MANTEL_PERMS, seed=SEED, scope=None, tag=""):
    """scope='main': chỉ cặp tầng chính (ngôn ngữ thiếu cặp, vd. swh khi dùng Dice/alt, bị drop_nan_langs loại) | 'all': mọi cặp. tag: hậu tố tên tệp (bản mở rộng dùng '_ext')."""
    scope = MANTEL_SCOPE if scope is None else scope
    run, meta = load_run(key); kw, K1 = waist_layer(key, run)
    M = pair_matrices(run, PAIR_MAIN if (scope == "main" and PAIR_MAIN.any()) else None); T = typology_distances()
    A = M[kw]; keep = drop_nan_langs(A)
    if len(keep) < len(LANGS):
        print(f"  [warn] {key}: bỏ {[SHORT[LANGS[i]] for i in range(len(LANGS)) if i not in keep]} khỏi Mantel (cặp thiếu dữ liệu)")
    sub = np.ix_(keep, keep); fam = [FAMILY[LANGS[i]] for i in keep]
    rng = np.random.default_rng(seed)
    pf, pb = full_perms(len(keep), n_perm, rng), block_perms(fam, n_perm, rng)
    rows = []
    for kind, D in T.items():
        for pk, pm in (("full", pf), ("block", pb)):
            r = mantel(A[sub], D[sub], pm); r.update(key=key, typology=kind, perm=pk, layer=kw, n_langs=len(keep)); rows.append(r)
    layer_r = pd.DataFrame({kind: mantel_r_layers(M[:, keep][:, :, keep], D[sub]) for kind, D in T.items()})
    layer_r["layer"] = np.arange(K1); layer_r["rho"] = layer_r.layer / (K1 - 1)
    mt = pd.DataFrame(rows); mt["scope"] = scope; mt.to_csv(f"{ANALYSIS_DIR}/mantel_{key}{RUN_TAG}{tag}.csv", index=False)
    layer_r.to_csv(f"{ANALYSIS_DIR}/mantel_layers_{key}{RUN_TAG}{tag}.csv", index=False)
    # mức ngôn ngữ: căn chỉnh trung bình (tâm kết nối) và fertility của tokenizer
    lang_R = np.nanmean(np.where(np.eye(len(LANGS), dtype=bool), np.nan, A), axis=1)
    fert = np.array([meta.get("fertility", {}).get(SHORT[l], np.nan) for l in LANGS])
    ok = np.isfinite(lang_R) & np.isfinite(fert)
    rho_f = spearmanr(lang_R[ok], fert[ok]) if ok.sum() >= 5 else (np.nan, np.nan)
    lang_tab = pd.DataFrame(dict(lang=[SHORT[l] for l in LANGS], family=[FAMILY[l] for l in LANGS], R_pool_mean=lang_R, fertility=fert))
    lang_tab.to_csv(f"{ANALYSIS_DIR}/lang_level_{key}{RUN_TAG}{tag}.csv", index=False)
    np.savez_compressed(f"{ANALYSIS_DIR}/pair_{key}{RUN_TAG}{tag}.npz", A=A, M=M, kw=kw, K1=K1, keep=np.array(keep), **{f"T_{k}": v for k, v in T.items()})
    print(f"[Mantel/{scope}] {key} (tầng {kw}, {len(keep)} ngôn ngữ): " + "; ".join(f"{r.typology}/{r.perm}: r={r.r:+.2f} p₋={r.p_neg:.3f}" for r in mt.itertuples()
                                                       if r.typology in ("syntactic", "genetic")) +
          f" | R~fertility ({int(ok.sum())} ngôn ngữ): ρ={rho_f[0]:+.2f} (p={rho_f[1]:.2f})")
    return mt, layer_r, lang_tab
