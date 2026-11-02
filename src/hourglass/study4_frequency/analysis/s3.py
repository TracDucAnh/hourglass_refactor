# S3: R_k theo log-hạng (Cell C5)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 30 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C5 — S3 trên văn bản tự nhiên: R_k theo log-hạng (Conjecture 2). Chạy cho từng model đã có kết quả; lưu ANALYSIS_DIR/s3_{key}.npz + csv
#   (1) GAM + bootstrap theo từ ở MỌI tầng -> đỉnh r*(k), biên độ, độ dốc hai đầu, p Lind–Mehlum
#   (2) tại tầng waist: CV so sánh mô hình đơn điệu (A2) / unimodal / GAM tự do; hiệu ứng từ loại (UPOS)
#   (3) kiểm tra độ bền: chỉ từ nội dung / chỉ từ chức năng / chỉ từ đã thấy trong FineWeb-2 / thang Zipf / chỉ cặp xác nhận bằng MUSE
# ============================================================
def s3_binned(x, y, nb=25):
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    bi = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    xm = np.array([x[bi == b].mean() for b in range(len(edges) - 1)])
    ym = np.array([y[bi == b].mean() for b in range(len(edges) - 1)])
    se = np.array([y[bi == b].std() / math.sqrt(max((bi == b).sum(), 1)) for b in range(len(edges) - 1)])
    return xm, ym, se

def upos_effects(res, j):
    """Hiệu ứng UPOS (so với NOUN) tại tầng index j + tương phản từ chức năng − từ nội dung, kèm CI bootstrap."""
    nb, names = res["nb"], res["names"]
    up = [(i, n.split(":")[1]) for i, n in enumerate(names) if n.startswith("upos:")]
    bo, bb = res["beta_obs"][nb:, j], res["beta_boot"][:, nb:, j]
    rows = []
    for i, u in up:
        lo, hi = np.percentile(bb[:, i], [2.5, 97.5]); rows.append(dict(upos=u, effect=float(bo[i]), lo=float(lo), hi=float(hi)))
    rows.append(dict(upos="NOUN (ref)", effect=0.0, lo=0.0, hi=0.0))
    def contrast(b):
        d = {u: b[i] for i, u in up}
        f = [d[u] for u in FUNC_UPOS if u in d]; c = [0.0] + [d[u] for u in CONTENT_UPOS if u in d and u != "NOUN"]
        return (np.mean(f) if f else np.nan) - np.mean(c)
    co = contrast(bo); cb = np.array([contrast(b) for b in bb])
    ci = np.nanpercentile(cb, [2.5, 97.5])
    return pd.DataFrame(rows), dict(func_minus_content=float(co), lo=float(ci[0]), hi=float(ci[1]))

def analyze_model(key, B=N_BOOT, subsets=True):
    t0 = time.time()
    run, meta = load_run(key); df_full, R_full = word_table(key); kw, K1 = waist_layer(key, run)
    pair_muse = [pi for pi in range(P_) if PAIR_STATS.confirm_src.iloc[pi] in MAIN_SRCS]          # = cặp TẦNG CHÍNH (MAIN_SRCS)
    main_muse = MAIN_PAIRS == "muse" and 0 < len(pair_muse) < P_
    if MAIN_PAIRS == "muse" and len(pair_muse) == 0:
        print("  [warn] MAIN_PAIRS='muse' nhưng không cặp nào thuộc MAIN_SRCS -> dùng mọi cặp")
    df0, R0 = word_table(key, pair_muse) if main_muse else (df_full, R_full)
    df, R = prep_main(df0, R0)
    if main_muse:
        print(f"  [pairs] phân tích chính chỉ dùng {len(pair_muse)}/{P_} cặp thuộc tầng chính {MAIN_SRCS}")
    print(f"[S3] {key}: {len(df):,} dạng từ (đơn vị bootstrap), tầng waist k={kw}/{K1 - 1}")
    res = gam_analysis(df, R, layers=None, B=B, k_ref=kw)
    tab = res["table"].copy(); tab["rho"] = tab.layer / (K1 - 1); tab["r_star"] = 10 ** tab.x_star
    tab.to_csv(f"{ANALYSIS_DIR}/s3_layers_{key}{RUN_TAG}.csv", index=False)
    # (2) tầng waist
    x = df.log_rank.values; y = res["y"][:, kw]
    se = cv_shapes(res["X"], res["nb"], res["P"], res["lam"], y, x)
    cv = {"mse_" + m: float(v.mean()) for m, v in se.items()}
    for m in ("const", "mono", "unimodal"):
        cv[f"d_{m}_vs_gam"], cv[f"d_{m}_vs_gam_lo"], cv[f"d_{m}_vs_gam_hi"] = paired_boot_ci(se[m] - se["gam"])
    cv["d_mono_vs_uni"], cv["d_mono_vs_uni_lo"], cv["d_mono_vs_uni_hi"] = paired_boot_ci(se["mono"] - se["unimodal"])
    ue, uc = upos_effects(res, kw); ue.to_csv(f"{ANALYSIS_DIR}/s3_upos_{key}{RUN_TAG}.csv", index=False)
    # đường đơn điệu / unimodal trên phần dư riêng (để vẽ chồng lên GAM)
    des, nb = res["design"], res["nb"]
    par = res["X"][:, nb:] @ res["beta_obs"][nb:, kw].astype(float)
    ys = y - par; off = float(res["X"][:, nb:].mean(0) @ res["beta_obs"][nb:, kw])
    inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x, ys); dec = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x, ys)
    mono = inc if ((ys - inc.predict(x)) ** 2).sum() <= ((ys - dec.predict(x)) ** 2).sum() else dec
    grid = res["grid"]; uni = unimodal_fit(x, ys)
    xb, yb, sb = s3_binned(x, y)
    xs_boot = np.asarray(res["grid"])[res["f_boot"][:, kw, :].astype(np.float32).argmax(-1)]
    row = res["table"].iloc[kw].to_dict(); row["r_star"] = 10 ** row["x_star"]; row.update(cv); row.update({f"upos_{k}": v for k, v in uc.items()})
    row.update(key=key, kw=kw, rho_w=kw / (K1 - 1), n_words=len(df), n_params=meta.get("n_params_nonemb"), family=meta.get("family"), subset="all")
    row.update(n_clip=meta.get("n_clip"), numerics_flag=bool(meta.get("numerics_flag", False)), rr_variant=RR_VARIANT)
    out_rows = [row]
    bare_res = None
    SUBC = {}          # (v6) đường cong quan sát + bootstrap + mốc phân vị của từng tập con, để C7c đo lại hình dạng
    # (3) độ bền
    def _sub(nm, d_, R_, **kw_):
        rs = gam_analysis(d_, R_, layers=[kw], B=B, k_ref=kw, **kw_)
        r2 = rs["table"].iloc[0].to_dict()
        r2.update(key=key, kw=kw, rho_w=kw / (K1 - 1), n_words=len(d_), subset=nm); out_rows.append(r2)
        SUBC[nm] = dict(grid=rs["grid"], f_obs=rs["f_obs"][0], f_boot=rs["f_boot"][:, 0, :], xq=np.quantile(d_.log_rank.values, ANCHOR_QS))
        return rs
    if subsets:
        if RUN_COVAR_CHECKS:       # (v5) B/E: GAM không hiệu chỉnh vs chấm thô; GAM không nsub/độ dài/ngôn ngữ; độ nhạy theo λ
            bare_res = _sub("gam_bare", df, R, covars="none")
            _sub("gam_upos_pmix", df, R, covars="upos_pmix")
            for f_ in LAM_SENS:
                _sub(f"lam_x{f_:g}", df, R, lam=res["lam"] * f_)
        if "fw_count" in df.columns:       # (v5) C: bỏ khối hapax (chung hạng trung bình ở biên phải)
            mh = (df.fw_count.values >= 2)
            if mh.sum() >= 500:
                _sub("no_hapax", df[mh].reset_index(drop=True), R[mh])
        if MASK_VALID_TRANSLATIONS:        # (v5) A: biến thể RR còn lại làm đối chiếu
            other = "strict" if RR_VARIANT == "valid" else "valid"
            d_o, R_o = word_table(key, pair_muse if main_muse else None, variant=other); d_o, R_o = prep_main(d_o, R_o)
            _sub(f"rr_{other}", d_o, R_o)
        if "tier" in ITEMS and ITEM_MAX_TIER > 0 and bool((ITEMS["tier"][ITEMS["tier"] <= 2] > 0).any()):      # (v7) hình dạng có giữ nguyên khi chỉ dùng mục xác nhận bởi từ điển GỐC (= tập v5), bỏ phần mở rộng lemma/Wiktionary?
            d_t, R_t = word_table(key, pair_muse if main_muse else None, max_tier=0); d_t, R_t = prep_main(d_t, R_t)
            if len(d_t) >= LOO_MIN_WORDS:
                _sub("tier0_items", d_t, R_t)
        _um = (~df.lang.isin([SHORT[l] for l in UPOS_PROJECTED]).values) if (UPOS_SUBSETS_EXCLUDE_PROJECTED and UPOS_PROJECTED) else np.ones(len(df), bool)
        specs = [("content", df.upos.isin(CONTENT_UPOS).values & _um), ("function", df.upos.isin(FUNC_UPOS).values & _um),
                 ("n_inst>=2", (df.n_inst >= 2).values)]
        for nm, m in specs:
            if m.sum() >= 500:
                _sub(nm, df[m].reset_index(drop=True), R[m])
        dz = df0.copy(); dz["log_rank"] = (-dz.zipf) if AXIS == "logrank" else dz.log_rank_raw      # trục thay thế
        _sub("alt_axis(-zipf)" if AXIS == "logrank" else "alt_axis(logrank)", *prep_main(dz, R0))
        _sub("with_unseen", *prep_main(df0, R0, drop_unseen=False))
        if len(MEASURE_SETS) > 1:                      # lặp lại theo từng nguồn/miền: hình dạng có giữ nguyên trên FLORES và NTREX riêng lẻ không?
            for si, ds in enumerate(MEASURE_SETS):
                d_, R_ = word_table(key, pair_muse if main_muse else None, src=si)
                d_, R_ = prep_main(d_, R_)
                if len(d_) >= 500:
                    _sub(f"src_{ds}", d_, R_)
        if main_muse:
            _sub("all_pairs", *prep_main(df_full, R_full))
        elif 0 < len(pair_muse) < P_:
            d2, R2 = word_table(key, pair_muse); _sub("muse_pairs_only", *prep_main(d2, R2))
        # ---- (v4) độ bền theo lựa chọn ngôn ngữ: kết luận có phụ thuộc jpn / zho / swh không? ----
        def _sub_pairs(nm, pairs):
            if not pairs:
                return
            d_, R_ = word_table(key, pairs); d_, R_ = prep_main(d_, R_)
            if len(d_) >= LOO_MIN_WORDS:
                _sub(nm, d_, R_)
            else:
                print(f"  [skip] {nm}: chỉ {len(d_)} dạng từ (< {LOO_MIN_WORDS})")
        base_pairs = pair_muse if main_muse else list(range(P_))
        pair_ext = [pi for pi in range(P_) if pi not in set(pair_muse)]
        if main_muse and pair_ext:
            _sub_pairs("ext_tier_only", pair_ext)                                  # chỉ các cặp tầng mở rộng (alt/dice), riêng biệt với tầng chính
        # (v6) kích thước pool theo cặp (số mục hợp lệ 3.6k–31.8k) có làm đổi hình dạng không: tam phân vị cặp nhỏ nhất / lớn nhất
        nv_p = np.diff(run["pair_ptr"]); srt = sorted(base_pairs, key=lambda pi: nv_p[pi]); t3 = len(srt) // 3
        if t3 >= 3:
            _sub_pairs("pool_small_pairs", srt[:t3]); _sub_pairs("pool_large_pairs", srt[-t3:])
        in_base = sorted({i for pi in base_pairs for i in PAIRS[pi]})
        loo = [LANGS[i] for i in in_base] if LOO_LANGS is None else [l for l in LOO_LANGS if LANGS.index(l) in in_base]
        for l in loo:                                                              # leave-one-language-out
            _sub_pairs(f"loo_{SHORT[l]}", pairs_without(base_pairs, [l]))
        prob = [l for l in PROBLEM_LANGS if LANGS.index(l) in in_base]
        if len(prob) >= 2:                                                         # bỏ CÙNG LÚC các ngôn ngữ nghi gây thiên lệch phương pháp
            _sub_pairs("loo_problem_langs", pairs_without(base_pairs, prob))
    fb_w = res["f_boot"][:, kw, :].astype(np.float32); w_ = fb_w - fb_w.min(1, keepdims=True)
    cent = (w_ * np.asarray(grid)[None]).sum(1) / np.maximum(w_.sum(1), 1e-12)          # trọng tâm của (f − min f): thống kê không dính biên
    extra = {} if bare_res is None else dict(f_boot_w_bare=bare_res["f_boot"][:, 0, :], f_obs_bare=bare_res["f_obs"][0])
    SUBC["all"] = dict(grid=grid, f_obs=res["f_obs"][kw], f_boot=res["f_boot"][:, kw, :], xq=np.quantile(df.log_rank.values, ANCHOR_QS))      # (v6)
    extra.update(pack_subcurves(SUBC))
    np.savez_compressed(f"{ANALYSIS_DIR}/s3_{key}{RUN_TAG}.npz", grid=grid, f_obs=res["f_obs"], f_boot_w=res["f_boot"][:, kw, :],
                        x_star_boot=xs_boot, bin_x=xb, bin_y=yb, bin_se=sb, mono_y=mono.predict(grid) + off, uni_y=uni(grid) + off,
                        kw=kw, K1=K1, off=off, centroid_boot=cent, **extra)
    print(f"   r*={10 ** row['x_star']:.0f} [{10 ** row['x_star_lo']:.0f}, {10 ** row['x_star_hi']:.0f}]  p_LM={row['p_LM']:.3f} (q={row['p_LM_q']:.3f}, amp={row['p_LM_amp']:.3f}) shape={row['shape']}/{row['shape_q']}  "
          f"ΔMSE(mono−gam)={row['d_mono_vs_gam']:.2e} [{row['d_mono_vs_gam_lo']:.1e}, {row['d_mono_vs_gam_hi']:.1e}]  ({time.time() - t0:.0f}s)")
    return pd.DataFrame(out_rows), ue, uc
