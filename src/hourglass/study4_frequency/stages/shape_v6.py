# Hình dạng + capacity + độ bền v6 (Cell C7c)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 38 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["shape_v6"]) hoặc `python -m hourglass run <experiment> shape_v6`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C7c — (v6) HÌNH DẠNG + CAPACITY + ĐỘ BỀN theo cách mới (chạy sau C4b, C5, C7b; không cần GPU)
#   shape_table / shape_summary : tăng, giảm, đỉnh trên lưới cửa sổ (LEFT_QS x RIGHT_QS), cho đường GAM hiệu chỉnh ("adj") và GAM chỉ-spline ("bare")
#   capacity_paired             : đỉnh x̂ trong cửa sổ + bootstrap GHÉP CẶP giữa các model (cùng seed, cùng mẫu từ/cụm -> replicate b của mọi model dùng cùng trọng số)
#                                 đỉnh dính biên = bị cắt (độ lớn dịch là cận dưới, ước lượng bị kéo về 0 -> bảo thủ); KHÔNG dùng p của Kendall (n=4 có p_min=0.083)
#   robustness_curves           : cùng phép đo trên MỌI tập con (LOO, no_hapax, rr_strict, pool_small/large_pairs, ...) -> tỉ lệ tập con còn tăng / giảm,
#                                 thay cho cờ robust_to_lang_choice (so nhãn chữ). Cần chạy lại C5 (v6) để lưu đường cong của tập con.
# Cell cũ giữ nguyên để đối chiếu; KHÔNG dùng S3_supported, robust_to_lang_choice, kendall_p làm kết luận.
# ============================================================
def print_x_support():
    x = _x_data(); n = len(x)
    print(f"[support] {n:,} dạng từ trong phân tích chính; log10 hạng ∈ [{x.min():.2f}, {x.max():.2f}]")
    edges = [0, 2, 3, 4, 4.5, 5, 9]
    for a, b in zip(edges[:-1], edges[1:]):
        m = int(((x >= a) & (x < b)).sum()); print(f"   log10 r ∈ [{a}, {b}): {m:,} từ ({m / n:.1%})")
    print("   mốc phân vị:", {q: round(float(v), 2) for q, v in zip(ANCHOR_QS, np.quantile(x, ANCHOR_QS))})

def shape_table(keys=None, curves=("adj", "bare")):
    rows = []
    for k in (keys or done_models()):
        for cv in curves:
            c = load_curve(k, "all" if cv == "adj" else "gam_bare")
            if c is None:
                continue
            for lq in LEFT_QS:
                for rq in RIGHT_QS:
                    r = shape_contrasts(c["grid"], c["f_obs"], c["f_boot"], c["xq"][lq], c["xq"][rq])
                    if r is None:
                        continue
                    r.update(key=k, curve=cv, left_q=lq, right_q=rq, xL=float(c["xq"][lq]), xR=float(c["xq"][rq])); r["label"] = shape_label(r)
                    rows.append(r)
    return pd.DataFrame(rows)

def shape_summary(T):
    rows = []
    for (k, cv), g in T.groupby(["key", "curve"], sort=False):
        m = g[(g.left_q == MAIN_ANCHOR[0]) & (g.right_q == MAIN_ANCHOR[1])]
        m = (m if len(m) else g).iloc[0]
        rows.append(dict(key=k, curve=cv, label=m["label"], r_star=10 ** m.x_hat, edge_l=m.edge_l, edge_r=m.edge_r,
                         rise=m.rise, rise_lo=m.rise_lo, rise_hi=m.rise_hi, fall=m.fall, fall_lo=m.fall_lo, fall_hi=m.fall_hi, p_rise=m.p_rise, p_fall=m.p_fall,
                         frac_rise=float((g.p_rise < ALPHA_SHAPE).mean()), frac_fall=float((g.p_fall < ALPHA_SHAPE).mean()), frac_inverted=float((g.p_iut < ALPHA_SHAPE).mean()),
                         rise_min=g.rise.min(), rise_max=g.rise.max(), fall_min=g.fall.min(), fall_max=g.fall.max(), n_anchors=len(g)))
    return pd.DataFrame(rows)

def shape_text(r):
    return (f"{MODELS[r['key']]['label']} [{r['curve']}]: {r['label']}; r*≈{r['r_star']:.0f}; tăng {r['rise']:+.3f} [{r['rise_lo']:+.3f}, {r['rise_hi']:+.3f}], "
            f"giảm {r['fall']:+.3f} [{r['fall_lo']:+.3f}, {r['fall_hi']:+.3f}]; qua {r['n_anchors']} cửa sổ: tăng {r['frac_rise']:.0%}, giảm {r['frac_fall']:.0%}, cả hai {r['frac_inverted']:.0%}")

def capacity_paired(keys=None, curve="adj", left_q=MAIN_ANCHOR[0], right_q=MAIN_ANCHOR[1]):
    """-> (bảng theo model, bảng cặp cùng họ, dict slope gộp theo họ) hoặc None. Replicate b của mọi model dùng CÙNG mẫu bootstrap (cùng seed/df/cụm)."""
    nm = "all" if curve == "adj" else "gam_bare"
    ks, PK, M = [], {}, {}
    for k in (keys or done_models()):
        c = load_curve(k, nm)
        if c is None:
            continue
        pk = peak_in_window(c["grid"], c["f_obs"], c["f_boot"], c["xq"][left_q], c["xq"][right_q])
        if pk is None:
            continue
        pk["B"] = c["f_boot"].shape[0]; ks.append(k); PK[k] = pk; M[k] = read_json(f"{run_dir(k)}/meta.json", {})
    if len(ks) < 2:
        return None
    if len({PK[k]["B"] for k in ks}) != 1:
        print("[warn] B khác nhau giữa các model -> không ghép cặp bootstrap được"); return None
    B = PK[ks[0]]["B"]
    P = np.array([M[k].get("n_params_nonemb") or np.nan for k in ks], float); fam = np.array([str(M[k].get("family")) for k in ks])
    xhat = np.array([PK[k]["x_hat"] for k in ks]); XB = np.stack([PK[k]["xb"] for k in ks], 1)       # [B, M]
    status = ["≤ giới hạn trái" if PK[k]["edge_l"] >= 0.5 else ("≥ giới hạn phải" if PK[k]["edge_r"] >= 0.5 else "interior") for k in ks]
    per = pd.DataFrame([dict(key=k, label=MODELS[k]["label"], family=fam[i], n_params=P[i], log10_r_star=xhat[i], lo=float(np.percentile(XB[:, i], 2.5)), hi=float(np.percentile(XB[:, i], 97.5)),
                             edge_l=PK[k]["edge_l"], edge_r=PK[k]["edge_r"], peak_status=status[i], numerics_flag=bool(M[k].get("numerics_flag", False)), n_clip=M[k].get("n_clip"))
                        for i, k in enumerate(ks)])
    pairs = []
    for f in np.unique(fam):
        idx = sorted([i for i in np.where(fam == f)[0] if np.isfinite(P[i])], key=lambda i: P[i])
        for a, b in itertools.combinations(idx, 2):
            d = XB[:, b] - XB[:, a]
            pairs.append(dict(family=f, small=ks[a], large=ks[b], d_log10_r_obs=float(xhat[b] - xhat[a]), d_lo=float(np.percentile(d, 2.5)), d_hi=float(np.percentile(d, 97.5)),
                              p_outward=(1.0 + float((d <= 0).sum())) / (B + 1.0), any_censored=bool(status[a] != "interior" or status[b] != "interior")))
    ok = np.isfinite(P); lb = np.log10(np.where(ok, P, 1.0))
    def fe(xm):
        num = np.zeros(xm.shape[0]); den = 0.0
        for f in np.unique(fam):
            m = (fam == f) & ok
            if m.sum() < 2:
                continue
            lc = lb[m] - lb[m].mean(); xc = xm[:, m] - xm[:, m].mean(1, keepdims=True)
            num += xc @ lc; den += float((lc ** 2).sum())
        return num / den if den > 0 else np.full(xm.shape[0], np.nan)
    sb = fe(XB); so = float(fe(xhat[None, :])[0])
    conc = sum(int(np.sign(p["d_log10_r_obs"]) > 0) for p in pairs)
    slope = dict(curve=curve, left_q=left_q, right_q=right_q, n_models=len(ks), slope=so, lo=float(np.nanpercentile(sb, 2.5)), hi=float(np.nanpercentile(sb, 97.5)),
                 p_outward=(1.0 + float((sb <= 0).sum())) / (B + 1.0), concordant_pairs=conc, n_pairs=len(pairs))
    return per, pd.DataFrame(pairs), slope

def capacity_sensitivity(curve="adj"):
    rows = []
    for lq in LEFT_QS:
        for rq in RIGHT_QS:
            r = capacity_paired(curve=curve, left_q=lq, right_q=rq)
            if r is not None:
                rows.append(r[2])
    return pd.DataFrame(rows)

def robustness_curves(keys=None, la=MAIN_ANCHOR[0], ra=MAIN_ANCHOR[1]):
    rows = []
    for k in (keys or done_models()):
        p = f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        if "sc_names" not in z.files:
            print(f"  [skip] {k}: npz cũ (v5) -> chạy lại C5 (v6) để lưu đường cong của các tập con"); continue
        for i, nm in enumerate(str(s) for s in z["sc_names"]):
            xq = dict(zip(ANCHOR_QS, z[f"sc_xq__{i}"].astype(float)))
            r = shape_contrasts(z[f"sc_grid__{i}"].astype(float), z[f"sc_fobs__{i}"].astype(float), z[f"sc_fboot__{i}"].astype(np.float32), xq[la], xq[ra])
            if r is None:
                continue
            r.update(key=k, subset=nm); r["label"] = shape_label(r); rows.append(r)
    return pd.DataFrame(rows)

def robustness_curves_summary(RC):
    rows = []
    for k, g in RC.groupby("key", sort=False):
        gx = g[~g.subset.str.startswith("alt_axis")]        # trục zipf: đơn vị x khác, không gộp vị trí đỉnh
        rows.append(dict(key=k, n_subsets=len(g), frac_rise=float((g.p_rise < ALPHA_SHAPE).mean()), frac_fall=float((g.p_fall < ALPHA_SHAPE).mean()),
                         frac_inverted=float((g.p_iut < ALPHA_SHAPE).mean()), rise_min=g.rise.min(), rise_med=g.rise.median(), rise_max=g.rise.max(),
                         fall_min=g.fall.min(), fall_med=g.fall.median(), fall_max=g.fall.max(),
                         log10_rstar_min=gx.x_hat.min(), log10_rstar_max=gx.x_hat.max(), n_peak_censored=int(((gx.edge_l >= 0.5) | (gx.edge_r >= 0.5)).sum())))
    return pd.DataFrame(rows)

def robustness_v6(S3_ALL):
    """Phương án dự phòng khi chưa chạy lại C5: chỉ dùng các cột đã có (tiêu chí cửa sổ cũ _q) -> chỉ báo tỉ lệ, không gán nhãn chữ."""
    rows = []
    pl, pr = ("p_left_up_q", "p_right_down_q") if "p_left_up_q" in S3_ALL else ("p_left_up", "p_right_down")
    for k, g in S3_ALL.groupby("key"):
        loo = g[g.subset.str.startswith("loo_") & (g.subset != "loo_problem_langs")]
        if loo.empty:
            continue
        rows.append(dict(key=k, n_loo=len(loo), frac_rise=float((loo[pl] < ALPHA_SHAPE).mean()), frac_fall=float((loo[pr] < ALPHA_SHAPE).mean()),
                         log10_rstar_min=float(loo.x_star.min()), log10_rstar_max=float(loo.x_star.max())))
    return pd.DataFrame(rows)

if len(done_models()):
    print_x_support(); print()
    SHAPE_T = shape_table()
    if len(SHAPE_T):
        SHAPE_S = shape_summary(SHAPE_T)
        SHAPE_T.to_csv(f"{ANALYSIS_DIR}/s3_shape_v6_grid{RUN_TAG}.csv", index=False); SHAPE_S.to_csv(f"{ANALYSIS_DIR}/s3_shape_v6{RUN_TAG}.csv", index=False)
        print(f"== Hình dạng (cửa sổ chính {MAIN_ANCHOR}; tỉ lệ cửa sổ có p<{ALPHA_SHAPE} trên lưới {LEFT_QS} x {RIGHT_QS}) ==")
        print(SHAPE_S.round(4).to_string(index=False)); print()
        for _, _r in SHAPE_S.iterrows():
            print(" -", shape_text(_r))
        print()
        for _cv in ("adj", "bare"):
            _res = capacity_paired(curve=_cv)
            if _res is None:
                continue
            _per, _pairs, _sl = _res
            print(f"== Capacity [{_cv}] cửa sổ {MAIN_ANCHOR}: đỉnh theo model (dính biên = bị cắt) ==")
            print(_per.round(3).to_string(index=False)); print()
            if len(_pairs):
                print(_pairs.round(3).to_string(index=False)); print()
            print(f"   slope gộp trong họ: {_sl['slope']:+.3f} [{_sl['lo']:+.3f}, {_sl['hi']:+.3f}] (log10 r*/log10 N), p_outward={_sl['p_outward']:.3f}, cặp cùng chiều {_sl['concordant_pairs']}/{_sl['n_pairs']}")
            _sens = capacity_sensitivity(_cv)
            print(_sens.round(3).to_string(index=False)); print()
            _per.to_csv(f"{ANALYSIS_DIR}/s3_capacity_v6_{_cv}{RUN_TAG}.csv", index=False); _pairs.to_csv(f"{ANALYSIS_DIR}/s3_capacity_v6_pairs_{_cv}{RUN_TAG}.csv", index=False)
    ROBUST_C = robustness_curves()
    if len(ROBUST_C):
        ROBUST_CS = robustness_curves_summary(ROBUST_C)
        ROBUST_C.to_csv(f"{ANALYSIS_DIR}/s3_robustness_v6_subsets{RUN_TAG}.csv", index=False); ROBUST_CS.to_csv(f"{ANALYSIS_DIR}/s3_robustness_v6{RUN_TAG}.csv", index=False)
        print("== Độ bền trên các tập con (LOO, no_hapax, rr_strict, pool_small/large_pairs, ...) =="); print(ROBUST_CS.round(3).to_string(index=False))
    elif "S3_ALL" in globals() and len(S3_ALL):
        print("== Độ bền (dự phòng, cửa sổ cũ; chạy lại C5 v6 để có bản đầy đủ) =="); print(robustness_v6(S3_ALL).round(3).to_string(index=False))
