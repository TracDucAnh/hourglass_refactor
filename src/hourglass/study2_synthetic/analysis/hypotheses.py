# Kiểm định P1, P2, P3, S2–S6 (Cell D3)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 37 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D3 — Kiểm định giả thuyết P1, P2, P3, S2, S3, S4, S5, S6
# Mỗi hàm trả về dict(hyp, verdict, evidence, table). verdict ∈ {ủng hộ, không ủng hộ, chưa kết luận, chưa có dữ liệu}.
# ============================================================
def test_P1(df):
    """P1: cả bốn họ có waist (G_k, D_k dip so với mô hình chưa huấn luyện); waist còn khi giao diện cân bằng (loại A1)."""
    rows = []
    d0 = sel(df, overlap=0, level="M4", task=["LM", "SEM"])
    for fam in FAMILIES:
        s = sel(d0, family=fam)
        if s.empty:
            continue
        diff = (s.S_edge - s.S_edge_init_hi).values
        p = wilcoxon_p(diff)
        rows.append(dict(part="4 họ (overlap 0%, M4, LM+SEM)", group=fam, n=len(s), n_waist=int(s.has_waist.sum()), frac=float(s.has_waist.mean()),
                         median_S_edge=float(s.S_edge.median()), median_S_edge_init_hi=float(s.S_edge_init_hi.median()), p_vs_init=p,
                         frac_Ushape=float(((s.slope_left > 0) & (s.slope_right < 0)).mean()), median_rho=float(s.rho_star.median())))
    for lv in ("M0", "M2", "M4"):
        s = sel(df, family="decoder", overlap=0, level=lv, task=["LM", "SEM"], grid="p1p2")
        b = sel(df, family="decoder", overlap=100, level=lv, task=["LM", "SEM"], grid="balanced")
        for name, x in (("overlap 0%", s), ("overlap 100% (giao diện cân bằng)", b)):
            if len(x):
                rows.append(dict(part=f"A1 decoder {lv}", group=name, n=len(x), n_waist=int(x.has_waist.sum()), frac=float(x.has_waist.mean()),
                                 median_S_edge=float(x.S_edge.median()), median_S_edge_init_hi=float(x.S_edge_init_hi.median()),
                                 p_vs_init=wilcoxon_p((x.S_edge - x.S_edge_init_hi).values), frac_Ushape=float(((x.slope_left > 0) & (x.slope_right < 0)).mean()),
                                 median_rho=float(x.rho_star.median())))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P1", NO_DATA, "")
    fam = t[t.part.str.startswith("4 họ")]
    ok_fam = {r.group: bool(r.frac >= 0.8 and r.p_vs_init < 0.05) for r in fam.itertuples()}
    bal = t[(t.part == "A1 decoder M4") & (t.group.str.startswith("overlap 100"))]
    ok_bal = bool(len(bal) and bal.iloc[0].frac >= 0.8 and bal.iloc[0].p_vs_init < 0.05)
    miss = [f for f in FAMILIES if f not in ok_fam]
    ev = f"họ có waist: {ok_fam}; còn waist khi overlap 100% (M4): {ok_bal if len(bal) else NO_DATA}" + (f"; thiếu dữ liệu: {miss}" if miss else "")
    if miss or not len(bal):
        return _res("P1", "chưa kết luận", ev, t)
    return _res("P1", "ủng hộ" if (all(ok_fam.values()) and ok_bal) else "không ủng hộ", ev + ("" if ok_bal else " -> A1 (giao diện) chưa bị loại"), t)

def test_P2(df):
    """P2: vị trí/độ rộng/độ sâu waist đi theo tác vụ: rộng nhất ở SEM, hẹp hoặc dịch ở SURF; kỳ vọng bị bác nếu eo cố định bất kể tác vụ."""
    rows = []
    for fam in ("decoder", "encoder"):
        for lv in LEVELS:
            s = sel(df, family=fam, level=lv, overlap=0, grid="p1p2")
            if s.empty:
                continue
            piv = {m: s.pivot_table(index="seed", columns="task", values=m) for m in ("w", "S_edge", "rho_star")}
            if not all(t in piv["w"].columns for t in TASKS):
                continue
            g = [piv["w"][t].dropna().values for t in ("SURF", "LM", "SEM")]
            J, z, pj = jonckheere_terpstra(g)
            dS = (piv["S_edge"]["SEM"] - piv["S_edge"]["SURF"]).values
            dw = (piv["w"]["SEM"] - piv["w"]["SURF"]).values
            dr = (piv["rho_star"]["SEM"] - piv["rho_star"]["SURF"]).values
            rows.append(dict(family=fam, level=lv, n_seed=len(dS), JT_p_width=pj, p_depth_SEM_gt_SURF=wilcoxon_p(dS), p_width_SEM_gt_SURF=wilcoxon_p(dw),
                             p_rho_shift=wilcoxon_p(dr, alt="two-sided"), mean_drho=float(np.nanmean(dr)),
                             tost_rho_fixed=tost_p(dr, RHO_MARGIN), tost_depth_equal=tost_p(dS, 0.05),
                             mean_w={t: round(float(piv["w"][t].mean()), 3) for t in TASKS}))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P2", NO_DATA, "")
    # Holm TRONG từng mức (M4 là mức chính: 2 họ -> với 6 seed p_min = 0.0156 x 2 = 0.031 < 0.05 vẫn đạt được);
    # M0/M2 được báo cáo để so sánh (thăm dò).
    for c in ("JT_p_width", "p_depth_SEM_gt_SURF", "p_width_SEM_gt_SURF", "p_rho_shift"):
        t[c + "_holm"] = np.nan
        for lv in t.level.unique():
            ix = t.level == lv
            t.loc[ix, c + "_holm"] = holm(t.loc[ix, c].fillna(1.0).values)
    t["task_matters"] = (t.JT_p_width_holm < 0.05) | (t.p_depth_SEM_gt_SURF_holm < 0.05) | ((t.p_rho_shift_holm < 0.05) & (t.mean_drho.abs() > RHO_MARGIN))
    t["waist_fixed"] = (t.tost_rho_fixed < 0.05) & (t.tost_depth_equal < 0.05)
    m4 = t[t.level == "M4"]
    ev = f"M4: nhạy với tác vụ {m4.set_index('family').task_matters.to_dict()}; eo cố định (TOST) {m4.set_index('family').waist_fixed.to_dict()}"
    if m4.empty:
        return _res("P2", "chưa kết luận", ev, t)
    if m4.waist_fixed.all():
        return _res("P2", "không ủng hộ", ev, t)
    return _res("P2", "ủng hộ" if m4.task_matters.all() else "chưa kết luận", ev, t)

def _delta_A(r):
    """ΔA[pair, tầng] = A_joint - A_indep cho một run có sinh đôi."""
    c, tw = cv(r.rid, "trained"), _ld(r.rid, "twins")
    return None if (c is None or tw is None) else c["Apair"] - tw["A_indep"]

def test_P3(df):
    """P3: huấn luyện chung thêm căn chỉnh so với sinh đôi đơn ngữ: ΔA > 0, đỉnh ở waist, giảm khi capacity tăng, đúng cho mọi cặp."""
    s = df[df.has_twins & df.grid.isin(["p3s2", "smoke"])]
    if s.empty:
        return _res("P3", NO_DATA, "")
    rows, dA_curves = [], {}
    for r in s.itertuples():
        dA = _delta_A(r)
        if dA is None:
            continue
        m = np.nanmean(dA, 0); k = r.k_star
        rows.append(dict(rid=r.rid, d=r.d, L=r.L, seed=r.seed, ln_params=r.ln_params, dA_kstar=float(m[k]), dA_peak_layer=int(np.nanargmax(m)), k_star=k,
                         frac_pairs_pos=float((dA[:, k] > 0).mean()), dA_max=float(np.nanmax(m)), A_joint_k=float(np.nanmean(cv(r.rid)["Apair"][:, k])),
                         A_indep_k=float(np.nanmean(_ld(r.rid, "twins")["A_indep"][:, k]))))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P3", NO_DATA, "")
    sd_seed = float(t.groupby(["d", "L"]).dA_kstar.std().mean())
    margin = max(0.05, sd_seed if np.isfinite(sd_seed) else 0.05)
    p_pos = wilcoxon_p(t.dA_kstar.values)
    p_tost = tost_p(t.dA_kstar.values, margin)
    sl = stats.linregress(t.ln_params, t.dA_kstar) if len(t) >= 5 else None
    p_slope = float(sl.pvalue / 2 if sl.slope < 0 else 1 - sl.pvalue / 2) if sl else np.nan
    at_waist = float((np.abs(t.dA_peak_layer - t.k_star) <= 1).mean())
    frac_pairs = float(t.frac_pairs_pos.mean())
    ev = (f"ΔA tại waist: median {t.dA_kstar.median():.3f} (Wilcoxon p={p_pos:.3g}); TOST≈0 (biên {margin:.3f}) p={p_tost:.3g}; "
          f"độ dốc theo log B = {sl.slope:.3f} (p một phía={p_slope:.3g}); đỉnh ΔA trong ±1 tầng quanh waist: {at_waist:.0%}; cặp có ΔA>0: {frac_pairs:.0%}") if sl else "quá ít run có sinh đôi"
    if not sl:
        return _res("P3", "chưa kết luận", ev, t)
    if p_tost < 0.05:
        return _res("P3", "không ủng hộ", ev + " -> ΔA≈0: A2 (hội tụ Platonic)", t)
    ok = (p_pos < 0.05) and (p_slope < 0.05) and at_waist >= 0.5 and frac_pairs >= 0.9
    return _res("P3", "ủng hộ" if ok else "chưa kết luận", ev, t)

def test_S2(df):
    """S2: căn chỉnh tại waist là hàm bậc thang theo capacity (và số ngôn ngữ), phương sai giữa seed đạt đỉnh ở bước nhảy."""
    s = sel(df, grid="p3s2", family="decoder")
    if s.empty:
        return _res("S2", NO_DATA, "")
    rows = []
    for L in sorted(s.L.unique()):
        x = s[s.L == L]
        if x.d.nunique() < 4:
            continue
        dAIC, i = step_vs_linear(np.log2(x.d.values), x.G_at_kstar.values)
        ds = sorted(x.d.unique())
        grp = [x[x.d == d].G_at_kstar.values for d in ds]
        bf = stats.levene(*grp, center="median").pvalue if all(len(g) > 1 for g in grp) else np.nan
        sds = [np.nanstd(g, ddof=1) for g in grp]
        peak = int(np.nanargmax(sds)) if np.isfinite(sds).any() else -1
        rows.append(dict(L=L, n=len(x), dAICc_step_vs_linear=dAIC, step_after_d=ds[i] if i >= 0 else None, levene_p=bf,
                         var_peak_d=ds[peak] if peak >= 0 else None, var_peak_adjacent_to_step=bool(i >= 0 and peak in (i, i + 1))))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("S2", NO_DATA, "cần >= 4 giá trị d")
    t["staircase"] = (t.dAICc_step_vs_linear > 2) & t.var_peak_adjacent_to_step
    ev = f"L có dạng bậc thang + đỉnh phương sai ở bước nhảy: {int(t.staircase.sum())}/{len(t)}; ΔAICc: {t.set_index('L').dAICc_step_vs_linear.round(2).to_dict()}"
    return _res("S2", "ủng hộ" if t.staircase.sum() >= 2 else ("không ủng hộ" if (t.dAICc_step_vs_linear < -2).all() else "chưa kết luận"), ev, t)

def _inverted_u(y, cnt, min_cnt=10):
    """y: G theo bin hạng. Trả về (hệ số bậc 2, đỉnh, độ dốc đầu, độ dốc cuối, SSE tuyến-đơn-điệu-tăng, SSE bậc 2)."""
    from sklearn.isotonic import IsotonicRegression
    x = np.arange(len(y)); ok = np.isfinite(y) & (cnt >= min_cnt)
    if ok.sum() < 5:
        return (np.nan,) * 6
    xo, yo = x[ok].astype(float), y[ok]
    c2, c1, c0 = np.polyfit(xo, yo, 2)
    sse_q = float(np.square(yo - np.polyval([c2, c1, c0], xo)).sum())
    iso = IsotonicRegression(increasing=True).fit(xo, yo)
    sse_m = float(np.square(yo - iso.predict(xo)).sum())
    peak = -c1 / (2 * c2) if c2 != 0 else np.nan
    return c2, peak, c1 + 2 * c2 * xo[0], c1 + 2 * c2 * xo[-1], sse_m, sse_q

def test_S3(df):
    """S3: căn chỉnh theo hạng khái niệm là chữ U ngược (không đơn điệu tăng); đỉnh dịch ra xa khi capacity tăng."""
    s = df[df.grid.isin(["p3s2", "smoke", "s5"]) & (df.family == "decoder")]
    rows = []
    for r in s.itertuples():
        c = cv(r.rid)
        if c is None or "Grank" not in c:
            continue
        c2, pk, sl0, sl1, sm, sq = _inverted_u(c["Grank"][r.k_star], c["rank_counts"])
        rows.append(dict(rid=r.rid, d=r.d, L=r.L, ln_params=r.ln_params, curv=c2, peak_bin=pk, slope_start=sl0, slope_end=sl1, sse_mono=sm, sse_quad=sq))
    t = pd.DataFrame(rows).dropna()
    if len(t) < 5:
        return _res("S3", NO_DATA, "")
    p_curv = float(stats.ttest_1samp(t.curv, 0, alternative="less").pvalue)
    p_l = float(stats.ttest_1samp(t.slope_start, 0, alternative="greater").pvalue)
    p_r = float(stats.ttest_1samp(t.slope_end, 0, alternative="less").pvalue)
    p_mono = wilcoxon_p((t.sse_mono - t.sse_quad).values)
    inter = float(((t.peak_bin > 0) & (t.peak_bin < N_RANK_BINS - 1)).mean())
    rho = stats.spearmanr(t.ln_params, t.peak_bin).correlation if len(t) > 4 else np.nan
    ev = (f"độ cong<0 p={p_curv:.3g}; dốc đầu>0 p={p_l:.3g}; dốc cuối<0 p={p_r:.3g}; đơn điệu bị bác (SSE_mono>SSE_quad) p={p_mono:.3g}; "
          f"đỉnh nằm trong khoảng hạng: {inter:.0%}; Spearman(log B, vị trí đỉnh)={rho:.2f}")
    ok = p_curv < 0.05 and p_l < 0.05 and p_r < 0.05 and p_mono < 0.05 and inter >= 0.5
    return _res("S3", "ủng hộ" if ok else ("không ủng hộ" if p_l >= 0.5 else "chưa kết luận"), ev, t)

def test_S4(df):
    """S4: với dữ liệu bằng nhau, ngôn ngữ có fertility thấp nhất trở thành pivot (logit-lens ở tầng quanh waist); A3 dự đoán theo khối lượng dữ liệu."""
    s = sel(df, grid="s4")
    rows = []
    for r in s.itertuples():
        lens = _ld(r.rid, "lens")
        if lens is None:
            continue
        M = lens["M"]; k = r.k_star; ks = [x for x in (k - 1, k, k + 1) if 0 < x < M.shape[0] - 1] or [k]
        score = np.array([np.mean([M[kk, [b for b in range(r.L) if b != l], l].mean() for kk in ks]) for l in range(r.L)])
        rows.append(dict(rid=r.rid, seed=r.seed, share=("40%" if _has(r.share_X) else "đều"), fert="thấp" if r.len_scale_X < 1 else "cao", fert_X=r.fert_X,
                         score_X=float(score[0]), diff_X=float(score[0] - score[1:].mean()), X_is_pivot=bool(score.argmax() == 0)))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("S4", NO_DATA, "")
    piv = t.pivot_table(index="seed", columns=["share", "fert"], values="diff_X")
    ev, verdict = "", "chưa kết luận"
    if ("đều", "thấp") in piv.columns and ("đều", "cao") in piv.columns:
        d_cost = (piv[("đều", "thấp")] - piv[("đều", "cao")]).values
        p_cost = wilcoxon_p(d_cost)
        ev = f"dữ liệu đều: pivot-score của X (fertility thấp) - (fertility cao) = {np.nanmean(d_cost):.4f}, p={p_cost:.3g}"
        d_data = np.nan
        if ("40%", "cao") in piv.columns:
            d_data = (piv[("40%", "cao")] - piv[("đều", "cao")]).values
            p_data = wilcoxon_p(d_data)
            ev += f"; hiệu ứng khối lượng dữ liệu (40% vs đều, fertility cao) = {np.nanmean(d_data):.4f}, p={p_data:.3g}"
            if p_cost >= 0.05 and p_data < 0.05:
                verdict = "không ủng hộ"; ev += " -> pivot chỉ theo dữ liệu (A3)"
        if p_cost < 0.05 and np.nanmean(d_cost) > 0:
            verdict = "ủng hộ"
    return _res("S4", verdict, ev, t)

def test_S5(df):
    """S5: độ sâu tương đối ρ* tương quan dương với fertility trung bình (cùng họ, cùng capacity)."""
    s = sel(df, grid="s5")
    if len(s) < 8:
        return _res("S5", NO_DATA, "")
    x = s.fert_mean.values - s.groupby("seed").fert_mean.transform("mean").values        # khử hiệu ứng seed
    y = s.rho_star.values - s.groupby("seed").rho_star.transform("mean").values
    rs, ps = stats.spearmanr(s.fert_mean, s.rho_star); rk, pk = stats.kendalltau(s.fert_mean, s.rho_star)
    rs_w, ps_w = stats.spearmanr(x, y)
    ev = f"Spearman(φ, ρ*)={rs:.2f} (p hai phía {ps:.3g}); Kendall={rk:.2f} (p={pk:.3g}); trong-seed Spearman={rs_w:.2f} (p={ps_w:.3g})"
    ok = rs > 0 and ps / 2 < 0.05 and rk > 0 and pk / 2 < 0.05
    bad = rs <= 0 and ps / 2 < 0.05
    return _res("S5", "ủng hộ" if ok else ("không ủng hộ" if bad else "chưa kết luận"), ev, s[["rid", "len_scale", "seed", "fert_mean", "rho_star", "S_edge"]])

def test_S6(df):
    """S6 (một phần): với diffusion, đổi mức che không làm dịch k* (TOST |Δρ*| <= 0.10) mà chủ yếu đổi chiều cao / các tầng cuối."""
    s = df[(df.family == "diffusion") & df.get("rho_t0.25", pd.Series(dtype=float)).notna()] if "rho_t0.25" in df.columns else df.iloc[0:0]
    if s.empty:
        return _res("S6", NO_DATA, "")
    rows = []
    for t in MASK_LEVELS[1:]:
        dr = (s[f"rho_t{t:g}"] - s.rho_star).values
        dS = (s[f"S_edge_t{t:g}"] - s.S_edge).values
        dL = (s[f"G_last_t{t:g}"] - s.G_last).values
        rows.append(dict(t=t, n=len(s), mean_drho=float(np.mean(dr)), tost_p=tost_p(dr, RHO_MARGIN), mean_dS_edge=float(np.mean(dS)), p_dS_two_sided=wilcoxon_p(dS, alt="two-sided"),
                         mean_dG_last=float(np.mean(dL)), p_dGlast_two_sided=wilcoxon_p(dL, alt="two-sided")))
    tt = pd.DataFrame(rows)
    fr = np.nan
    try:
        cols = [s.rho_star.values] + [s[f"rho_t{t:g}"].values for t in MASK_LEVELS[1:]]
        fr = float(stats.friedmanchisquare(*cols).pvalue)
    except Exception:
        pass
    ev = f"TOST k* không đổi (biên {RHO_MARGIN}): p={tt.tost_p.round(3).tolist()}; Friedman qua t: p={fr:.3g}; ΔG_last theo t: {tt.mean_dG_last.round(3).tolist()}"
    return _res("S6", "ủng hộ" if (tt.tost_p < 0.05).all() else "chưa kết luận", ev + " (kiểm 'đáy không dịch cùng Pha III' đầy đủ cần so sánh Pha III riêng)", tt)

def run_all_tests(df):
    return [f(df) for f in (test_P1, test_P2, test_P3, test_S2, test_S3, test_S4, test_S5, test_S6)]
