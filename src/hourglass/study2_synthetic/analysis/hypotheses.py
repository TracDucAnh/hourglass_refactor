# Kiểm định P1, P2, P3, S2–S6 (Cell D3)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 37 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D3 — Kiểm định giả thuyết P1, P2 (chính) và S3 (bổ trợ) — phạm vi Study 2a
# Mỗi hàm trả về dict(hyp, verdict, evidence, table). verdict ∈ {ủng hộ, không ủng hộ, chưa kết luận, chưa có dữ liệu}.
# ============================================================
def test_P1(df):
    """P1: cả bốn họ có waist (G_k, D_k dip so với mô hình chưa huấn luyện) trên ngôn ngữ tổng hợp (overlap 0%, M4, LM+SEM).
    Bảng có thêm từng tác vụ riêng để thấy tác vụ nào thiếu waist."""
    rows = []
    d0 = sel(df, overlap=0, level="M4", task=["LM", "SEM"], grid=["gate_g2", "p1p2", "moe_diff", "smoke"])    # không gồm lưới chẩn đoán
    for fam in FAMILIES:
        s = sel(d0, family=fam)
        for task, x in [("LM+SEM", s), ("LM", sel(s, task="LM")), ("SEM", sel(s, task="SEM"))]:
            if len(x):
                rows.append(dict(group=fam, task=task, n=len(x), n_waist=int(x.has_waist.sum()), frac=float(x.has_waist.mean()),
                                 median_S_edge=float(x.S_edge.median()), median_S_edge_init_hi=float(x.S_edge_init_hi.median()),
                                 p_vs_init=wilcoxon_p((x.S_edge - x.S_edge_init_hi).values),
                                 frac_Ushape=float(((x.slope_left > 0) & (x.slope_right < 0)).mean()), median_rho=float(x.rho_star.median())))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P1", NO_DATA, "")
    ok_fam = {r.group: bool(r.frac >= 0.8 and r.p_vs_init < 0.05) for r in t[t.task == "LM+SEM"].itertuples()}
    by_task = {f"{r.group}/{r.task}": round(r.frac, 2) for r in t[t.task != "LM+SEM"].itertuples()}
    miss = [f for f in FAMILIES if f not in ok_fam]
    ev = f"họ có waist: {ok_fam}; tỉ lệ run có waist theo tác vụ: {by_task}" + (f"; thiếu dữ liệu: {miss}" if miss else "")
    if miss:
        return _res("P1", "chưa kết luận", ev, t)
    return _res("P1", "ủng hộ" if all(ok_fam.values()) else "không ủng hộ", ev, t)

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
    """S3 (bổ trợ, tiêu chí nới): căn chỉnh tại waist theo hạng tần suất danh từ có dạng chữ U ngược = cong xuống và đỉnh nằm
    trong khoảng hạng. Gộp mọi họ / mức / tác vụ ở overlap 0%; đơn vị thống kê là seed. Không kiểm đỉnh dịch theo capacity."""
    s = df[df.grid.isin(["p1p2", "moe_diff", "smoke"]) & (df.overlap == 0)]
    rows = []
    for r in s.itertuples():
        c = cv(r.rid)
        if c is None or "Grank" not in c:
            continue
        c2, pk = _inverted_u(c["Grank"][r.k_star], c["rank_counts"])[:2]
        pk = pk if c2 < 0 else np.nan                       # độ cong dương: đỉnh parabol là điểm THẤP nhất, không phải đỉnh
        rows.append(dict(rid=r.rid, family=r.family, level=r.level, task=r.task, seed=r.seed, k_star=r.k_star, curv=c2, peak_bin=pk,
                         peak_inside=bool(np.isfinite(pk) and 0 < pk < N_RANK_BINS - 1)))
    t = pd.DataFrame(rows).dropna(subset=["curv"]) if rows else pd.DataFrame()
    if len(t) < 5:
        return _res("S3", NO_DATA, "")
    by_seed = t.groupby("seed").curv.mean()
    p_curv = wilcoxon_p(-by_seed.values)                     # một phía: độ cong trung bình theo seed < 0
    is_u = (t.curv < 0) & t.peak_inside
    frac_u = float(is_u.mean())
    by_fam = {f: round(float(is_u[t.family == f].mean()), 2) for f in t.family.unique()}
    ev = (f"độ cong < 0 (theo seed, n={len(by_seed)}): p={p_curv:.3g}; run có chữ U ngược (cong xuống, đỉnh trong khoảng hạng): "
          f"{frac_u:.0%}; theo họ: {by_fam}")
    return _res("S3", "ủng hộ" if (p_curv < 0.05 and frac_u >= 0.5) else "chưa kết luận", ev, t)

def test_anchor(df):
    """Chẩn đoán (không phải giả thuyết của paper): eo nông ở họ causal có do thiếu ĐIỂM NEO chung giữa các ngôn ngữ không?
    Mỗi nhánh so với đối chứng p1p2 (LM, M4, overlap 0%, cùng họ) ghép theo seed; Wilcoxon một phía theo chiều dự đoán:
      encoder bỏ [MASK] -> eo NÔNG hơn | decoder che 15% đầu vào bằng [MASK] -> SÂU hơn | decoder có điểm neo 5% / 20% -> SÂU hơn."""
    base = sel(df, grid="p1p2", level="M4", task="LM", overlap=0)
    arms = [("encoder bỏ [MASK]", "encoder", sel(df, grid="mask_ablation", family="encoder"), -1),
            ("decoder che 15% đầu vào", "decoder", sel(df, grid="mask_ablation", family="decoder"), 1)]
    if "anchor" in df.columns:
        arms += [(f"decoder điểm neo {a:g}", "decoder", sel(df, grid="anchors", anchor=a), 1) for a in (0.05, 0.2)]
    rows = []
    for name, fam, s, sign in arms:
        b, x = sel(base, family=fam).set_index("seed"), s.set_index("seed")
        sd = sorted(set(b.index) & set(x.index))
        if not sd:
            continue
        dS = (x.loc[sd, "S_edge"] - b.loc[sd, "S_edge"]).values
        rows.append(dict(arm=name, family=fam, n_seed=len(sd), expected="sâu hơn" if sign > 0 else "nông hơn",
                         S_edge_base=float(b.loc[sd, "S_edge"].median()), S_edge_arm=float(x.loc[sd, "S_edge"].median()),
                         G_peak_base=float(b.loc[sd, "G_at_kstar"].median()), G_peak_arm=float(x.loc[sd, "G_at_kstar"].median()),
                         waist_base=float(b.loc[sd, "has_waist"].mean()), waist_arm=float(x.loc[sd, "has_waist"].mean()),
                         p=wilcoxon_p(sign * dS)))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("D_anchor", NO_DATA, "")
    ev = "; ".join(f"{r.arm}: S_edge {r.S_edge_base:.3f} -> {r.S_edge_arm:.3f} (dự đoán {r.expected}, p={r.p:.3g}), "
                   f"có eo {r.waist_base:.0%} -> {r.waist_arm:.0%}" for r in t.itertuples())
    ok = t.p < 0.05
    verdict = "ủng hộ" if (len(t) == 4 and ok.all()) else ("không ủng hộ" if not ok.any() else "chưa kết luận")
    return _res("D_anchor", verdict, ev, t)

def run_all_tests(df):
    return [f(df) for f in (test_P1, test_P2, test_S3, test_anchor)]
