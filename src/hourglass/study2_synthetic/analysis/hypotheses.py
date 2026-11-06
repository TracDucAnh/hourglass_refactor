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
def _mci(x):
    """Trung bình và CI95 (t) qua các phần tử của x (ở đây: seed)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - h, m + h

def _waist_label(waist, s_lo):
    """Nhãn MÔ TẢ như Bảng 9 của paper (Study 1), theo cận dưới CI95 của S_edge: rõ >= 0.20 | vừa 0.10-0.20 | nông 0.05-0.10 |
    sát ngưỡng < 0.05. Chỉ để đọc; kết luận theo `waist` (CI của dS > 0, đỉnh trong thân)."""
    if not waist:
        return "không có eo"
    return "rõ" if s_lo >= 0.20 else "vừa" if s_lo >= 0.10 else "nông" if s_lo >= 0.05 else "sát ngưỡng"

def test_P1(df):
    """P1: cả bốn họ có waist trên ngôn ngữ tổng hợp - bằng ƯỚC LƯỢNG, không bỏ phiếu theo run. Đơn vị là seed (trung bình các run
    của seed đó); mỗi (họ, mức, tác vụ) báo, kèm CI95 (t) qua seed:
      dS  = S_edge(trained) - S_edge(init)   eo sâu hơn mô hình chưa huấn luyện bao nhiêu (thêm Wilcoxon một phía)
      rel = S_edge / G_k*                    độ sâu tương đối, không phụ thuộc thang G của từng họ
      rho = k*/K                             vị trí đỉnh
    Có eo <=> CI95 của dS nằm trên 0 và CI95 của rho nằm trong (0, 1). Kết luận chính: M4, LM+SEM (thiết kế P1); các mức,
    tác vụ khác là mô tả. label: nhãn mô tả theo cận dưới CI95 (qua seed) của S_edge, cùng mốc với Bảng 9 của paper
    (ở Study 1 là CI bootstrap theo nội dung của một mô hình). frac_runs_study1 = tỉ lệ run qua tiêu chí từng run của Study 1 (có ngưỡng tuyệt đối 0.05), chỉ đối chiếu.
    gate_g2 không dùng: cùng cấu hình với p1p2 decoder M4/SEM seed 0-2 (tránh đếm lặp)."""
    d0 = sel(df, overlap=0, grid=["p1p2", "moe_diff", "smoke"])
    rows = []
    for fam in FAMILIES:
        for lv in LEVELS:
            for task in ("LM+SEM", "LM", "SEM", "SURF"):
                x = sel(d0, family=fam, level=lv, task=["LM", "SEM"] if task == "LM+SEM" else task)
                if x.empty:
                    continue
                g = x.assign(dS=x.S_edge - x.S_edge_init, rel=x.S_edge / x.G_at_kstar).groupby("seed")[["S_edge", "dS", "rel", "rho_star"]].mean()
                S, dS, rel, rho = _mci(g.S_edge), _mci(g.dS), _mci(g.rel), _mci(g.rho_star)
                has = bool(dS[1] > 0 and rho[1] > 0 and rho[2] < 1)
                rows.append(dict(family=fam, level=lv, task=task, primary=bool(lv == "M4" and task == "LM+SEM"), n_seed=len(g), n_run=len(x),
                                 dS=dS[0], dS_lo=dS[1], dS_hi=dS[2], p_dS=wilcoxon_p(g.dS.values),
                                 rel=rel[0], rel_lo=rel[1], rel_hi=rel[2], rho=rho[0], rho_lo=rho[1], rho_hi=rho[2],
                                 G_peak=float(x.G_at_kstar.mean()), frac_interior=float(x.interior.mean()),
                                 S=S[0], S_lo=S[1], S_hi=S[2], frac_runs_study1=float(x.has_waist.mean()), waist=has,
                                 label=_waist_label(has, S[1])))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P1", NO_DATA, "")
    m = t[t.primary].set_index("family")
    ev = "; ".join(f"{f}: {r.label}, S_edge={r.S:.3f} [{r.S_lo:.3f}, {r.S_hi:.3f}], ΔS={r.dS:.3f} [{r.dS_lo:.3f}, {r.dS_hi:.3f}] (p={r.p_dS:.3g}), S/G*={r.rel:.2f} [{r.rel_lo:.2f}, {r.rel_hi:.2f}], "
                   f"ρ*={r.rho:.2f} [{r.rho_lo:.2f}, {r.rho_hi:.2f}]" for f, r in m.iterrows())
    other = t[(t.task == "LM+SEM") & (t.level != "M4")]
    if len(other):
        ev += " | mức khác (mô tả): " + ", ".join(f"{r.family} {r.level} {r.label}, ΔS={r.dS:.3f} [{r.dS_lo:.3f}, {r.dS_hi:.3f}]" for r in other.itertuples())
    miss = [f for f in FAMILIES if f not in m.index]
    if miss:
        return _res("P1", "chưa kết luận", ev + f"; thiếu dữ liệu: {miss}", t)
    verdict = "ủng hộ" if m.waist.all() else ("không ủng hộ" if (m.dS_hi <= 0).any() else "chưa kết luận")
    return _res("P1", verdict, ev, t)

P2_CONTRASTS = (("SEM", "LM"), ("SEM", "SURF"), ("SURF", "LM"))
P2_METRICS = ("S_edge", "G_at_kstar", "rho_star", "w")      # độ sâu, đỉnh G, vị trí, độ rộng

def test_P2(df):
    """P2: waist đi theo tác vụ - bằng ƯỚC LƯỢNG như P1. Với mỗi (họ, mức) và mỗi cặp tác vụ (SEM-LM, SEM-SURF, SURF-LM): hiệu số
    ghép theo seed của độ sâu S_edge, đỉnh G_k*, vị trí rho* và độ rộng w; báo trung bình từng tác vụ, hiệu số với CI95 (t) qua
    seed, và Wilcoxon hai phía (mô tả mức nhất quán về dấu qua seed; kết luận theo CI).
      độ sâu đổi theo tác vụ <=> CI95 của hiệu S_edge ở ít nhất một cặp không chứa 0
      vị trí dịch             <=> CI95 của hiệu rho* không chứa 0 và |hiệu| > RHO_MARGIN (rho* đã chuẩn hóa theo độ sâu mạng)
      vị trí cố định          <=> CI95 của hiệu rho* ở mọi cặp nằm trong [-RHO_MARGIN, RHO_MARGIN]
      eo cố định              <=> vị trí cố định và độ sâu không đổi
    Kết luận chính: M4 (cả bốn họ); M0 / M2 (decoder, encoder) là mô tả. Đỉnh G và độ rộng chỉ mô tả (độ rộng không xác định
    khi eo nông). gate_g2 không dùng (trùng cấu hình p1p2)."""
    d0 = sel(df, overlap=0, grid=["p1p2", "moe_diff", "smoke"])
    rows = []
    for fam in FAMILIES:
        for lv in LEVELS:
            x = sel(d0, family=fam, level=lv)
            if x.empty:
                continue
            for m in P2_METRICS:
                piv = x.pivot_table(index="seed", columns="task", values=m)
                part = []
                for a, b in P2_CONTRASTS:
                    if a not in piv.columns or b not in piv.columns:
                        continue
                    d = (piv[a] - piv[b]).dropna().values
                    mu, lo, hi = _mci(d)
                    part.append(dict(family=fam, level=lv, metric=m, contrast=f"{a}-{b}", n_seed=len(d), mean_a=float(piv[a].mean()),
                                     mean_b=float(piv[b].mean()), diff=mu, lo=lo, hi=hi, p=wilcoxon_p(d, alt="two-sided")))
                rows += part
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("P2", NO_DATA, "")
    t["excl0"] = (t.lo > 0) | (t.hi < 0)
    summ = {}
    for (fam, lv), g in t.groupby(["family", "level"], sort=False):
        dep, pos = g[g.metric == "S_edge"], g[g.metric == "rho_star"]
        depth = bool(dep.excl0.any())
        shift = bool((pos.excl0 & (pos["diff"].abs() > RHO_MARGIN)).any())
        fixed_pos = bool(len(pos) and ((pos.lo >= -RHO_MARGIN) & (pos.hi <= RHO_MARGIN)).all())
        summ[(fam, lv)] = dict(depth_changes=depth, position_shifts=shift, position_fixed=fixed_pos,
                               task_matters=depth or shift, waist_fixed=fixed_pos and not depth)
    for k in ("depth_changes", "position_shifts", "position_fixed", "task_matters", "waist_fixed"):
        t[k] = [summ[(r.family, r.level)][k] for r in t.itertuples()]
    fmt = lambda g, m: ", ".join(f"{r.contrast} {r.diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]" for r in g[g.metric == m].itertuples())
    m4 = {f: s for (f, lv), s in summ.items() if lv == "M4"}
    parts = []
    for fam, s in m4.items():
        g = t[(t.family == fam) & (t.level == "M4")]
        what = "độ sâu" * s["depth_changes"] + (" + " if s["depth_changes"] and s["position_shifts"] else "") + "vị trí" * s["position_shifts"]
        tag = f"đổi theo tác vụ ({what})" if s["task_matters"] else ("eo cố định" if s["waist_fixed"] else "chưa rõ")
        parts.append(f"{fam}: {tag}; độ sâu {fmt(g, 'S_edge')}; vị trí {fmt(g, 'rho_star')}")
    ev = "M4 - " + "; ".join(parts)
    other = [f"{f} {lv}: " + ("đổi theo tác vụ" if s["task_matters"] else "không rõ") for (f, lv), s in summ.items() if lv != "M4"]
    if other:
        ev += " | mức khác (mô tả): " + ", ".join(other)
    if not m4:
        return _res("P2", "chưa kết luận", ev, t)
    if all(s["waist_fixed"] for s in m4.values()):
        return _res("P2", "không ủng hộ", ev, t)
    return _res("P2", "ủng hộ" if all(s["task_matters"] for s in m4.values()) else "chưa kết luận", ev, t)

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
