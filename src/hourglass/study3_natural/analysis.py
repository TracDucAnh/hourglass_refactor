# Đặc trưng eo, ΔA, kiểm định P3 / S2 / S4, P4 / S5, pilot, báo cáo, hình (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(); chỉ cần numpy / pandas (không cần GPU). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
# Mọi kết luận là ƯỚC LƯỢNG qua seed (trung bình, CI95 t qua seed; seed = đơn vị lặp độc lập), như Study 2a.
import pandas as pd
from scipy import stats

def primary_readout(family):
    """Cách đọc chính, chốt trước: READOUT="family" -> causal đọc ở token cuối, hai chiều đọc trung bình (như 2a)."""
    return READOUT if READOUT != "family" else ("last" if family in CAUSAL else "mean")

def _mci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    m = float(x.mean())
    if len(x) < 2:
        return m, np.nan, np.nan
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
    return m, m - h, m + h

def _fmt(m, lo, hi, d=3):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"

_NPZ = {}
def _ld(rid, name):
    key = (rid, name)
    if key not in _NPZ:
        p = f"{RUN_DIR}/{rid}/{name}.npz"
        _NPZ[key] = None if not os.path.exists(p) else dict(np.load(p))
    return _NPZ[key]

_GRID_IDS = {}
def in_grid(rid, g):
    if g not in _GRID_IDS:
        _GRID_IDS[g] = {run_id(c) for c in grid_configs(g)}
    return rid in _GRID_IDS[g]

def scan_runs():
    rows = []
    for rid in sorted(os.listdir(RUN_DIR)) if os.path.isdir(RUN_DIR) else []:
        m = read_json(f"{RUN_DIR}/{rid}/meta.json", {}) or {}
        if m.get("status") == "ok" and m.get("train_version") == TRAIN_VERSION and m.get("metric_version") == METRIC_VERSION:
            rows.append(dict(rid=rid, **{k: v for k, v in m["cfg"].items() if k != "grid"}, val_loss=m["val_loss"],
                             val_by_lang=m["val_by_lang"], conv_drop=m["conv_drop"], converged=m["converged"],
                             params=m["params_nonemb"], langs=m["langs"]))
    return pd.DataFrame(rows)

_TWA = {}
def _twin_A(rid_a, rid_b, r):
    key = (rid_a, rid_b, r)
    if key not in _TWA:
        za, zb = _ld(rid_a, "trained"), _ld(rid_b, "trained")
        _TWA[key] = None if (za is None or zb is None) else twin_A(za[f"Z_{r}"].astype(np.float32), zb[f"Z_{r}"].astype(np.float32))
    return _TWA[key]

def joint_features(runs):
    """Một dòng mỗi run joint: đặc trưng eo theo cách đọc chính (và S_edge của cách đọc còn lại), A ở eo, ΔA ở eo
    (trung bình cặp; cặp cùng / khác trật tự từ cơ bản), độ kém của twin so với mô hình chung."""
    twins = runs[runs.kind == "twin"].set_index("rid") if len(runs) else runs
    rows = []
    for r in runs[runs.kind == "joint"].itertuples() if len(runs) else []:
        tr, ini = _ld(r.rid, "trained"), _ld(r.rid, "init")
        if tr is None:
            continue
        ro = primary_readout(r.family)
        f = curve_features(np.nanmean(tr[f"G_{ro}"], 0))
        row = dict(rid=r.rid, family=r.family, d=r.d, L=r.L, nest=r.nest, frac=r.frac, tok=r.tok, share=r.share, seed=r.seed,
                   params=r.params, val_loss=r.val_loss, conv_drop=r.conv_drop, converged=r.converged, readout=ro, **f)
        row["S_edge_init"] = curve_features(np.nanmean(ini[f"G_{ro}"], 0))["S_edge"] if ini is not None else np.nan
        row["dS"] = row["S_edge"] - row["S_edge_init"]
        alt = "mean" if ro == "last" else "last"
        row[f"S_edge_{alt}"] = curve_features(np.nanmean(tr[f"G_{alt}"], 0))["S_edge"]
        Aj, k = tr[f"A_{ro}"], f["k_star"]
        row["A_waist"] = float(np.nanmean(Aj[:, k]))
        langs, pairs = r.langs, [tuple(p) for p in tr["pairs"]]
        tw = {l: run_id(make_cfg("", r.family, "twin", r.d, lang=l, frac=r.frac, tok=r.tok, seed=r.seed)) for l in langs}
        if r.share == "eq" and all(t in twins.index for t in tw.values()):
            dA = np.full_like(Aj, np.nan)
            for pi, (a, b) in enumerate(pairs):
                ai = _twin_A(tw[langs[a]], tw[langs[b]], ro)
                if ai is not None:
                    dA[pi] = Aj[pi] - ai
            same = np.array([LANG_INFO[langs[a]]["order"] == LANG_INFO[langs[b]]["order"] for a, b in pairs])
            m = np.nanmean(dA, 0)
            row.update(dA_waist=float(m[k]), dA_peak_layer=int(np.nanargmax(m)), frac_pairs_pos=float((dA[:, k] > 0).mean()),
                       dA_same_order=float(np.nanmean(dA[same, k])) if same.any() else np.nan,
                       dA_diff_order=float(np.nanmean(dA[~same, k])) if (~same).any() else np.nan,
                       twin_excess=float(np.mean([twins.loc[tw[l], "val_loss"] - r.val_by_lang[l] for l in langs])))
        rows.append(row)
    return pd.DataFrame(rows)

def _res(hyp, verdict, evidence, table=None):
    return dict(hyp=hyp, verdict=verdict, evidence=evidence, table=table)

# ---------------- P3 (phần 2b): ΔA ở eo theo |L| và theo d ----------------
def test_P3(df):
    """P3 (2b): mô hình chung có căn chỉnh hơn hai twin độc lập (ΔA > 0 ở eo) và ΔA ở eo TĂNG theo |L| khi giữ d và
    lượng dữ liệu mỗi ngôn ngữ. Kết luận chính: decoder, thứ tự lồng A. Bác (A2) nếu mọi CI của ΔA nằm trong
    ±max(DELTA_A_MARGIN, 1 SD seed). B (dồn khác trật tự từ về cuối), MoE, diffusion, và tách cặp cùng / khác trật
    tự là mô tả. Run có twin kém hơn TWIN_EXCESS_MAX được báo riêng (ΔA có thể bị thổi phồng)."""
    s = df[df.rid.map(lambda x: in_grid(x, "p3") or in_grid(x, "p3_families")) & df.dA_waist.notna()] if "dA_waist" in df else df.iloc[0:0]
    if s.empty:
        return _res("P3", "chưa có dữ liệu", "")
    rows = []
    for (fam, d, nest, L), g in s.groupby(["family", "d", "nest", "L"]):
        m, lo, hi = _mci(g.dA_waist)
        rows.append(dict(part="ΔA ở eo", family=fam, d=d, nest=nest, L=L, n_seed=len(g), est=m, lo=lo, hi=hi,
                         sd_seed=float(g.dA_waist.std(ddof=1)) if len(g) > 1 else np.nan, frac_pairs_pos=float(g.frac_pairs_pos.mean()),
                         same_order=float(g.dA_same_order.mean()), diff_order=float(g.dA_diff_order.mean()),
                         twin_excess=float(g.twin_excess.mean()), bad_twins=int((g.twin_excess > TWIN_EXCESS_MAX).sum())))
    for (fam, d, nest), g in s.groupby(["family", "d", "nest"]):
        sl = [stats.linregress(np.log2(x.L), x.dA_waist).slope for _, x in g.groupby("seed") if x.L.nunique() >= 2]
        m, lo, hi = _mci(sl)
        rows.append(dict(part="độ dốc ΔA theo log2 |L|", family=fam, d=d, nest=nest, L=np.nan, n_seed=len(sl), est=m, lo=lo, hi=hi))
    t = pd.DataFrame(rows)
    main = t[(t.family == "decoder") & (t.nest == "A")]
    lev, slo = main[main.part == "ΔA ở eo"], main[main.part != "ΔA ở eo"]
    if lev.empty:
        return _res("P3", "chưa có dữ liệu", "chưa có run decoder, thứ tự lồng A", t)
    margin = max(DELTA_A_MARGIN, float(np.nanmax(lev.sd_seed)) if lev.sd_seed.notna().any() else 0.0)
    pos = bool((lev.lo > 0).all()); rises = bool(len(slo) and (slo.lo > 0).all())
    null = bool(((lev.lo >= -margin) & (lev.hi <= margin)).all())
    ev = ("decoder, A: ΔA ở eo " + "; ".join(f"d={r.d} |L|={int(r.L)}: {_fmt(r.est, r.lo, r.hi)}" for r in lev.itertuples())
          + " | độ dốc theo log2|L|: " + "; ".join(f"d={r.d}: {_fmt(r.est, r.lo, r.hi)}" for r in slo.itertuples())
          + f" | cặp cùng / khác trật tự từ (trung bình): {lev.same_order.mean():+.3f} / {lev.diff_order.mean():+.3f}"
          + f" | twin kém > {TWIN_EXCESS_MAX}: {int(lev.bad_twins.sum())} run")
    verdict = "ủng hộ" if (pos and rises) else ("không ủng hộ (ΔA ≈ 0, phù hợp A2)" if null else "chưa kết luận")
    return _res("P3", verdict, ev, t)

# ---------------- S2: bậc thang theo capacity ----------------
def _aicc(sse, n, k):
    return n * math.log(max(sse, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

def test_S2(df):
    """S2: căn chỉnh ở eo (A ở eo) theo log d là hàm bậc thang, và phương sai giữa seed đạt đỉnh ở bước nhảy. So mô hình
    tuyến tính với mô hình một bậc (điểm gãy tốt nhất) bằng AICc trên dữ liệu từng seed."""
    s = df[df.rid.map(lambda x: in_grid(x, "s2"))]
    if s.d.nunique() < 4 if len(s) else True:
        return _res("S2", "chưa có dữ liệu", "")
    x, y = np.log2(s.d.values.astype(float)), s.A_waist.values
    lin = np.polyfit(x, y, 1); sse_lin = float(((y - np.polyval(lin, x)) ** 2).sum())
    ds = sorted(s.d.unique()); best = None
    for i in range(1, len(ds)):
        cut = math.log2(ds[i]); yy = np.where(x < cut, y[x < cut].mean(), y[x >= cut].mean())
        sse = float(((y - yy) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, ds[i - 1], ds[i])
    d_aicc = _aicc(sse_lin, len(y), 2) - _aicc(best[0], len(y), 3)
    t = s.groupby("d").A_waist.agg(["mean", "std", "count"]).reset_index()
    d_sd = int(t.loc[t["std"].idxmax(), "d"])
    near = min(abs(ds.index(d_sd) - ds.index(best[1])), abs(ds.index(d_sd) - ds.index(best[2]))) <= 1
    ev = (f"ΔAICc (tuyến tính - một bậc) = {d_aicc:+.2f}; bước nhảy giữa d={best[1]} và d={best[2]}; "
          f"SD giữa seed lớn nhất ở d={d_sd} ({'gần' if near else 'xa'} bước nhảy)")
    verdict = "ủng hộ" if (d_aicc >= 2 and near) else ("không ủng hộ" if d_aicc <= -2 else "chưa kết luận")
    return _res("S2", verdict, ev, t)

# ---------------- S4: pivot theo chi phí hay theo dữ liệu ----------------
def test_S4(df):
    """S4: pivot (logit-lens quanh eo) là ngôn ngữ TỐN ÍT TOKEN nhất (LE-D) hay ngôn ngữ NHIỀU DỮ LIỆU nhất (A3).
    Ô phân định: X nhiều dữ liệu nhưng tốn token (A3 -> X, LE-D -> ngôn ngữ rẻ nhất khác X) và dữ liệu đều với X rẻ
    (LE-D -> X). Pivot(l) = trung bình qua b != l của M[k, b, l] trên các tầng k* - 1 .. k* + 1."""
    s = df[df.rid.map(lambda x: in_grid(x, "s4"))]
    if s.empty:
        return _res("S4", "chưa có dữ liệu", "")
    rows = []
    for r in s.itertuples():
        tr = _ld(r.rid, "trained")
        if tr is None or "lens" not in tr:
            continue
        langs = NESTINGS[r.nest][:r.L]; M = tr["lens"]
        ks = range(max(0, r.k_star - 1), min(M.shape[0], r.k_star + 2))
        score = [np.mean([M[k, b, l] for k in ks for b in range(len(langs)) if b != l]) for l in range(len(langs))]
        cost = token_cost(r.tok, langs)
        rows.append(dict(rid=r.rid, tok=r.tok, share=r.share, seed=r.seed, pivot=langs[int(np.argmax(score))],
                         cheapest=min(cost, key=cost.get), most_data=S4_LANG if r.share != "eq" else None,
                         cost_X=cost[S4_LANG], cost_min=min(cost.values())))
    t = pd.DataFrame(rows)
    if t.empty:
        return _res("S4", "chưa có dữ liệu", "chưa có logit-lens")
    t["led_ok"] = t.pivot == t.cheapest
    t["a3_ok"] = t.most_data.notna() & (t.pivot == t.most_data)
    cell = t.groupby(["share", "tok"]).agg(n=("rid", "size"), led=("led_ok", "mean"), a3=("a3_ok", "mean"),
                                           pivots=("pivot", lambda p: dict(Counter(p)))).reset_index()
    ev = "; ".join(f"dữ liệu {c.share}, tokenizer {c.tok}: pivot {c.pivots}, khớp LE-D {c.led:.0%}"
                   + (f", khớp A3 {c.a3:.0%}" if c.share != "eq" else "") for c in cell.itertuples())
    ok = bool((cell.led >= 2 / 3).all()); a3 = bool((cell[cell.share != "eq"].a3 > 0.5).any())
    return _res("S4", "ủng hộ" if ok else ("không ủng hộ (pivot theo dữ liệu, A3)" if a3 else "chưa kết luận"), ev, cell)

# ---------------- S5 (phụ) và P4 (phụ) ----------------
def test_S5(df):
    """S5 (phụ): rho* tương quan dương với chi phí token trung bình (cùng họ, cùng d): dùng các tokenizer khác nhau của S4."""
    s = df[df.rid.map(lambda x: in_grid(x, "s4"))]
    if s.tok.nunique() < 2 if len(s) else True:
        return _res("S5", "chưa có dữ liệu", "")
    cost = {k: float(np.mean(list(token_cost(k, NESTINGS["A"][:4]).values()))) for k in s.tok.unique()}
    rho, p = stats.spearmanr(s.tok.map(cost), s.rho_star)
    return _res("S5", "bằng chứng phụ", f"Spearman(chi phí token trung bình, rho*) = {rho:+.2f} (p={p:.3g}, n={len(s)})")

def p4_holdout(df):
    """P4 (phụ): hiệu chỉnh ΔA ở eo ~ log2|L| + log2 d trên p3 (decoder, A), dự đoán các ô của holdout_p4; so MAE với
    dự đoán hằng số (trung bình p3). Xuất bảng đầu vào cho Study 5."""
    fit = df[df.rid.map(lambda x: in_grid(x, "p3")) & (df.family == "decoder") & (df.nest == "A")].dropna(subset=["dA_waist"]) if "dA_waist" in df else df.iloc[0:0]
    ho = df[df.rid.map(lambda x: in_grid(x, "holdout_p4") and not in_grid(x, "p3"))].dropna(subset=["dA_waist"]) if "dA_waist" in df else df.iloc[0:0]
    if len(fit) < 4 or ho.empty:
        return _res("P4", "chưa có dữ liệu", "")
    X = lambda z: np.c_[np.ones(len(z)), np.log2(z.L), np.log2(z.d)]
    beta = np.linalg.lstsq(X(fit), fit.dA_waist, rcond=None)[0]
    mae = float(np.abs(X(ho) @ beta - ho.dA_waist).mean()); mae0 = float(np.abs(fit.dA_waist.mean() - ho.dA_waist).mean())
    return _res("P4", "bằng chứng phụ", f"ΔA ở eo trên {len(ho)} run giữ ra: MAE {mae:.3f} (hàm log2|L| + log2 d) so với {mae0:.3f} (hằng số)")

def pilot_table(df):
    """pilot_data: đặc trưng theo phần dữ liệu -> chọn DATA_FRAC nhỏ nhất mà đặc trưng ổn định và twin đủ tốt."""
    s = df[df.rid.map(lambda x: in_grid(x, "pilot_data"))]
    if s.empty:
        return None
    cols = [c for c in ("val_loss", "S_edge", "rho_star", "A_waist", "dA_waist", "twin_excess") if c in s]
    return s.groupby("frac")[cols].agg(["mean", "std"]).round(4)

# ---------------- hình ----------------
def make_figures(df):
    import matplotlib.pyplot as plt
    s = df[df.rid.map(lambda x: in_grid(x, "p3"))].dropna(subset=["dA_waist"]) if "dA_waist" in df else df.iloc[0:0]
    if len(s):
        nests = sorted(s.nest.unique())
        fig, axes = plt.subplots(1, len(nests), figsize=(3.4 * len(nests) + 0.4, 3.0), squeeze=False)
        for ax, n in zip(axes[0], nests):
            for d, g in s[(s.nest == n) & (s.family == "decoder")].groupby("d"):
                a = g.groupby("L").dA_waist.apply(lambda v: _mci(v)).apply(pd.Series)
                ax.errorbar(a.index, a[0], yerr=[a[0] - a[1], a[2] - a[0]], marker="o", capsize=3, label=f"d={d}")
            ax.axhline(0, color="k", lw=0.8, ls=":"); ax.set_xscale("log", base=2); ax.set_xlabel(r"number of languages $|\mathbb{L}|$")
            ax.set_title(f"nesting {n}", fontsize=9); ax.set_ylabel(r"$\Delta A$ at the waist")
        axes[0, 0].legend(frameon=False, fontsize=8); fig.tight_layout()
        fig.savefig(f"{FIG_DIR}/fig_P3_deltaA_by_languages.pdf", bbox_inches="tight"); plt.close(fig)
    s = df[df.rid.map(lambda x: in_grid(x, "s2"))]
    if len(s):
        fig, ax = plt.subplots(figsize=(3.8, 3.0))
        ax.plot(s.d, s.A_waist, ".", color="0.6", ms=4)
        g = s.groupby("d").A_waist.agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"], marker="o", capsize=3, color="C0")
        ax.set_xscale("log", base=2); ax.set_xlabel("width $d$"); ax.set_ylabel("alignment $A$ at the waist")
        fig.tight_layout(); fig.savefig(f"{FIG_DIR}/fig_S2_alignment_by_width.pdf", bbox_inches="tight"); plt.close(fig)

# ---------------- báo cáo ----------------
def regenerate_all():
    runs = scan_runs()
    if runs.empty:
        print("Chưa có run nào hoàn thành trong", RUN_DIR); return None, []
    df = joint_features(runs)
    df.to_csv(f"{ANALYSIS_DIR}/features_joint_runs.csv", index=False)
    res = [f(df) for f in (test_P3, test_S2, test_S4, test_S5, p4_holdout)]
    pd.DataFrame([dict(hypothesis=r["hyp"], verdict=r["verdict"], evidence=r["evidence"]) for r in res]).to_csv(
        f"{ANALYSIS_DIR}/hypotheses_summary.csv", index=False)
    for r in res:
        if r["table"] is not None and len(r["table"]):
            r["table"].to_csv(f"{ANALYSIS_DIR}/table_{r['hyp']}.csv", index=False)
    pil = pilot_table(df)
    if pil is not None:
        pil.to_csv(f"{ANALYSIS_DIR}/table_pilot.csv")
    p4 = df.assign(holdout=df.rid.map(lambda x: in_grid(x, "holdout_p4") and not in_grid(x, "p3")))
    p4.to_csv(f"{ANALYSIS_DIR}/p4_inputs_for_study5.csv", index=False)
    try:
        make_figures(df)
    except Exception as e:
        print("[figures] lỗi:", repr(e)[:200])
    tw = runs[runs.kind == "twin"]
    L_ = [f"# Báo cáo Study 2b — ngôn ngữ tự nhiên ({datetime.now():%Y-%m-%d %H:%M})\n",
          f"> Nguồn dữ liệu: {DATA_SOURCE}. Cách đọc chính: {READOUT} (causal = token cuối, hai chiều = trung bình). "
          "Kết luận là ước lượng qua seed (CI95 t).\n",
          "## Kết luận theo giả thuyết\n", "| giả thuyết | kết luận | bằng chứng |", "|---|---|---|"]
    L_ += [f"| **{r['hyp']}** | {r['verdict']} | {r['evidence']} |" for r in res]
    L_ += ["\n## Hội tụ\n", "| loại | họ | run | hội tụ |", "|---|---|---|---|"]
    for (k, f), g in runs.groupby(["kind", "family"]):
        L_.append(f"| {k} | {f} | {len(g)} | {int(g.converged.sum())} |")
    if "twin_excess" in df and df.twin_excess.notna().any():
        L_.append(f"\nTwin kém mô hình chung (trung vị) {df.twin_excess.median():+.3f} nats; "
                  f"{int((df.twin_excess > TWIN_EXCESS_MAX).sum())} run joint có twin kém hơn {TWIN_EXCESS_MAX} (ΔA có thể bị thổi phồng).")
    if pil is not None:
        L_ += ["\n## Pilot dữ liệu (chọn DATA_FRAC)\n", "```", pil.to_string(), "```"]
    L_.append(f"\n{len(runs)} run ({len(tw)} twin). Hình: {sorted(os.listdir(FIG_DIR))}")
    txt = "\n".join(L_)
    with open(f"{ANALYSIS_DIR}/report_study2b.md", "w", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return df, res
