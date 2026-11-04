# Hình theo giả thuyết (Cell D5)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 39 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D5 — Hình (mỗi hình ứng với một giả thuyết)
# ============================================================
def fig_P1(df):
    df = sel(df, grid=["gate_g2", "p1p2", "moe_diff", "smoke"])                              # không gồm lưới chẩn đoán
    fams = [f for f in FAMILIES if len(sel(df, family=f, level="M4", overlap=0, task=["LM", "SEM"]))]
    if not fams:
        return print("[P1] chưa có dữ liệu")
    fig, axes = plt.subplots(3, len(fams), figsize=(2.6 * len(fams) + 0.4, 6.4), squeeze=False)
    for j, fam in enumerate(fams):
        base = sel(df, family=fam, level="M4", overlap=0)
        for t, ls in (("LM", "-"), ("SEM", "-")):
            s = sel(base, task=t)
            draw(axes[0, j], s, "D", C[t], ls, mark_min=True)
            draw(axes[1, j], s, "G", C[t], ls, mark_peak=True)
            draw(axes[2, j], s, "U", C[t], ls, norm_L=True)
        s = sel(base, task="SEM")
        for i, k in enumerate(("D", "G", "U")):
            draw(axes[i, j], s, k, C["init"], "--", variant="init", norm_L=(k == "U"))
        axes[0, j].set_ylim(0, 2.2); axes[1, j].set_ylim(-0.05, 1.0); axes[2, j].set_ylim(-0.05, 1.05)
        for i in range(3):
            axes[i, j].set_xlim(0, 1)
            if i < 2: axes[i, j].set_xticklabels([])
        _xlab(axes[2, j]); subcap(axes[2, j], f"({'abcd'[j]}) {fam}")
    axes[0, 0].set_ylabel(r"separation $D_k$"); axes[1, 0].set_ylabel(r"alignment $\bar G_k$"); axes[2, 0].set_ylabel(r"$U_k/\log|\mathcal{L}|$")
    legend_top(fig, _lines([(C["LM"], "-", "LM"), (C["SEM"], "-", "SEM"), (C["init"], "--", "untrained (init)")]))
    fig.tight_layout(); save_fig(fig, "fig_P1_four_families")

def fig_P2(df):
    fams = [f for f in ("decoder", "encoder") if len(sel(df, family=f, level="M4", overlap=0, grid="p1p2"))]
    if not fams:
        return print("[P2] chưa có dữ liệu")
    fig, axes = plt.subplots(1, len(fams) + 1, figsize=(3.0 * (len(fams) + 1) + 0.4, 3.2), squeeze=False); axes = axes[0]
    for j, fam in enumerate(fams):
        for t in TASKS:
            draw(axes[j], sel(df, family=fam, level="M4", overlap=0, grid="p1p2", task=t), "G", C[t], mark_peak=True)
        axes[j].set_ylim(-0.05, 1.0); _xlab(axes[j]); subcap(axes[j], f"({'ab'[j]}) {fam}, level M4"); axes[j].set_ylabel(r"alignment $\bar G_k$")
    ax = axes[-1]
    for fi, fam in enumerate(fams):
        for ti, t in enumerate(TASKS):
            s = sel(df, family=fam, level="M4", overlap=0, grid="p1p2", task=t)
            if len(s):
                m, h = s.w.mean(), (stats.t.ppf(0.975, len(s) - 1) * s.w.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0)
                ax.errorbar(fi + (ti - 1) * 0.18, m, yerr=h, fmt="o", color=C[t], capsize=3)
    ax.set_xticks(range(len(fams))); ax.set_xticklabels(fams); ax.set_ylabel(r"waist width $w$"); subcap(ax, f"({'abc'[len(fams)]}) waist width by task")
    legend_top(fig, _lines([(C[t], "-", t) for t in TASKS])); fig.tight_layout(); save_fig(fig, "fig_P2_task_dependence")

def fig_S3(df):
    t = test_S3(df)["table"]
    if t is None or not len(t):
        return print("[S3] chưa có dữ liệu")
    fig, ax = plt.subplots(figsize=(3.8, 3.2)); xs = np.arange(N_RANK_BINS)
    for fam in [f for f in FAMILIES if (t.family == f).any()]:
        Ms = np.array([cv(r.rid)["Grank"][r.k_star] for r in t[t.family == fam].itertuples()])
        m, lo, hi = tcrit_band(Ms)
        ax.plot(xs, m, "-o", color=C[fam], ms=3, lw=1.5, label=fam); ax.fill_between(xs, lo, hi, color=C[fam], alpha=0.12, lw=0)
    ax.set_xlabel(r"concept frequency rank $\lfloor\log_2 r\rfloor$ (0 = most frequent)"); ax.set_ylabel(r"alignment $G$ at the waist")
    ax.legend(frameon=False)
    fig.tight_layout(); save_fig(fig, "fig_S3_frequency_inverted_U")

def fig_forest(df):
    s = sel(df, overlap=0, grid=["p1p2", "moe_diff"])
    if s.empty:
        return print("[forest] chưa có dữ liệu")
    g = s.groupby(["family", "level", "task"]).S_edge.agg(["mean", "std", "count"]).reset_index()
    g["h"] = [stats.t.ppf(0.975, c - 1) * sd / math.sqrt(c) if c > 1 else 0 for sd, c in zip(g["std"].fillna(0), g["count"])]
    g = g.sort_values(["family", "level", "task"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(4.8, 0.22 * len(g) + 1.2))
    for i, r in g.iterrows():
        ax.errorbar(r["mean"], i, xerr=r.h, fmt="o", color=C[r.family], capsize=2, ms=4)
    ax.axvline(WAIST_S_FLOOR, color="k", ls=":", lw=1); ax.set_yticks(range(len(g))); ax.set_yticklabels([f"{r.family} {r.level} {r.task}" for r in g.itertuples()], fontsize=7)
    ax.set_xlabel(r"edge depth $S_{edge}$ (mean, CI95 over seeds)"); ax.invert_yaxis(); fig.tight_layout(); save_fig(fig, "fig_forest_edge_depth")

def make_all_figures(df):
    for f in (fig_P1, fig_P2, fig_S3, fig_forest):
        try:
            f(df)
        except Exception as e:
            print(f"[{f.__name__}] lỗi: {repr(e)[:200]}"); traceback.print_exc()
