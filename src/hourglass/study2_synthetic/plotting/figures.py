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
def fig_P1(df, pool="last"):
    """pool = "last": họ causal (decoder, MoE) đo ở token nội dung cuối (cách đo chính); "mean": họ causal đo theo trung bình
    theo từ ({variant}_mean.npz). Họ hai chiều (encoder, diffusion) luôn theo trung bình theo từ, giống nhau ở hai hình."""
    df = sel(df, grid=["p1p2", "moe_diff", "smoke"])          # không gồm lưới chẩn đoán; gate_g2 trùng cấu hình p1p2
    fams = [f for f in FAMILIES if len(sel(df, family=f, level="M4", overlap=0, task=["LM", "SEM"]))]
    if not fams:
        return print("[P1] chưa có dữ liệu")
    fig, axes = plt.subplots(3, len(fams), figsize=(2.6 * len(fams) + 0.4, 6.4), squeeze=False)
    for j, fam in enumerate(fams):
        base = sel(df, family=fam, level="M4", overlap=0)
        mean_pool = pool == "mean" or fam not in ("decoder", "moe")
        tr, ini = ("trained_mean", "init_mean") if (pool == "mean" and fam in ("decoder", "moe")) else ("trained", "init")
        for t, ls in (("LM", "-"), ("SEM", "-")):
            s = sel(base, task=t)
            draw(axes[0, j], s, "D", C[t], ls, variant=tr, mark_min=True)
            draw(axes[1, j], s, "G", C[t], ls, variant=tr, mark_peak=True)
            draw(axes[2, j], s, "U", C[t], ls, variant=tr, norm_L=True)
        s = sel(base, task="SEM")
        for i, k in enumerate(("D", "G", "U")):
            draw(axes[i, j], s, k, C["init"], "--", variant=ini, norm_L=(k == "U"))
        for i in range(3):
            axes[i, j].set_xlim(0, 1)
            if i < 2: axes[i, j].set_xticklabels([])
        _xlab(axes[2, j]); subcap(axes[2, j], f"({'abcd'[j]}) {fam}, {'word mean' if mean_pool else 'last token'}")
    axes[0, 0].set_ylabel(r"separation $D_k$"); axes[1, 0].set_ylabel(r"alignment $\bar G_k$"); axes[2, 0].set_ylabel(r"$U_k/\log|\mathcal{L}|$")
    legend_top(fig, _lines([(C["LM"], "-", "LM"), (C["SEM"], "-", "SEM"), (C["init"], "--", "untrained (init)")]))
    fig.tight_layout(); save_fig(fig, f"fig_P1_four_families_{'meanpool' if pool == 'mean' else 'lasttoken'}")

def fig_P1_mean(df):
    return fig_P1(df, pool="mean")

def fig_P2(df):
    """P2: đường G theo tác vụ của cả bốn họ (M4), và hai đại lượng P2 dự đoán - độ rộng w và vị trí rho* - theo tác vụ
    (trung bình, CI95 t qua seed). SEM / SURF gồm cả seed 6-11 của p2_seeds."""
    df = sel(df, overlap=0, level="M4", grid=["p1p2", "moe_diff", "p2_seeds", "smoke"])
    fams = [f for f in FAMILIES if len(sel(df, family=f))]
    if not fams:
        return print("[P2] chưa có dữ liệu")
    nf = len(fams)
    fig = plt.figure(figsize=(2.6 * nf + 0.4, 6.2)); gs = fig.add_gridspec(2, 2 * nf)
    for j, fam in enumerate(fams):
        ax = fig.add_subplot(gs[0, 2 * j: 2 * j + 2])
        for t in TASKS:
            draw(ax, sel(df, family=fam, task=t), "G", C[t], mark_peak=True)
        ax.set_xlim(0, 1); _xlab(ax); subcap(ax, f"({'abcd'[j]}) {fam}, level M4")
        if j == 0:
            ax.set_ylabel(r"alignment $\bar G_k$")
    for k, (m, lab, cap) in enumerate((("w", r"waist width $w$", "waist width by task"), ("rho_star", r"waist position $\rho^*$", "waist position by task"))):
        ax = fig.add_subplot(gs[1, k * nf: (k + 1) * nf])
        for fi, fam in enumerate(fams):
            for ti, t in enumerate(TASKS):
                s = sel(df, family=fam, task=t)
                if len(s):
                    mu, lo, hi = _mci(s[m].values)
                    ax.errorbar(fi + (ti - 1) * 0.18, mu, yerr=[[mu - lo], [hi - mu]] if np.isfinite(lo) else None, fmt="o", color=C[t], capsize=3)
        ax.set_xticks(range(nf)); ax.set_xticklabels(fams); ax.set_ylabel(lab); subcap(ax, f"({'ef'[k]}) {cap}")
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
    for f in (fig_P1, fig_P1_mean, fig_P2, fig_S3, fig_forest):
        try:
            f(df)
        except Exception as e:
            print(f"[{f.__name__}] lỗi: {repr(e)[:200]}"); traceback.print_exc()
