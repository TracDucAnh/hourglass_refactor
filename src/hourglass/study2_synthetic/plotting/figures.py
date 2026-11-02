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

def fig_A1(df):
    levels = [lv for lv in ("M0", "M2", "M4") if len(sel(df, family="decoder", overlap=100, level=lv))]
    if not levels:
        return print("[A1] chưa có dữ liệu (lưới balanced)")
    fig, axes = plt.subplots(2, len(levels), figsize=(3.0 * len(levels) + 0.4, 5.0), squeeze=False)
    for j, lv in enumerate(levels):
        for ov, col in ((0, C["o0"]), (100, C["o100"])):
            for t, ls in (("LM", "-"), ("SEM", "--")):
                s = sel(df, family="decoder", level=lv, overlap=ov, task=t)
                draw(axes[0, j], s, "G", col, ls, mark_peak=True); draw(axes[1, j], s, "U", col, ls, norm_L=True)
        axes[0, j].set_ylim(-0.05, 1.0); axes[1, j].set_ylim(-0.05, 1.05); axes[0, j].set_xticklabels([])
        _xlab(axes[1, j]); subcap(axes[1, j], f"({'abc'[j]}) level {lv}")
    axes[0, 0].set_ylabel(r"alignment $\bar G_k$"); axes[1, 0].set_ylabel(r"$U_k/\log|\mathcal{L}|$")
    legend_top(fig, _lines([(C["o0"], "-", "phoneme overlap 0%"), (C["o100"], "-", "overlap 100% (balanced interface)"),
                            ("k", "-", "LM"), ("k", "--", "SEM")]), ncol=4)
    fig.tight_layout(); save_fig(fig, "fig_A1_balanced_interface")

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

def fig_P3(df):
    s = df[df.has_twins & df.grid.isin(["p3s2", "smoke"])]
    if s.empty:
        return print("[P3] chưa có dữ liệu sinh đôi")
    def dA_curve(rows):
        M = [np.nanmean(x, 0) for x in (_delta_A(r) for r in rows.itertuples()) if x is not None]
        return np.array(M)
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    d0 = 128 if (s.d == 128).any() else int(s.d.min())
    for L in sorted(s.L.unique()):
        M = dA_curve(s[(s.d == d0) & (s.L == L)])
        if M.size:
            m, lo, hi = tcrit_band(M); axes[0].plot(RHO, m, color=LCOL.get(L), lw=1.6, label=f"|L|={L}"); axes[0].fill_between(RHO, lo, hi, color=LCOL.get(L), alpha=0.15, lw=0)
    Lm = 4 if (s.L == 4).any() else int(s.L.min())
    for d, col in zip(sorted(s.d.unique()), plt.cm.viridis(np.linspace(0, 0.9, s.d.nunique()))):
        M = dA_curve(s[(s.d == d) & (s.L == Lm)])
        if M.size:
            m, lo, hi = tcrit_band(M); axes[1].plot(RHO, m, color=col, lw=1.6, label=f"d={d}"); axes[1].fill_between(RHO, lo, hi, color=col, alpha=0.15, lw=0)
    t = test_P3(df)["table"]
    if t is not None and len(t):
        for L in sorted(t.L.unique()):
            g = t[t.L == L].groupby("d").agg(x=("ln_params", "mean"), m=("dA_kstar", "mean"), sd=("dA_kstar", "std")).reset_index()
            axes[2].errorbar(g.x, g.m, yerr=g.sd.fillna(0), fmt="-o", color=LCOL.get(L), capsize=3, ms=4, label=f"|L|={L}")
    for ax in axes[:2]:
        ax.axhline(0, color="k", lw=0.8); _xlab(ax); ax.set_ylabel(r"$\Delta A=A_{joint}-A_{indep}$"); ax.legend(frameon=False)
    axes[2].axhline(0, color="k", lw=0.8); axes[2].set_xlabel(r"$\ln B$ (non-embedding params)"); axes[2].set_ylabel(r"$\Delta A$ at waist"); axes[2].legend(frameon=False)
    subcap(axes[0], f"(a) by |L| (d={d0})"); subcap(axes[1], f"(b) by capacity (|L|={Lm})"); subcap(axes[2], "(c) ΔA at waist vs capacity")
    fig.tight_layout(); save_fig(fig, "fig_P3_joint_vs_twins")

def fig_S2(df):
    s = sel(df, grid="p3s2", family="decoder")
    if s.empty:
        return print("[S2] chưa có dữ liệu")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for L in sorted(s.L.unique()):
        x = s[s.L == L]; g = x.groupby("d").G_at_kstar.agg(["mean", "std", "count"]).reset_index()
        axes[0].plot(np.log2(x.d), x.G_at_kstar, ".", color=LCOL.get(L), alpha=0.25, ms=4)
        axes[0].plot(np.log2(g.d), g["mean"], "-o", color=LCOL.get(L), lw=1.6, ms=4, label=f"|L|={L}")
        axes[1].plot(np.log2(g.d), g["std"], "-s", color=LCOL.get(L), lw=1.6, ms=4, label=f"|L|={L}")
    for ax in axes:
        ax.set_xlabel(r"$\log_2 d$ (capacity)"); ax.legend(frameon=False)
    axes[0].set_ylabel(r"$\bar G_{k^*}$ at waist"); axes[1].set_ylabel(r"SD across seeds")
    subcap(axes[0], "(a) alignment at the waist"); subcap(axes[1], "(b) seed variance")
    fig.tight_layout(); save_fig(fig, "fig_S2_staircase")

def fig_S3(df):
    s = df[df.grid.isin(["p3s2", "smoke"]) & (df.family == "decoder")]
    if s.empty:
        return print("[S3] chưa có dữ liệu")
    Lm = 4 if (s.L == 4).any() else int(s.L.min()); s4 = s[s.L == Lm]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2)); xs = np.arange(N_RANK_BINS)
    for d, col in zip(sorted(s4.d.unique()), plt.cm.viridis(np.linspace(0, 0.9, s4.d.nunique()))):
        Ms = [cv(r.rid)["Grank"][r.k_star] for r in s4[s4.d == d].itertuples() if cv(r.rid) is not None and "Grank" in cv(r.rid)]
        if Ms:
            m, lo, hi = tcrit_band(np.array(Ms)); axes[0].plot(xs, m, "-o", color=col, ms=3, lw=1.5, label=f"d={d}"); axes[0].fill_between(xs, lo, hi, color=col, alpha=0.12, lw=0)
    axes[0].set_xlabel(r"concept frequency rank $\lfloor\log_2 r\rfloor$ (0 = most frequent)"); axes[0].set_ylabel(r"alignment $G$ at the waist"); axes[0].legend(frameon=False, ncol=2)
    t = test_S3(df)["table"]
    if t is not None and len(t):
        sc = axes[1].scatter(t.ln_params, t.peak_bin, c=t.L, cmap="viridis", s=14)
        axes[1].set_xlabel(r"$\ln B$"); axes[1].set_ylabel("rank bin of the peak"); fig.colorbar(sc, ax=axes[1], label="|L|")
    subcap(axes[0], "(a) alignment vs concept rank"); subcap(axes[1], "(b) peak position vs capacity")
    fig.tight_layout(); save_fig(fig, "fig_S3_frequency_inverted_U")

def fig_S4(df):
    t = test_S4(df)["table"]
    if t is None or t.empty:
        return print("[S4] chưa có dữ liệu")
    conds = [(s_, f_) for s_ in ("đều", "40%") for f_ in ("thấp", "cao")]
    names = {"đều": "equal data", "40%": "X: 40% data", "thấp": "low fertility", "cao": "high fertility"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
    for i, (s_, f_) in enumerate(conds):
        v = t[(t.share == s_) & (t.fert == f_)].diff_X.values
        if len(v):
            axes[0].bar(i, v.mean(), color="#9ecae1" if f_ == "thấp" else "#fdae6b", edgecolor="k", lw=0.6)
            axes[0].plot(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(v)), v, "k.", ms=4)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].set_xticks(range(4)); axes[0].set_xticklabels([f"{names[a]}\n{names[b]}" for a, b in conds], fontsize=7)
    axes[0].set_ylabel("pivot score of X minus mean of others")
    sub = sel(df, grid="s4"); cols = {}
    for (s_, f_), col in zip(conds, ["#08519c", "#6baed6", "#a63603", "#fd8d3c"]):
        ys = []
        for r in sub.itertuples():
            if ("40%" if _has(r.share_X) else "đều") == s_ and ("thấp" if r.len_scale_X < 1 else "cao") == f_:
                ln = _ld(r.rid, "lens")
                if ln is not None:
                    M = ln["M"]; ys.append([M[k, 1:, 0].mean() for k in range(M.shape[0])])
        if ys:
            m, lo, hi = tcrit_band(np.array(ys)); axes[1].plot(RHO, m, color=col, lw=1.6, label=f"{names[s_]}, {names[f_]}"); axes[1].fill_between(RHO, lo, hi, color=col, alpha=0.12, lw=0)
    _xlab(axes[1]); axes[1].set_ylabel("logit-lens mass on X tokens (input in other languages)"); axes[1].legend(frameon=False, fontsize=7)
    subcap(axes[0], "(a) pivot score at the waist"); subcap(axes[1], "(b) X-mass across depth")
    fig.tight_layout(); save_fig(fig, "fig_S4_cost_driven_pivot")

def fig_S5(df):
    s = sel(df, grid="s5")
    if s.empty:
        return print("[S5] chưa có dữ liệu")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2)); lss = sorted(s.len_scale.unique())
    sc = axes[0].scatter(s.fert_mean, s.rho_star, c=s.len_scale, cmap="viridis", s=18)
    if len(s) > 3:
        b, a = np.polyfit(s.fert_mean, s.rho_star, 1); xx = np.linspace(s.fert_mean.min(), s.fert_mean.max(), 20); axes[0].plot(xx, a + b * xx, "k--", lw=1)
    axes[0].set_xlabel(r"mean tokens per word $\phi$"); axes[0].set_ylabel(r"waist depth $\rho^*$"); fig.colorbar(sc, ax=axes[0], label="len_scale")
    for ls_, col in zip(lss, plt.cm.viridis(np.linspace(0, 0.9, len(lss)))):
        draw(axes[1], s[s.len_scale == ls_], "G", col, label=f"len_scale={ls_:g}", mark_peak=True)
    _xlab(axes[1]); axes[1].set_ylabel(r"alignment $\bar G_k$"); axes[1].legend(frameon=False, fontsize=7)
    subcap(axes[0], r"(a) $\rho^*$ vs fertility"); subcap(axes[1], "(b) alignment curves")
    fig.tight_layout(); save_fig(fig, "fig_S5_tokenizer_depth")

def fig_S6(df):
    s = sel(df, family="diffusion", level="M4", overlap=0)
    if s.empty or "rho_t0.25" not in df.columns:
        return print("[S6] chưa có dữ liệu diffusion")
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2)); tl = MASK_LEVELS
    for t, col in zip(tl, plt.cm.plasma(np.linspace(0.1, 0.8, len(tl)))):
        draw(axes[0], s[s.task == "SEM"] if (s.task == "SEM").any() else s, "G", col, variant="trained" if t == 0 else f"t{t:g}", label=f"t={t:g}", mark_peak=True)
    _xlab(axes[0]); axes[0].set_ylabel(r"alignment $\bar G_k$"); axes[0].legend(frameon=False)
    for task in TASKS:
        x = s[s.task == task]
        if len(x):
            ys = np.stack([x.rho_star.values] + [x[f"rho_t{t:g}"].values for t in tl[1:]], 1)
            m, lo, hi = tcrit_band(ys); axes[1].errorbar(tl, m, yerr=(m - lo), fmt="-o", color=C[task], capsize=3, ms=4, label=task)
    axes[1].axhspan(-RHO_MARGIN, RHO_MARGIN, alpha=0.0); axes[1].set_xlabel("mask level t"); axes[1].set_ylabel(r"waist depth $\rho^*$"); axes[1].legend(frameon=False)
    subcap(axes[0], "(a) alignment by mask level"); subcap(axes[1], r"(b) $\rho^*$ vs mask level")
    fig.tight_layout(); save_fig(fig, "fig_S6_diffusion_mask")

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
    for f in (fig_P1, fig_A1, fig_P2, fig_P3, fig_S2, fig_S3, fig_S4, fig_S5, fig_S6, fig_forest):
        try:
            f(df)
        except Exception as e:
            print(f"[{f.__name__}] lỗi: {repr(e)[:200]}"); traceback.print_exc()
