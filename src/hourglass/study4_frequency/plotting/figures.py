# Hình Study 4 (Cell D2)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 40 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell D2 — Hình cho Study 4 (chỉ đọc ANALYSIS_DIR, không cần GPU/model). Mỗi hàm tự bỏ qua nếu chưa có kết quả.
#   plot_s3_curves     : R_k theo log-hạng tại tầng waist, mỗi model một panel (GAM + CI, đơn điệu, unimodal, trung bình theo bin)   [S3]
#   plot_peak_vs_depth : đỉnh r*(ρ) và biên độ chữ U ngược theo độ sâu                                                              [S3]
#   plot_heatmaps      : R_k fitted trên mặt phẳng (log-hạng, độ sâu) + đường đỉnh                                                    [S3]
#   plot_capacity      : r* và biên độ theo log capacity                                                                              [Conjecture 2]
#   plot_upos          : hiệu ứng từ loại tại tầng waist (từ chức năng vs từ nội dung)                                                [S3]
#   plot_mantel        : căn chỉnh giữa các cặp ngôn ngữ và khoảng cách loại hình                                                     [loại hình]
#   plot_lang_fertility: căn chỉnh trung bình theo ngôn ngữ vs fertility của tokenizer
# ============================================================
def _ci_curve(z):
    f = z["f_boot_w"].astype(np.float32)
    return z["grid"], np.percentile(f, 50, axis=0), np.percentile(f, 2.5, axis=0), np.percentile(f, 97.5, axis=0)

def plot_s3_curves(keys=None, ncols=4, sharey=True):
    """(v5) sharey=True: mọi panel dùng CHUNG trục y (biên độ thật chỉ ~0.03–0.05 ở XLM-R; trục y riêng từng panel làm các biến thiên nhỏ trông như chữ U lớn).
    Vẽ thêm GAM chỉ-spline (không covariate) nét gạch-chấm để so thẳng với chấm thô (chấm xám = trung bình thô theo bin phân vị)."""
    ks = _models_with(keys=keys)
    if not ks:
        print("[skip] s3 curves: chưa có kết quả"); return
    fs = STACK_FS
    fig, axes, nr = grid_axes(len(ks), ncols, sharey=sharey)
    for ax, k, cap in zip(axes, ks, "abcdefghijkl"):
        z = _load_s3(k); g, m, lo, hi = _ci_curve(z); col = MODEL_COLOR[k]
        ax.errorbar(z["bin_x"], z["bin_y"], yerr=1.96 * z["bin_se"], fmt="o", ms=4.5, color="#9a9a9a", ecolor="#c8c8c8", elinewidth=1.2, zorder=1)
        band(ax, g, m, lo, hi, col, lw=2.8, alpha=0.22)
        if "f_boot_w_bare" in z.files:
            fb = z["f_boot_w_bare"].astype(np.float32); ax.plot(g, np.percentile(fb, 50, axis=0), color=col, ls="-.", lw=2.2, alpha=0.9)
        ax.plot(g, z["mono_y"], color="black", ls="--", lw=1.8)
        ax.plot(g, z["uni_y"], color="black", ls=":", lw=1.8)
        xs = float(np.median(z["x_star_boot"])); kk = int(np.argmin(np.abs(g - xs)))
        ax.axvline(xs, color=col, ls=":", lw=1.4, alpha=0.8)
        ax.scatter([xs], [m[kk]], s=110, color=col, edgecolor="black", zorder=5)
        ax.text(0.03, 0.04, f"range of GAM = {float(m.max() - m.min()):.3f}", transform=ax.transAxes, fontsize=fs["tick"] - 1, color="#333333")
        ax.set_xlabel(X_RANK, fontsize=fs["label"]) if (ax in axes[-ncols:] or len(axes) <= ncols) else None
        if ax in axes[::ncols]:
            ax.set_ylabel(Y_R, fontsize=fs["label"])
        ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=fs["tick"])
        panel_caption(ax, f"({cap}) {MODELS[k]['label']}", fs, cap_y=-0.36)
    h = [Line2D([], [], color="#9a9a9a", marker="o", ls="", ms=6), Line2D([], [], color="#444444", lw=2.8), Line2D([], [], color="#444444", lw=2.2, ls="-."),
         Line2D([], [], color="black", ls="--", lw=1.8), Line2D([], [], color="black", ls=":", lw=1.8),
         Line2D([], [], color="#444444", marker="o", mec="black", ls="", ms=9)]
    fig.legend(h, ["raw mean per rank bin ($\\pm$95%)", "GAM adjusted (95% CI)", "GAM spline only (no covariates)", "best monotone fit", "unimodal fit", "peak $r^{\\ast}$"],
               loc="lower center", bbox_to_anchor=(0.5, 0.87 if nr == 1 else 0.93), ncol=3, fontsize=LEGEND_FS - 2, frameon=False, columnspacing=1.4)
    save_fig(fig, "s3_R_vs_logrank_waist")

def plot_peak_vs_depth(keys=None):
    ks = _models_with(keys=keys)
    if not ks:
        print("[skip] peak vs depth: chưa có kết quả"); return
    fig, axes = make_fig(1, 2)
    for k in ks:
        t = pd.read_csv(f"{ANALYSIS_DIR}/s3_layers_{k}{RUN_TAG}.csv"); col = MODEL_COLOR[k]
        band(axes[0, 0], t.rho, t.x_star, t.x_star_lo, t.x_star_hi, col, label=MODELS[k]["label"])
        band(axes[0, 1], t.rho, t.amp, t.amp_lo, t.amp_hi, col)
        for a in axes[0]:
            a.axvline(float(t.rho.iloc[int(np.load(f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz")["kw"])]), color=col, ls=":", lw=1.3, alpha=0.7)
    axes[0, 1].axhline(0, color="black", lw=1.0, ls=":")
    style_ax(axes[0, 0], X_RHO, r"peak $\log_{10} r^{\ast}$", "(a)", r"$r^{\ast}(\rho)$")
    style_ax(axes[0, 1], X_RHO, "edge depth", "(b)", "peak minus higher end")
    for a in axes[0]:
        a.set_xlim(0, 1)
    h, l = axes[0, 0].get_legend_handles_labels(); add_legend(fig, h, l, ncol=min(4, len(l)))
    save_fig(fig, "s3_peak_vs_depth")

def plot_heatmaps(keys=None, ncols=4):
    ks = _models_with(keys=keys)
    if not ks:
        print("[skip] heatmaps: chưa có kết quả"); return
    fs = STACK_FS; fig, axes, nr = grid_axes(len(ks), ncols, hspace=0.8)
    for ax, k, cap in zip(axes, ks, "abcdefghijkl"):
        z = _load_s3(k); f = z["f_obs"]; g = z["grid"]; K1 = f.shape[0]
        im = ax.imshow(f, origin="lower", aspect="auto", extent=[g[0], g[-1], 0, 1], cmap="viridis")
        t = pd.read_csv(f"{ANALYSIS_DIR}/s3_layers_{k}{RUN_TAG}.csv")
        ax.plot(t.x_star, t.rho, color="white", lw=2.0); ax.fill_betweenx(t.rho, t.x_star_lo, t.x_star_hi, color="white", alpha=0.25, linewidth=0)
        ax.axhline(int(z["kw"]) / (K1 - 1), color="white", ls="--", lw=1.2)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=fs["tick"] - 2)
        ax.set_xlabel(X_RANK, fontsize=fs["label"] - 2)
        if ax in axes[::ncols]:
            ax.set_ylabel(X_RHO, fontsize=fs["label"] - 2)
        ax.tick_params(axis="both", labelsize=fs["tick"])
        panel_caption(ax, f"({cap}) {MODELS[k]['label']}", fs, cap_y=-0.36)
    save_fig(fig, "s3_heatmap_depth_x_rank")

def plot_capacity():
    p = f"{ANALYSIS_DIR}/capacity_table{RUN_TAG}.csv"
    if not os.path.exists(p):
        print("[skip] capacity: chưa có kết quả"); return
    cap = pd.read_csv(p).dropna(subset=["n_params"])
    if len(cap) < 2:
        print("[skip] capacity: cần ≥ 2 model"); return
    S = pd.read_csv(f"{ANALYSIS_DIR}/s3_summary{RUN_TAG}.csv"); S = S[S.subset == "all"].set_index("key")
    fig, axes = make_fig(1, 2); mk = {"encoder": "s", "decoder": "o"}
    for _, r in cap.iterrows():
        z = np.load(f"{ANALYSIS_DIR}/s3_{r.key}{RUN_TAG}.npz"); lb = math.log10(r.n_params)
        lo, hi = np.percentile(z["x_star_boot"], [2.5, 97.5]); col = MODEL_COLOR[r.key]; m = mk.get(r.family, "D")
        axes[0, 0].errorbar(lb, r.x_obs, yerr=[[max(r.x_obs - lo, 0)], [max(hi - r.x_obs, 0)]], fmt=m, ms=12, color=col, mfc=(col if bool(r.get("peak_ok", True)) else "white"), mec="black", capsize=4, lw=2, label=r.label)
        s = S.loc[r.key]
        axes[0, 1].errorbar(lb, s.amp, yerr=[[max(s.amp - s.amp_lo, 0)], [max(s.amp_hi - s.amp, 0)]], fmt=m, ms=12, color=col, mec="black", capsize=4, lw=2)
    for fam, g in cap.groupby("family"):
        if len(g) >= 2:
            lb = np.log10(g.n_params.values.astype(float)); o = np.argsort(lb)
            axes[0, 0].plot(lb[o], g.x_obs.values[o], color="gray", lw=1.2, ls="--", zorder=0)
            axes[0, 1].plot(lb[o], S.loc[g.key.values[o]].amp.values, color="gray", lw=1.2, ls="--", zorder=0)
    style_ax(axes[0, 0], r"$\log_{10}$ non-embedding parameters $B$", r"peak $\log_{10} r^{\ast}$", "(a)", "peak position")
    style_ax(axes[0, 1], r"$\log_{10}$ non-embedding parameters $B$", "edge depth", "(b)", "edge depth of the peak")
    h, l = axes[0, 0].get_legend_handles_labels(); add_legend(fig, h, l, ncol=min(4, len(l)))
    save_fig(fig, "s3_capacity")

def plot_upos(keys=None):
    ks = [k for k in (keys or [k for k, c in MODELS.items() if c["group"] == "core"]) if os.path.exists(f"{ANALYSIS_DIR}/s3_upos_{k}{RUN_TAG}.csv")]
    if not ks:
        print("[skip] upos: chưa có kết quả"); return
    fs = STACK_FS; fig, axes, nr = grid_axes(len(ks), 2, panel_w=7.0, panel_h=5.0, hspace=0.9)
    for ax, k, cap in zip(axes, ks, "abcd"):
        d = pd.read_csv(f"{ANALYSIS_DIR}/s3_upos_{k}{RUN_TAG}.csv").sort_values("effect").reset_index(drop=True)
        col = [FUNC_COLOR if u in FUNC_UPOS else CONTENT_COLOR if u.split(" ")[0] in CONTENT_UPOS else OTHER_COLOR for u in d.upos]
        ax.barh(range(len(d)), d.effect, xerr=[d.effect - d.lo, d.hi - d.effect], color=col, ecolor="black", capsize=3, alpha=0.85)
        ax.set_yticks(range(len(d))); ax.set_yticklabels(d.upos, fontsize=fs["tick"]); ax.axvline(0, color="black", lw=1.0)
        ax.set_xlabel(r"effect on $R_k$ vs. NOUN", fontsize=fs["label"]); ax.tick_params(axis="x", labelsize=fs["tick"]); ax.grid(alpha=0.3, axis="x")
        panel_caption(ax, f"({cap}) {MODELS[k]['label']}", fs, cap_y=-0.28)
    h = [Line2D([], [], color=FUNC_COLOR, lw=8), Line2D([], [], color=CONTENT_COLOR, lw=8), Line2D([], [], color=OTHER_COLOR, lw=8)]
    fig.legend(h, ["function words", "content words", "other"], loc="lower center", bbox_to_anchor=(0.5, 0.90), ncol=3, fontsize=LEGEND_FS, frameon=False)
    save_fig(fig, "s3_upos_effects")

def plot_mantel(key, kind="syntactic", perm="block"):
    p = f"{ANALYSIS_DIR}/pair_{key}{RUN_TAG}.npz"
    if not os.path.exists(p):
        print(f"[skip] mantel {key}: chưa có kết quả"); return
    z = np.load(p); A, M, kw, K1, keep = z["A"], z["M"], int(z["kw"]), int(z["K1"]), z["keep"]
    T = z[f"T_{kind}"]; mt = pd.read_csv(f"{ANALYSIS_DIR}/mantel_{key}{RUN_TAG}.csv"); lr = pd.read_csv(f"{ANALYSIS_DIR}/mantel_layers_{key}{RUN_TAG}.csv")
    order = sorted(range(len(LANGS)), key=lambda i: (FAMILY[LANGS[i]], i)); labs = [SHORT[LANGS[i]] for i in order]
    fig, axes = make_fig(1, 3, pw=8.2, ph=5.4)
    ax = axes[0, 0]; im = ax.imshow(A[np.ix_(order, order)], cmap="viridis")
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labs, rotation=90, fontsize=TICK_FS - 3); ax.set_yticks(range(len(order))); ax.set_yticklabels(labs, fontsize=TICK_FS - 3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=TICK_FS - 2)
    ax.text(0.5, -0.30, f"(a) {MODELS[key]['label']}" if SUBFIG_WITH_DESC else "(a)", transform=ax.transAxes, ha="center", va="top", fontsize=SUBFIG_FS)
    ax = axes[0, 1]; L = len(LANGS); iu = np.triu_indices(L, 1); ok = np.isfinite(A[iu])
    same = np.array([FAMILY[LANGS[i]] == FAMILY[LANGS[j]] for i, j in zip(*iu)])
    ax.scatter(T[iu][ok & ~same], A[iu][ok & ~same], s=55, color=CONTENT_COLOR, alpha=0.8, edgecolor="black", lw=0.5, label="different family")
    ax.scatter(T[iu][ok & same], A[iu][ok & same], s=55, color=FUNC_COLOR, alpha=0.9, edgecolor="black", lw=0.5, label="same family")
    if ok.sum() > 2:
        c = np.polyfit(T[iu][ok], A[iu][ok], 1); xx = np.linspace(T[iu][ok].min(), T[iu][ok].max(), 50); ax.plot(xx, np.polyval(c, xx), color="black", lw=2.0)
    r = mt[(mt.typology == kind) & (mt.perm == perm)].iloc[0]
    ax.text(0.97, 0.95, f"Mantel $r_s={r.r:+.2f}$\n$p={r.p_neg:.3f}$ ({perm})", transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND_FS)
    style_ax(ax, f"{kind} distance", r"pair alignment $R_{\mathrm{pool}}$", "(b)", "alignment vs. typology")
    ax = axes[0, 2]
    for i, kd in enumerate([c for c in lr.columns if c in TYPO_SETS]):
        ax.plot(lr.rho, lr[kd], lw=2.4, color=PALETTE[i], label=kd)
    ax.axhline(0, color="black", ls=":", lw=1.0); ax.axvline(kw / (K1 - 1), color="gray", ls="--", lw=1.2)
    style_ax(ax, X_RHO, r"Mantel $r_s$", "(c)", "by depth")
    h, l = axes[0, 1].get_legend_handles_labels(); h2, l2 = axes[0, 2].get_legend_handles_labels()
    add_legend(fig, h + h2, l + l2, ncol=6)
    axes[0, 0].text(0.5, 1.02, "", transform=axes[0, 0].transAxes)
    save_fig(fig, f"mantel_{key}_{kind}")

def plot_lang_fertility(keys=None):
    ks = [k for k in (keys or [k for k, c in MODELS.items() if c["group"] == "core"]) if os.path.exists(f"{ANALYSIS_DIR}/lang_level_{k}{RUN_TAG}.csv")]
    if not ks:
        print("[skip] lang fertility: chưa có kết quả"); return
    fig, axes = make_fig(1, len(ks))
    for ax, k, cap in zip(axes[0], ks, "abcd"):
        d = pd.read_csv(f"{ANALYSIS_DIR}/lang_level_{k}{RUN_TAG}.csv")
        ax.scatter(d.fertility, d.R_pool_mean, s=90, color=MODEL_COLOR[k], edgecolor="black")
        for _, r in d.iterrows():
            ax.annotate(r.lang, (r.fertility, r.R_pool_mean), textcoords="offset points", xytext=(6, 5), fontsize=TICK_FS - 2)
        ok = d.dropna()
        if len(ok) >= 5:
            rho, p = spearmanr(ok.fertility, ok.R_pool_mean)
            ax.text(0.97, 0.95, f"$\\rho_s={rho:+.2f}$, $p={p:.2f}$", transform=ax.transAxes, ha="right", va="top", fontsize=LEGEND_FS)
        style_ax(ax, "tokens per word (fertility)", r"mean alignment $R_{\mathrm{pool}}$", f"({cap})", MODELS[k]["label"])
    save_fig(fig, "lang_alignment_vs_fertility")
