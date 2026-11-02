# Tóm tắt waist rho*, S, w (Cell D6)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 19 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D6 — Tóm tắt waist của mọi model: rho*, S, w với CI 95% bootstrap (ô rỗng = random-init)
# ============================================================
def plot_summary(ds):
    df = feature_table(ds, save=False)
    if df.empty:
        print(f"[skip] {ds_tag(ds)} summary: chưa có kết quả"); return
    order = []
    for fam in FAMILIES:
        sub = df[df.family == fam]
        order += [r for _, r in sub[~sub.random_init].iterrows()] + [r for _, r in sub[sub.random_init].iterrows()]
    n = len(order)
    fig, axes = plt.subplots(1, 3, figsize=(8.0 * 3, 0.55 * n + 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.12, top=0.92)
    specs = [("rho_star", "rho_lo", "rho_hi", r"$\rho^*$", "(a)", r"$\rho^*$"),
             ("S", "S_lo", "S_hi", r"waist depth $S$", "(b)", r"$S$"),
             ("w", "w_lo", "w_hi", r"waist width $w$", "(c)", r"$w$")]
    for ax, (v, lo, hi, xl, cap, desc) in zip(axes, specs):
        for yi, r in enumerate(order):
            key = r["run"].replace("__random", "")
            col = RANDOM_COLOR if r["random_init"] else MODEL_COLOR[key]
            xe = [[max(r[v] - r[lo], 0)], [max(r[hi] - r[v], 0)]]
            ax.errorbar(r[v], yi, xerr=xe, fmt="s" if r["random_init"] else "o", color=col, ms=10, capsize=4, lw=2.2,
                        mfc="white" if r["random_init"] else col, mew=2)
        prev = None
        for yi, r in enumerate(order):                                   # vạch ngăn giữa các nhóm kiến trúc
            if prev is not None and r["family"] != prev:
                ax.axhline(yi - 0.5, color="gray", lw=1.0, ls="--")
            prev = r["family"]
        ax.set_yticks(range(n)); ax.set_yticklabels([r["label"] for r in order], fontsize=TICK_FS)
        ax.set_ylim(n - 0.5, -0.5)
        style_ax(ax, xl, "", cap, desc)
    h = [Line2D([], [], marker="o", color="black", ls="", ms=10, label="pretrained"),
         Line2D([], [], marker="s", mfc="white", mec=RANDOM_COLOR, color=RANDOM_COLOR, ls="", ms=10, mew=2, label="random init")]
    add_legend(fig, h, [x.get_label() for x in h], ncol=2)
    save_fig(fig, f"{ds_tag(ds)}_waist_summary")
