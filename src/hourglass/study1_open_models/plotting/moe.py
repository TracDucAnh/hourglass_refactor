# MoE: J_k routing (Cell D4)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 17 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D4 — MoE: J_k (JS routing divergence so với English) — hình kiểu Figure 4
# ============================================================
def plot_moe_routing(ds):
    items = []
    for key, cfg in MODELS.items():
        if cfg["family"] == "moe":
            d = load_variant(ds, key, "base")
            if d is not None and "JS" in d:
                items.append((key, cfg, d))
    if not items:
        print(f"[skip] {ds_tag(ds)} moe routing: chưa có kết quả"); return
    rk = RANDOM_BASELINE["moe"]
    dr = load_variant(ds, rk + "__random", "base")
    dr = dr if (dr is not None and "JS" in dr) else None

    def prep(d):
        js = d["JS"].mean(2)                                    # [n_lang-1, Lm]: trung bình theo content
        x = (d["JS_layers"] + 1) / max(int(d["n_layers"]), 1)   # độ sâu tương đối của tầng MoE
        return x, js
    n_pan = 1 + len(items)
    ncols = 2; nrows = int(math.ceil(n_pan / ncols))
    fig, axes = make_fig(nrows, ncols)
    axl = axes.ravel()
    for ax in axl[n_pan:]:
        ax.axis("off")
    ax = axl[0]                                                 # (a) mean ± std qua ngôn ngữ, mọi model MoE
    for key, cfg, d in items:
        x, js = prep(d); m, s = js.mean(0), js.std(0)
        band(ax, x, m, m - s, m + s, MODEL_COLOR[key], label=cfg["label"])
    if dr is not None:
        x, js = prep(dr); m, s = js.mean(0), js.std(0)
        band(ax, x, m, m - s, m + s, RANDOM_COLOR, label=f"random init ({MODELS[rk]['label']})", ls="--", lw=2.2, alpha=0.15)
    style_ax(ax, X_RHO, r"$J_k$ (JS vs English)", "(a)", "all MoE models")
    for ax, (key, cfg, d), cap in zip(axl[1:n_pan], items, "bcd"):    # (b..) từng model: mọi ngôn ngữ + mean ± std
        x, js = prep(d); c = MODEL_COLOR[key]
        for row in js:
            ax.plot(x, row, color=c, lw=0.9, alpha=0.30)
        m, s = js.mean(0), js.std(0)
        band(ax, x, m, m - s, m + s, c, lw=2.8, alpha=0.20)
        style_ax(ax, X_RHO, r"$J_k$ (JS vs English)", f"({cap})", cfg["label"])
    h, l = axl[0].get_legend_handles_labels()
    add_legend(fig, h, l, ncol=min(3, len(l)))
    save_fig(fig, f"{ds_tag(ds)}_moe_routing_Jk")
