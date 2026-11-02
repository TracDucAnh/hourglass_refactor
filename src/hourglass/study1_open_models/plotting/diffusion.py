# Diffusion: mức che t (Cell D5)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 18 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D5 — Diffusion: ảnh hưởng của mức che t (S6), pool trên mọi token vs chỉ token bị che
# Màu = mức che (mỗi panel là một model)
# ============================================================
T_COLORS = {0.0: "#444444", 0.25: "#1f77b4", 0.5: "#ff7f0e", 0.75: "#d62728"}

def plot_diffusion_masklevels(ds):
    keys = [k for k, c in MODELS.items() if c["family"] == "diffusion"]
    panels = [(k, p) for k in keys for p in ("all", "masked")]
    ncols = 2; nrows = int(math.ceil(len(panels) / ncols))
    fig, axes = make_fig(nrows, ncols)
    axl = axes.ravel(); any_data = False
    for ax in axl[len(panels):]:
        ax.axis("off")
    for ax, (k, pool), cap in zip(axl, panels, "abcdefgh"):
        for t in MASK_LEVELS:
            if pool == "masked" and t == 0:
                continue
            c = get_curves(ds, k, variant_name(t, pool))
            if c is None:
                continue
            any_data = True
            m, lo, hi = c["G"]; x = np.arange(len(m)) / (len(m) - 1)
            band(ax, x, m, lo, hi, T_COLORS[t], label=rf"$t={t}$")
            kk = int(np.nanargmax(m))
            ax.scatter([x[kk]], [m[kk]], s=110, color=T_COLORS[t], edgecolor="black", zorder=5)
        ax.axhline(0, color="black", lw=1.0, ls=":"); ax.set_xlim(0, 1)
        style_ax(ax, X_RHO, PANELS["G"][0], f"({cap})",
                 f"{MODELS[k]['label']}, {'all tokens' if pool == 'all' else 'masked tokens'}")
    if not any_data:
        print(f"[skip] {ds_tag(ds)} diffusion: chưa có kết quả"); plt.close(fig); return
    hs, ls_ = [], []
    for ax in axl:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h); ls_.append(l)
    add_legend(fig, hs, ls_, ncol=len(ls_))
    save_fig(fig, f"{ds_tag(ds)}_diffusion_masklevels")
