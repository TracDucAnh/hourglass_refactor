# Hình S3 v6 (Cell D3)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 42 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["figures_v6"]) hoặc `python -m hourglass run <experiment> figures_v6`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell D3 — (v6) Hình S3: đường cong chỉ vẽ ĐẬM trong vùng đủ dữ liệu (đến xR = phân vị MAIN_ANCHOR[1]); vùng sau xR xám và nét chấm (không diễn giải).
#   Chú thích mỗi panel: tăng = f(x̂) − f(xL), giảm = f(x̂) − f(xR) (độ lớn MRR; xem C4b). Mỗi panel một trục y -> đọc số, đừng đọc độ cong.
# ============================================================
from matplotlib.lines import Line2D

def plot_s3_v6(keys=None, ncols=4):
    ks = _models_with(keys=keys)
    if not ks:
        print("[skip] s3 v6: chưa có kết quả"); return
    fs = STACK_FS
    fig, axes, nr = grid_axes(len(ks), ncols, sharey=False)
    for ax, k, cap in zip(axes, ks, "abcdefghijkl"):
        z = _load_s3(k); c = load_curve(k, "all"); col = MODEL_COLOR[k]
        if c is None:
            continue
        g, fo, fb = c["grid"], c["f_obs"], c["f_boot"]
        xL, xR = float(c["xq"][MAIN_ANCHOR[0]]), float(c["xq"][MAIN_ANCHOR[1]])
        lo, hi = np.percentile(fb, [2.5, 97.5], axis=0); ins = g <= xR
        ax.errorbar(z["bin_x"], z["bin_y"], yerr=1.96 * z["bin_se"], fmt="o", ms=4.5, color="#9a9a9a", ecolor="#c8c8c8", elinewidth=1.2, zorder=1)
        band(ax, g[ins], fo[ins], lo[ins], hi[ins], col, lw=2.8, alpha=0.22)
        ax.plot(g[~ins], fo[~ins], color=col, lw=1.6, ls=":", alpha=0.6)
        ax.axvspan(xR, g[-1], color="#bbbbbb", alpha=0.15, lw=0)
        ax.axvline(xL, color="#777777", ls=":", lw=1.0)
        r = shape_contrasts(g, fo, fb, xL, xR)
        if r is not None:
            kk = int(np.argmin(np.abs(g - r["x_hat"])))
            ax.scatter([r["x_hat"]], [fo[kk]], s=110, color=col, edgecolor="black", zorder=5)
            ax.text(0.03, 0.04, f"rise {r['rise']:+.3f}\nfall {r['fall']:+.3f}", transform=ax.transAxes, fontsize=fs["tick"] - 1, color="#333333")
        ax.set_xlabel(X_RANK, fontsize=fs["label"]) if (ax in axes[-ncols:] or len(axes) <= ncols) else None
        if ax in axes[::ncols]:
            ax.set_ylabel(Y_R, fontsize=fs["label"])
        ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=fs["tick"])
        panel_caption(ax, f"({cap}) {MODELS[k]['label']}", fs, cap_y=-0.36)
    h = [Line2D([], [], color="#9a9a9a", marker="o", ls="", ms=6), Line2D([], [], color="#444444", lw=2.8), Line2D([], [], color="#444444", lw=1.6, ls=":"),
         Line2D([], [], color="#444444", marker="o", mec="black", ls="", ms=9)]
    fig.legend(h, ["raw mean per rank bin ($\\pm$95%)", "GAM (95% CI), supported range", "beyond $x_R$ (not interpreted)", "peak $\\hat{x}$ in window"],
               loc="lower center", bbox_to_anchor=(0.5, 0.87 if nr == 1 else 0.93), ncol=4, fontsize=LEGEND_FS - 2, frameon=False, columnspacing=1.4)
    save_fig(fig, "s3_R_vs_logrank_waist_v6")

plot_s3_v6()
