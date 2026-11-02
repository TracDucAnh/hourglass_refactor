# Hình theo nhóm kiến trúc (Cell D3)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 15 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D3 — Hình theo nhóm kiến trúc: encoder / decoder LLM / MoE / diffusion
# Mỗi model một màu cố định; dải mờ cùng màu = CI 95% bootstrap theo content (G, R) hoặc ± std (D, CKA, A, U)
# ============================================================
def collect_family(ds, fam):
    items = []
    for key, cfg in MODELS.items():
        if cfg["family"] != fam:
            continue
        c = get_curves(ds, key, main_variant(key))
        if c is not None:
            items.append(dict(key=key, label=cfg["label"], color=MODEL_COLOR[key], c=c))
    rk = RANDOM_BASELINE[fam]
    cr = get_curves(ds, rk + "__random", main_variant(rk, True))
    rnd = None if cr is None else dict(label=f"random init ({MODELS[rk]['label']})", color=RANDOM_COLOR, c=cr)
    return items, rnd

def draw_metric(ax, metric, items, rnd, mark_waist=False):
    for it in items:
        m, lo, hi = it["c"][metric]
        x = np.arange(len(m)) / (len(m) - 1)
        band(ax, x, m, lo, hi, it["color"], label=it["label"])
        if mark_waist and np.isfinite(m).any():
            k = int(np.nanargmax(m))
            ax.scatter([x[k]], [m[k]], s=110, color=it["color"], edgecolor="black", zorder=5)
    if rnd is not None:
        m, lo, hi = rnd["c"][metric]
        x = np.arange(len(m)) / (len(m) - 1)
        band(ax, x, m, lo, hi, rnd["color"], label=rnd["label"], ls="--", lw=2.2, alpha=0.15)
    if metric in ("G", "A"):
        ax.axhline(0, color="black", lw=1.0, ls=":")
    ax.set_xlim(0, 1)

def plot_family(ds, fam, kind="main"):
    items, rnd = collect_family(ds, fam)
    if not items:
        print(f"[skip] {ds_tag(ds)} {fam}: chưa có kết quả"); return
    metrics = ["G", "CKA"] if kind == "main" else ["D", "A", "R", "U"]
    nrows, ncols = (1, 2) if kind == "main" else (2, 2)
    fig, axes = make_fig(nrows, ncols)
    for ax, mt, cap in zip(axes.ravel(), metrics, "abcd"):
        draw_metric(ax, mt, items, rnd, mark_waist=(mt == "G"))
        style_ax(ax, X_RHO, PANELS[mt][0], f"({cap})", PANELS[mt][1])
    h, l = axes[0, 0].get_legend_handles_labels()
    add_legend(fig, h, l, ncol=min(3, len(l)))
    save_fig(fig, f"{ds_tag(ds)}_{FAMILY_FILE[fam]}_{kind}")
