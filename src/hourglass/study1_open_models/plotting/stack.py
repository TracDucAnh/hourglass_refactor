# Hình xếp chồng D/G/CKA + so sánh FLORES | NTREX (Cell D3b)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 16 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D3b — Hình xếp chồng cho TỪNG dataset (FLORES và NTREX vẽ riêng, không gom chung). Mỗi canvas 3 x 4:
#   cột  = 4 họ kiến trúc (encoder / dense decoder / MoE / diffusion), mỗi cột gồm mọi model của họ đó;
#   hàng 1 = D_k   (separation): các model pretrained + đường random-init của họ đó (đứt, xám);
#   hàng 2 = G_k   (alignment): các model pretrained + đường random-init (đứt, xám);
#   hàng 3 = linear CKA: các model pretrained + đường random-init (đứt, xám).
# - plot_stack(ds)  : vẽ + lưu canvas 3 x 4 của MỘT dataset -> {ds}_stack_D-pre+r_G-pre+r_CKA-pre+r.pdf
# - plot_stack_all(): gọi plot_stack cho flores rồi ntrex (dataset chưa có kết quả thì bỏ qua). Gọi ở cuối cell này.
# Đổi hàng: rows=[(metric, kind), ...]; kind = "pretrained" | "pretrained+random" (thêm đường random-init) | "random".
#   Ví dụ bỏ random-init ở CKA: rows=[("D","pretrained+random"),("G","pretrained+random"),("CKA","pretrained")]
# [SỬA] label model: 1 legend chung phía TRÊN ĐẦU CỘT (kiểu code mẫu), màu lấy từ MODEL_COLOR/PALETTE (cell A2);
#   đường random-init (đứt) giữ nguyên D3b gốc: vẽ chung panel + legend nhỏ trong panel kèm S.
# Phụ thuộc: D1 (get_curves), D2 (band/save_fig/PANELS/X_RHO), D3 (collect_family), A2 (MODEL_COLOR).
# - plot_moe_js_datasets()      : J_k (JS routing) của các model MoE, (a) FLORES | (b) NTREX cạnh nhau  -> moe_routing_Jk_flores_ntrex.pdf
# - plot_diffusion_G_datasets() : Ḡ_k của diffusion LM theo mức che t, 1 hàng x 2 cột = FLORES | NTREX;
#   mỗi panel vẽ chung LLaDA (nét liền) + Dream (nét đứt)  -> diffusion_G_masklevels_flores_ntrex.pdf
#   (định nghĩa ở cuối cell này, gọi ngay sau plot_stack_all())
# ============================================================
FAM_NAME  = {"encoder": "Encoder / byte", "decoder": "Dense decoder", "moe": "MoE", "diffusion": "Diffusion LM"}
STACK_FS  = dict(label=19, tick=14, cap=22, legend=12.5)
CTRL_BG   = "#f5f5f5"          # nền hàng kind="random" (nếu dùng) để phân biệt nhóm đối chứng
DEFAULT_ROWS = [("D", "pretrained+random"), ("G", "pretrained+random"), ("CKA", "pretrained+random")]

def _shape(m, metric):
    """(k, độ sâu): D có đáy (chữ U) -> k=argmin, depth = trung bình hai đầu - đáy;
    các số đo khác (G, CKA, A, R, U) có đỉnh -> k=argmax, depth = đỉnh - trung bình hai đầu."""
    m = np.asarray(m, float)
    if not np.isfinite(m).any():
        return 0, np.nan
    ends = 0.5 * (m[0] + m[-1])
    if metric == "D":
        k = int(np.nanargmin(m)); return k, float(ends - m[k])
    k = int(np.nanargmax(m)); return k, float(m[k] - ends)

def _row_spec(rows):
    if rows is None:
        return list(DEFAULT_ROWS)
    if isinstance(rows, str):                      # plot_stack(ds, "D") -> 1 hàng: model pretrained + random-init
        return [(rows, "pretrained+random")]
    return list(rows)

def _row_tag(spec):
    return "_".join(f"{mt}-{k.split('+')[0][:3]}{'+r' if '+' in k else ''}" for mt, k in spec)

def _has_results(ds, fams=None):
    return any(collect_family(ds, fam)[0] for fam in (fams or FAMILIES))

def plot_stack(ds, rows=None, fams=None, panel_w=5.6, panel_h=3.2, sharey="col", annotate=True,
               annotate_rho=False, ref_pretrained=True, cap_y=-0.30, name=None):
    """Canvas len(rows) x len(fams) cho MỘT dataset (mặc định 3 x 4: D_k / G_k / CKA, hàng nào cũng có đường random-init).
    sharey: "col" (mỗi hàng một số đo riêng nên thực chất mỗi panel tự co thang), "all" (cùng số đo dùng chung thang qua các họ), "row", hoặc False.
    [SỬA] label model: 1 legend chung phía trên đầu CỘT (kiểu code mẫu), tạo bằng matplotlib.legend.Legend + add_artist
    để không bị ghi đè; chừa top=0.86 để legend không bị cắt. Màu model = MODEL_COLOR (PALETTE, cell A2).
    annotate: ghi độ sâu S vào legend đường random-init. annotate_rho: thêm rho của đáy/đỉnh vào legend model."""
    from matplotlib.legend import Legend

    spec = _row_spec(rows); nr = len(spec)
    fams = list(fams or FAMILIES); nc = len(fams); fs = STACK_FS
    fig, axes = plt.subplots(nr, nc, figsize=(panel_w * nc, panel_h * nr), sharex=True, squeeze=False)
    # [SỬA] chừa khoảng trống phía trên cho legend model của mỗi cột (tránh bị cắt khỏi canvas)
    # Chừa vùng phía trên đủ rộng cho một legend tên model ở mỗi cột kiến trúc.
    fig.subplots_adjust(wspace=0.26, hspace=0.10, top=0.78)

    first = {}                                                      # chia sẻ trục y theo nhóm (cùng số đo)
    for i, (mt, _) in enumerate(spec):
        for j in range(nc):
            key = {"col": (mt, j), "all": (mt,), "row": (i,)}.get(sharey)
            if key is None:
                continue
            if key in first:
                axes[i, j].sharey(first[key])
                if sharey in ("all", "row") and j > 0:
                    axes[i, j].tick_params(labelleft=False)
            else:
                first[key] = axes[i, j]

    # [SỬA] hàng gắn nhãn model = hàng đầu tiên có model pretrained -> legend chung đặt trên đầu cột (giống code mẫu)
    lab_row = next((i for i, (_, k) in enumerate(spec) if k.startswith("pretrained")), None)

    for j, fam in enumerate(fams):
        items, rnd = collect_family(ds, fam)
        for i, (mt, kind) in enumerate(spec):
            ax = axes[i, j]
            rsym = r"$\rho_{\min}$" if mt == "D" else r"$\rho^{\ast}$"
            ssym = "$S_{%s}$" % mt
            if kind == "random":
                ax.set_facecolor(CTRL_BG)
            if not items:
                ax.text(0.5, 0.5, "chưa có kết quả", transform=ax.transAxes, ha="center", va="center",
                        fontsize=fs["tick"], color="gray")
            elif "random" in kind and rnd is None:
                ax.text(0.5, 0.06, "chưa chạy random-init", transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=fs["tick"] - 1, color="gray")
            if kind.startswith("pretrained"):
                for it in items:
                    col = MODEL_COLOR[it["key"]]                   # [SỬA] màu theo PALETTE/MODEL_COLOR của cell A2
                    m, lo, hi = it["c"][mt]
                    x = np.arange(len(m)) / (len(m) - 1)
                    k, _ = _shape(m, mt)
                    lab = None
                    if i == lab_row:
                        lab = f"{it['label']} ({rsym}={x[k]:.2f})" if annotate_rho else it["label"]
                    band(ax, x, m, lo, hi, col, label=lab)
                    ax.scatter([x[k]], [m[k]], s=95, marker="v" if mt == "D" else "o", color=col,
                               edgecolor="black", zorder=5)
            if "random" in kind and rnd is not None:               # đường random-init của họ, vẽ chung panel với model pretrained (GIỮ NGUYÊN D3b)
                m, lo, hi = rnd["c"][mt]
                x = np.arange(len(m)) / (len(m) - 1)
                lab = rnd["label"] + (f", {ssym}={_shape(m, mt)[1]:.3f}" if annotate else "")
                band(ax, x, m, lo, hi, rnd["color"], label=lab, ls="--", lw=2.6, alpha=0.20)
                ref = next((it for it in items if it["key"] == RANDOM_BASELINE[fam]), None) \
                    if (kind == "random" and ref_pretrained) else None
                if ref is not None:                                  # chỉ kind="random": vẽ mờ model pretrained cùng kiến trúc làm mốc
                    mp, lop, hip = ref["c"][mt]
                    xp = np.arange(len(mp)) / (len(mp) - 1)
                    labp = f"pretrained ({ref['label']})" + (f", {ssym}={_shape(mp, mt)[1]:.3f}" if annotate else "")
                    ax.plot(xp, mp, color=ref["color"], lw=1.9, alpha=0.55, label=labp)
                    ax.fill_between(xp, lop, hip, color=ref["color"], alpha=0.07, linewidth=0)
            ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=fs["tick"])
            if mt in ("G", "A"):
                ax.axhline(0, color="black", lw=1.0, ls=":")
            # Legend random-init vẫn nằm trong từng panel; legend model được tạo một lần
            # cho cả cột ở phía trên sau khi vẽ xong toàn bộ các hàng.
            h, l = ax.get_legend_handles_labels()
            if items and l:
                rl = rnd["label"] if rnd is not None else None
                hr = [hh for hh, lb in zip(h, l) if rl and lb.startswith(rl)]
                lr = [lb for lb in l if rl and lb.startswith(rl)]
                if hr:
                    leg_random = ax.legend(hr, lr, loc="best", fontsize=fs["legend"] - 1.5,
                                           frameon=True, framealpha=0.85, handlelength=1.8,
                                           labelspacing=0.3)
                    ax.add_artist(leg_random)

        # Một legend duy nhất cho tên model ở phía trên mỗi cột.
        # Dùng Line2D trực tiếp thay vì phụ thuộc vào legend handles của band().
        from matplotlib.lines import Line2D
        if items:
            model_handles = [
                Line2D([0], [0], color=MODEL_COLOR[it["key"]], lw=2.4, label=it["label"])
                for it in items
            ]
            axes[0, j].legend(
                handles=model_handles, loc="lower center",
                bbox_to_anchor=(0.5, 1.03),
                ncol=min(2, len(model_handles)), fontsize=fs["legend"],
                frameon=False, handlelength=1.8, columnspacing=1.0,
                labelspacing=0.3, borderaxespad=0.0
            )
        axes[-1, j].set_xlabel(X_RHO, fontsize=fs["label"])
        axes[-1, j].text(0.5, cap_y, f"({'abcdefgh'[j]}) {FAM_NAME.get(fam, fam)}", transform=axes[-1, j].transAxes,
                         ha="center", va="top", fontsize=fs["cap"])
    for i, (mt, kind) in enumerate(spec):
        axes[i, 0].set_ylabel(PANELS[mt][0], fontsize=fs["label"])
    save_fig(fig, name or f"{ds_tag(ds)}_stack_{_row_tag(spec)}")

def plot_stack_all(dss=("flores", "ntrex"), rows=None, **kw):
    """Vẽ + lưu canvas riêng cho từng dataset. Dataset chưa có kết quả sẽ bị bỏ qua."""
    for ds in dss:
        if _has_results(ds, kw.get("fams")):
            plot_stack(ds, rows, **kw)
        else:
            print(f"[skip] {ds_tag(ds)}: chưa có kết quả")

# ============================================================
# HAI HÌNH SO SÁNH HAI DATASET (FLORES-200 | NTREX-128), xếp cạnh nhau, panel nào cũng có sub-caption (a), (b), ...
#   plot_moe_js_datasets()        : J_k (JS routing divergence vs English) của các model MoE   -> (a) FLORES, (b) NTREX
#   plot_diffusion_G_datasets()   : alignment Ḡ_k của diffusion LM ở các mức che t           -> cột = dataset; LLaDA (liền) + Dream (đứt) chung panel
# Phụ thuộc thêm: C1 (load_variant), A2 (MODELS, RANDOM_BASELINE, variant_name), A1 (MASK_LEVELS), D1 (get_curves).
# Dataset/model chưa có kết quả (vd. NTREX chỉ chạy OLMoE và LLaDA) thì panel đó ghi "chưa có kết quả", không báo lỗi.
# ============================================================
DS_NAME     = {"flores": "FLORES-200", "ntrex": "NTREX-128"}
MASK_COLORS = {0.0: "#444444", 0.25: "#1f77b4", 0.5: "#ff7f0e", 0.75: "#d62728"}   # cùng màu với D5

def _ds_name(ds):
    return DS_NAME.get(ds, ds)

PAIR_TOP = 0.93      # mép trên vùng vẽ (tỉ lệ chiều cao hình); legend nằm ngay phía trên

def _pair_axes(nr, nc, panel_w, panel_h, sharey):
    # sharex=False: mọi panel đều giữ nhãn trục x (mỗi panel có sub-caption riêng); xlim đặt tay (0, 1)
    fig, axes = plt.subplots(nr, nc, figsize=(panel_w * nc, panel_h * nr), sharex=False, sharey=sharey, squeeze=False)
    fig.subplots_adjust(wspace=0.16, hspace=0.62 if nr > 1 else 0.1, top=PAIR_TOP)
    return fig, axes

def _pair_caption(ax, text, fs, cap_y=-0.27):
    ax.text(0.5, cap_y, text, transform=ax.transAxes, ha="center", va="top", fontsize=fs["cap"])

def _shared_legend(fig, axes, fs, ncol_max=4, y=PAIR_TOP + 0.01):
    hs, ls_ = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in ls_:
                hs.append(h); ls_.append(l)
    if hs:
        fig.legend(hs, ls_, loc="lower center", bbox_to_anchor=(0.5, y), ncol=min(ncol_max, len(ls_)),
                   fontsize=fs["legend"] + 3, frameon=False, handlelength=2.2, columnspacing=1.6)

def _no_result(ax, fs, msg="chưa có kết quả"):
    ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=fs["tick"], color="gray")

def _moe_js_runs(ds):
    """([(key, cfg, d)] các model MoE có J_k, d_random hoặc None, key_random)."""
    items = []
    for key, cfg in MODELS.items():
        if cfg["family"] == "moe":
            d = load_variant(ds, key, "base")
            if d is not None and "JS" in d:
                items.append((key, cfg, d))
    rk = RANDOM_BASELINE["moe"]
    dr = load_variant(ds, rk + "__random", "base")
    return items, (dr if (dr is not None and "JS" in dr) else None), rk

def _moe_js_xy(d):
    """x = độ sâu tương đối của tầng MoE; js [n_lang-1, Lm] = trung bình theo content (giống D4)."""
    js = d["JS"].mean(2)
    x = (d["JS_layers"] + 1) / max(int(d["n_layers"]), 1)
    return x, js

def plot_moe_js_datasets(dss=("flores", "ntrex"), show_random=True, panel_w=6.6, panel_h=4.0, name=None):
    """Hình 1 x len(dss): (a) mọi model MoE trên FLORES, (b) mọi model MoE trên NTREX. Đường = mean qua ngôn ngữ, dải = ± std qua ngôn ngữ;
    đường đứt xám = random-init (OLMoE). Hai panel dùng chung trục y để so sánh trực tiếp độ lớn J_k giữa hai dataset."""
    fs = STACK_FS; dss = list(dss)
    runs = {ds: _moe_js_runs(ds) for ds in dss}
    if not any(runs[ds][0] for ds in dss):
        print("[skip] moe J_k flores/ntrex: chưa có kết quả"); return
    fig, axes = _pair_axes(1, len(dss), panel_w, panel_h, sharey=True)
    for j, ds in enumerate(dss):
        ax = axes[0, j]; items, dr, rk = runs[ds]
        if not items:
            _no_result(ax, fs)
        for key, cfg, d in items:
            x, js = _moe_js_xy(d); m, s = js.mean(0), js.std(0)
            band(ax, x, m, np.clip(m - s, 0, None), m + s, MODEL_COLOR[key], label=cfg["label"])
        if show_random and dr is not None:
            x, js = _moe_js_xy(dr); m, s = js.mean(0), js.std(0)
            band(ax, x, m, np.clip(m - s, 0, None), m + s, RANDOM_COLOR, label=f"random init ({MODELS[rk]['label']})",
                 ls="--", lw=2.6, alpha=0.20)
        ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=fs["tick"])
        ax.set_xlabel(X_RHO, fontsize=fs["label"])
        if j == 0:
            ax.set_ylabel(r"$J_k$ (JS vs English)", fontsize=fs["label"])
        _pair_caption(ax, f"({'abcdefgh'[j]}) {_ds_name(ds)}", fs)
    _shared_legend(fig, axes, fs, ncol_max=4)
    save_fig(fig, name or "moe_routing_Jk_" + "_".join(ds_tag(d) for d in dss))

def plot_diffusion_G_datasets(dss=("flores", "ntrex"), models=None, pool="all", t_levels=None, mark_peak=True,
                              panel_w=6.6, panel_h=4.0, name=None):
    """Hình alignment Ḡ_k của diffusion LM ở các mức che t: 1 hàng, CỘT = dataset (FLORES | NTREX).
    Mọi model diffusion có kết quả được vẽ CHUNG một panel: model đầu tiên (LLaDA) nét liền, các model sau (Dream) nét đứt.
    Màu = mức che t (dải = CI 95% bootstrap theo content); chấm = đỉnh k* (tròn = nét liền, vuông = nét đứt). Trục y dùng chung.
    models=["llada_8b"] chỉ vẽ một model; pool="masked" vẽ riêng token bị che (bỏ t=0)."""
    from matplotlib.lines import Line2D
    fs = STACK_FS; dss = list(dss)
    t_levels = list(MASK_LEVELS if t_levels is None else t_levels)
    ts = [t for t in t_levels if not (pool == "masked" and t == 0)]
    keys = list(models) if models else [k for k, c in MODELS.items() if c["family"] == "diffusion"]
    has = lambda ds, k: any(get_curves(ds, k, variant_name(t, pool)) is not None for t in ts)
    keys = [k for k in keys if any(has(ds, k) for ds in dss)]
    if not keys:
        print("[skip] diffusion Ḡ_k flores/ntrex: chưa có kết quả"); return

    # kiểu nét theo thứ tự model (cố định cho mọi panel): model đầu = liền, các model sau = đứt
    LINE_KW = [dict(), dict(ls="--", lw=2.6, alpha=0.12)]
    MARKERS = ["o", "s"]
    sty = lambda n: LINE_KW[min(n, len(LINE_KW) - 1)]
    mk  = lambda n: MARKERS[min(n, len(MARKERS) - 1)]

    fig, axes = _pair_axes(1, len(dss), panel_w, panel_h, sharey=True)
    for j, ds in enumerate(dss):
        ax = axes[0, j]; any_data = False
        for n_k, k in enumerate(keys):
            for n_t, t in enumerate(ts):
                c = get_curves(ds, k, variant_name(t, pool))
                if c is None:
                    continue
                any_data = True
                m, lo, hi = c["G"]; x = np.arange(len(m)) / (len(m) - 1)
                col = MASK_COLORS.get(t, PALETTE[n_t % len(PALETTE)])
                band(ax, x, m, lo, hi, col, label=rf"$t={t:g}$", **sty(n_k))
                if mark_peak and np.isfinite(m).any():
                    kk = int(np.nanargmax(m))
                    ax.scatter([x[kk]], [m[kk]], s=95, marker=mk(n_k), color=col, edgecolor="black", zorder=5)
        if not any_data:
            _no_result(ax, fs)
        ax.axhline(0, color="black", lw=1.0, ls=":")
        ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.tick_params(axis="both", labelsize=fs["tick"])
        ax.set_xlabel(X_RHO, fontsize=fs["label"])
        if j == 0:
            ax.set_ylabel(PANELS["G"][0], fontsize=fs["label"])
        _pair_caption(ax, f"({'abcdefgh'[j]}) {_ds_name(ds)}" + (", masked tokens" if pool == "masked" else ""), fs)

    # legend kiểu nét -> model (đặt trong panel đầu tiên; màu t nằm ở legend chung phía trên)
    if len(keys) > 1:
        mh = [Line2D([0], [0], color="black", lw=2.6, ls=sty(n).get("ls", "-"), marker=mk(n), markersize=8,
                     markeredgecolor="black", label=MODELS[k]["label"]) for n, k in enumerate(keys)]
        axes[0, 0].legend(handles=mh, loc="best", fontsize=fs["legend"], frameon=True, framealpha=0.85,
                          handlelength=2.4, labelspacing=0.3)
    _shared_legend(fig, axes, fs, ncol_max=len(ts))
    save_fig(fig, name or "diffusion_G_masklevels_" + "_".join(ds_tag(d) for d in dss) + ("" if pool == "all" else f"_{pool}"))
