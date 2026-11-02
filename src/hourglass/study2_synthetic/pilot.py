# Báo cáo pilot hội tụ (Cell C5b)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 32 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C5b — Báo cáo PILOT hội tụ: chọn BATCH / SEQ_MULT nhỏ nhất mà kết quả đã ổn định
# Chạy sau RUN_GRIDS = ["pilot_conv"]. Quy tắc chọn (mọi điều kiện phải thỏa, so với nhánh ngân sách lớn nhất):
#   (1) val cách val của nhánh lớn nhất < 0.10 nats; (2) val còn giảm < CONV_TOL trong ~1/6 bước cuối;
#   (3) k* trùng nhánh lớn nhất ở >= 2/3 seed (ghép theo seed); (4) val - trn < OVERFIT_TOL.
# ============================================================
def pilot_report(grid="pilot_conv", plot=True):
    import pandas as pd
    rows, hists = [], {}
    for c in grid_configs(grid):
        rdir = f"{RUN_DIR}/{run_id(c)}"
        m = read_json(f"{rdir}/meta.json", {}); tr, ini = load_npz(rdir, "trained"), load_npz(rdir, "init")
        if m.get("status") != "ok" or tr is None or m.get("train_version") != TRAIN_VERSION or m.get("metric_version") != METRIC_VERSION:
            continue
        Gm = np.nanmean(tr["G"], 0); Gi = np.nanmean(ini["G"], 0); k = int(np.nanargmax(Gm))
        rows.append(dict(tag=c["tag"], seed=c["seed"], batch=m["batch"], steps=m["steps"], seq_M=m["batch"] * m["steps"] / 1e6,
                         val=m["val_loss"], trn=m["trn_loss"], gap=m["val_loss"] - m["trn_loss"], floor=m.get("floor_nats"),
                         conv_drop=m["conv_drop"], k_star=k, G_peak=float(Gm[k]), edge=float(Gm[k] - max(Gm[0], Gm[-1])),
                         edge_init=float(Gi.max() - max(Gi[0], Gi[-1])), minutes=m["seconds"] / 60))
        hists[(c["tag"], c["seed"])] = (m["history"], m["batch"])
    df = pd.DataFrame(rows)
    if df.empty:
        print("Chưa có kết quả pilot (chạy RUN_GRIDS = ['pilot_conv'])."); return df
    print(df.round(3).to_string(index=False))
    order = list(dict.fromkeys(df.sort_values(["seq_M", "steps"]).tag))
    ref = order[-1]                                                     # nhánh nhiều chuỗi nhất
    agg = df.groupby("tag").agg(batch=("batch", "first"), steps=("steps", "first"), seq_M=("seq_M", "first"), val=("val", "mean"),
                                gap=("gap", "mean"), conv_drop=("conv_drop", "mean"), k_mean=("k_star", "mean"),
                                edge=("edge", "mean"), minutes=("minutes", "mean")).loc[order]
    agg["val_vs_ref"] = agg.val - agg.loc[ref, "val"]
    kref = df[df.tag == ref].set_index("seed").k_star
    agg["k_agree_ref"] = [float(np.mean([(df[(df.tag == t) & (df.seed == s)].k_star.values[0] == kref[s]) for s in kref.index
                                         if len(df[(df.tag == t) & (df.seed == s)])])) for t in agg.index]
    agg["ok"] = (agg.val_vs_ref < 0.10) & (agg.conv_drop < CONV_TOL) & (agg.k_agree_ref >= 2 / 3) & (agg.gap < OVERFIT_TOL)
    print("\nTổng hợp theo nhánh (ref =", ref, "):"); print(agg.round(3).to_string())
    good = agg[agg.ok].sort_values("minutes")
    if len(good):
        t = good.index[0]
        print(f"\n=> Ngân sách nhỏ nhất đạt cả 4 tiêu chí: {t}  (BATCH={int(good.loc[t, 'batch'])}, {int(good.loc[t, 'steps'])} bước, "
              f"~{good.loc[t, 'minutes']:.1f} phút/run). Đặt ở Cell A1: BATCH và SEQ_MULT tương ứng rồi đóng băng.")
    else:
        print("\n=> Chưa có nhánh nào đạt: nếu nhánh lớn nhất vẫn còn conv_drop >= CONV_TOL hoặc val còn giảm so với x2 -> tăng SEQ_MULT hơn nữa;"
              "\n   nếu gap > OVERFIT_TOL -> thêm mẫu mới (tăng N_POOL), không thêm epoch. k* đổi theo ngân sách => kết quả waist phụ thuộc giai đoạn huấn luyện.")
    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.2, 3.4))
            for tag, col in zip(order, plt.cm.viridis(np.linspace(0, 0.9, len(order)))):
                cur = [h for (t_, _s), (h, _b) in hists.items() if t_ == tag]
                if not cur: continue
                x = np.array([r["seqs"] for r in cur[0]]) / 1e6
                y = np.mean([[r["val"] for r in h] for h in cur], 0)
                ax.plot(x, y, "-o", ms=3, color=col, label=tag)
            fl = df.floor.dropna()
            if len(fl): ax.axhline(float(fl.mean()), color="k", ls=":", lw=1, label="entropy floor")
            ax.set_xlabel("sequences seen (M)"); ax.set_ylabel("held-out LM loss (nats/token)"); ax.legend(frameon=False, fontsize=7)
            fig.tight_layout(); fig.savefig(f"{FIG_DIR}/pilot_convergence.pdf", bbox_inches="tight"); plt.show(); plt.close(fig)
        except Exception as e:
            print("[plot] bỏ qua:", repr(e)[:120])
    return df
