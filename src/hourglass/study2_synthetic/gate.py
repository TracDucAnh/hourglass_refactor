# Báo cáo cổng G2 + dự toán thời gian (Cell C5)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 31 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C5 — Báo cáo cổng G2 + dự toán thời gian toàn lưới
# ============================================================
def gate_g2_report(grid="gate_g2"):
    rows = []
    for c in grid_configs(grid):
        rdir = f"{RUN_DIR}/{run_id(c)}"
        m = read_json(f"{rdir}/meta.json", {})
        tr, ini = load_npz(rdir, "trained"), load_npz(rdir, "init")
        if m.get("status") != "ok" or tr is None:
            continue
        if m.get("metric_version") != METRIC_VERSION or m.get("train_version") != TRAIN_VERSION:
            print(f"[bỏ qua] {run_id(c)}: kết quả đo/huấn luyện bằng phiên bản cũ -> chạy lại cell C4"); continue
        L = c["L"]
        Gm = np.nanmean(tr["G"], 0); Gi = np.nanmean(ini["G"], 0)
        k = int(np.nanargmax(Gm))
        rows.append(dict(seed=c["seed"], U0_over_logL=float(tr["U_lin"][0] / math.log(L)), U0_mlp_over_logL=float(tr["U_mlp"][0] / math.log(L)), acc0=float(tr["U_acc"][0]), k_star=k, K=len(Gm) - 1,
                         G_peak=float(Gm[k]), edge_depth=float(Gm[k] - max(Gm[0], Gm[-1])),
                         edge_depth_init=float(Gi.max() - max(Gi[0], Gi[-1])),
                         interior=bool(0 < k < len(Gm) - 1), loss=m["final_lm_loss"], eval_loss=m["eval_lm_loss"], val=m.get("val_loss"), floor=m.get("floor_nats"),
                         conv_drop=m.get("conv_drop"), converged=bool(m.get("converged", False)), k_ok=bool(k >= GATE_MIN_K), minutes=m["seconds"] / 60))
    import pandas as pd
    df = pd.DataFrame(rows)
    if df.empty:
        print("Chưa có kết quả gate."); return df
    print(df.round(3).to_string(index=False))
    ok_u = bool((df.U0_over_logL > 0.9).all())
    ok_w = bool((df.interior & df.k_ok & (df.edge_depth > 0.05) & (df.edge_depth > df.edge_depth_init + 0.03)).all())
    ok_c = bool(df.converged.all())
    print(f"\nG2: U_0/log L ≈ 1 -> {'OK' if ok_u else 'KHÔNG ĐẠT'} | có eo (đỉnh G trong thân, k* >= {GATE_MIN_K}, sâu > 0.05, hơn init) -> "
          f"{'OK' if ok_w else 'KHÔNG ĐẠT'} | hội tụ (val còn giảm < {CONV_TOL} nats ở ~1/6 bước cuối) -> {'OK' if ok_c else 'KHÔNG ĐẠT'}")
    mean_min = float(df.minutes.mean())
    tot = sum(len(grid_configs(g)) for g in ALL_GRIDS if g not in ("gate_g2", "smoke", "pilot_conv"))
    print(f"Thời gian trung bình/run ≈ {mean_min:.1f} phút -> ước tính toàn lưới ({tot} run, chưa tính sinh đôi) ≈ {mean_min * tot / 60:.0f} giờ GPU")
    if not ok_c:
        print("-> Chưa hội tụ: chạy pilot_conv (Cell C5b) rồi tăng SEQ_MULT / giảm BATCH ở Cell A1; ĐỪNG kết luận về waist khi chưa hội tụ.")
    elif not (ok_u and ok_w):
        print("-> Đã hội tụ nhưng không có eo: sửa bộ sinh (độ khác biệt, kích thước pool) trước khi chạy lưới.")
    return df
