# Tạo lại toàn bộ bảng + hình (Cell D7)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 20 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D7 — Tạo lại TOÀN BỘ bảng + hình từ /mnt/cl (không chạy lại model). Chạy cell này sau khi đã có kết quả,
# hoặc chạy lại bất cứ lúc nào (kể cả ở kernel mới: chỉ cần Cell A1, A2, C1 và D1–D6, D3b).
# ============================================================
def regenerate_all(ds):
    df = feature_table(ds, save=True)
    if df.empty:
        print(f"Chưa có kết quả cho {ds_tag(ds)}."); return
    print_p1_summary(df)
    diffusion_s6_table(ds, save=True)
    export_curves_csv(ds)
    for fam in FAMILIES:
        plot_family(ds, fam, "main")
        plot_family(ds, fam, "extra")
    plot_stack(ds, "D")                 # <-- mới: 2 x 4, D_k pretrained (trên) vs random-init (dưới)
    plot_moe_routing(ds)
    plot_diffusion_masklevels(ds)
    plot_summary(ds)
    print(f"Bảng ở {ANALYSIS_DIR}, hình PDF ở {FIG_DIR}")
