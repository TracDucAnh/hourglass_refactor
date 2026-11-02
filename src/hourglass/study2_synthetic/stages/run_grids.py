# Chạy các lưới trong RUN_GRIDS (Cell C4)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 30 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["run_grids"]) hoặc `python -m hourglass run <experiment> run_grids`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C4 — CỔNG G2: một cấu hình decoder M4/SEM, 3 seed.  Phải có U_0/log L ≈ 1 ở embedding và có eo.
# Sau khi chạy: đo thời gian thực mỗi run và nhân lại ngân sách cho toàn bộ lưới (in ở cuối).
# (với SMOKE=True đặt RUN_GRIDS = ["smoke"])
# ============================================================
for _g in RUN_GRIDS:
    run_grid(_g)
