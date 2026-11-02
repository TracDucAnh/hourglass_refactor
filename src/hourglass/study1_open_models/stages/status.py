# Bảng trạng thái các run (Cell C2)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 8 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["status"]) hoặc `python -m hourglass run <experiment> status`.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell C2 — Trạng thái hiện tại (đọc từ /mnt/cl, không cần GPU)
# ============================================================
show_status()
