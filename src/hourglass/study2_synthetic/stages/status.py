# Trạng thái hiện tại (Cell C3)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 28 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["status"]) hoặc `python -m hourglass run <experiment> status`.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C3 — Trạng thái hiện tại (đọc từ STUDY_DIR, không cần GPU)
# ============================================================
show_status()
