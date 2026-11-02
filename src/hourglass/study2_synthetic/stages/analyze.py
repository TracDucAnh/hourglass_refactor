# Đặc trưng -> kiểm định -> hình -> báo cáo (Cell D7)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 41 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["analyze"]) hoặc `python -m hourglass run <experiment> analyze`.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D7 — CHẠY phân tích: đặc trưng -> kiểm định -> hình -> báo cáo (tất cả lưu ở STUDY_DIR/analysis, STUDY_DIR/figures)
# ============================================================
df_feat, results = regenerate_all()
