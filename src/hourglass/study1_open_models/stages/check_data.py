# Tải + kiểm tra nhanh dữ liệu (cuối Cell A3)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 3 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["check_data"]) hoặc `python -m hourglass run <experiment> check_data`.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# Kiểm tra nhanh dữ liệu (không cần GPU). Lần đầu sẽ tải và xử lý, lần sau đọc từ /mnt/cl/data.
_ = get_texts("flores", N_SENT)
if "ntrex" in DATASETS_TO_RUN:
    _ = get_texts("ntrex", N_SENT)
