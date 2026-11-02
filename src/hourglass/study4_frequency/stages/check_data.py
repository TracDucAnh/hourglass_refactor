# Kiểm tra nhanh dữ liệu
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 5 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["check_data"]) hoặc `python -m hourglass run <experiment> check_data`.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# Kiểm tra nhanh dữ liệu (không cần GPU). Lần đầu sẽ tải và xử lý, lần sau đọc lại từ DATA_DIR.
for _ds in MEASURE_SETS + ALIGN_EXTRA:
    _ = get_texts(_ds, N_SENT)
