# CHẠY trích biểu diễn + truy hồi (Cell C2)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 24 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["run_models"]) hoặc `python -m hourglass run <experiment> run_models`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C2 — CHẠY: trích biểu diễn từ + truy hồi cho các model trong RUN_MODELS (có thể chạy lại, model đã xong sẽ được bỏ qua)
# Gợi ý thứ tự: RUN_MODELS = ["xlmr_large"] trước (nhỏ, nhanh) để kiểm tra, rồi qwen3_8b, rồi các model sweep.
# ============================================================
report = run_all()
show_status()
