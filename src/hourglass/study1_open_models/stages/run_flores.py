# CHẠY FLORES-200: 15 model + random-init (Cell C3)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 9 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["run_flores"]) hoặc `python -m hourglass run <experiment> run_flores`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell C3 — CHẠY FLORES-200 devtest: 15 model + random-init control mỗi kiến trúc
# Có thể chạy lại nhiều lần: model nào đã có kết quả sẽ được bỏ qua; model lỗi sẽ được thử lại.
# (cổng G1: đặt RUN_MODELS = ["xlmr_base", "qwen3_1.7b", "olmoe"] ở Cell A1 trước khi chạy cả 15 model)
# ============================================================
report_flores = run_all(["flores"])
