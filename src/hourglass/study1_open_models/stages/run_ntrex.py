# CHẠY NTREX-128 (Cell C4)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 10 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["run_ntrex"]) hoặc `python -m hourglass run <experiment> run_ntrex`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell C4 — CHẠY NTREX-128 (mặc định: các model trong NTREX_MODEL_KEYS; đặt None để chạy tất cả)
# ============================================================
report_ntrex = run_all(["ntrex"])
