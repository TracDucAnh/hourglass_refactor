# In báo cáo cổng G2 (cuối Cell C5)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 31 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["gate_report"]) hoặc `python -m hourglass run <experiment> gate_report`.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

gate_g2_report("smoke" if SMOKE else "gate_g2")
