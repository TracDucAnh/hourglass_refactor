# Hình xếp chồng + MoE J_k + diffusion theo dataset (cuối Cell D3b)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 16 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["figures_stack"]) hoặc `python -m hourglass run <experiment> figures_stack`.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

plot_stack_all()
plot_moe_js_datasets()           # (a) MoE trên FLORES | (b) MoE trên NTREX
plot_diffusion_G_datasets()      # Ḡ_k theo mức che t: cột = FLORES | NTREX, mỗi panel LLaDA (liền) + Dream (đứt)
