# Vẽ toàn bộ hình D2
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 41 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["figures"]) hoặc `python -m hourglass run <experiment> figures`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

plot_s3_curves()
plot_peak_vs_depth()
plot_heatmaps()
plot_capacity()
plot_upos()
for _k in done_models():
    plot_mantel(_k)
plot_lang_fertility()
