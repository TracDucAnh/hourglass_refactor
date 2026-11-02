# Chạy Mantel
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 34 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["mantel"]) hoặc `python -m hourglass run <experiment> mantel`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

MANTEL = {k: analyze_mantel(k) for k in done_models()}
# Bản mở rộng (mọi cặp, kể cả cặp alt/dice): để biết kết luận loại hình có đổi khi thêm các ngôn ngữ không thuộc tầng chính hay không
MANTEL_EXT = {k: analyze_mantel(k, scope="all", tag="_ext") for k in done_models()} if (MANTEL_SCOPE == "main" and (~PAIR_MAIN).any()) else {}
