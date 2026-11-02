# Bảng capacity + kết luận S3
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 36 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["capacity"]) hoặc `python -m hourglass run <experiment> capacity`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

if "S3_ALL" in globals() and len(S3_ALL):
    CAP = capacity_table(S3_ALL); SLOPES = capacity_slopes(CAP); VERDICT = verdict_table(S3_ALL)
    CAP.to_csv(f"{ANALYSIS_DIR}/capacity_table{RUN_TAG}.csv", index=False); SLOPES.to_csv(f"{ANALYSIS_DIR}/capacity_slopes{RUN_TAG}.csv", index=False)
    VERDICT.to_csv(f"{ANALYSIS_DIR}/s3_verdict{RUN_TAG}.csv", index=False)
    print(CAP.round(3).to_string(index=False)); print(); print(SLOPES.round(3).to_string(index=False)); print()
    print(fmt_sci(VERDICT, ["d_mono_minus_gam", "d_mono_ci_lo"]).round(3).to_string(index=False))
