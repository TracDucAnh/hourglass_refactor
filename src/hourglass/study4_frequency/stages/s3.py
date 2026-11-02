# Chạy S3 cho mọi model đã có kết quả -> S3_ALL
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 31 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["s3"]) hoặc `python -m hourglass run <experiment> s3`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

def fmt_sci(df, cols, p=2):
    """In cột nhỏ ở dạng khoa học có dấu (vd. +5.30e-05): .round(3) làm các chênh lệch cỡ 1e-5 hiện thành 0.0 và mất cả dấu lẫn độ lớn."""
    d = df.copy()
    for c in cols:
        if c in d:
            d[c] = d[c].map(lambda v: f"{v:+.{p}e}" if pd.notna(v) else "nan")
    return d

S3_ROWS, S3_UPOS = [], {}
for _k in done_models():
    _rows, _ue, _uc = analyze_model(_k)
    S3_ROWS.append(_rows); S3_UPOS[_k] = (_ue, _uc)
if S3_ROWS:
    S3_ALL = pd.concat(S3_ROWS, ignore_index=True)
    S3_ALL.to_csv(f"{ANALYSIS_DIR}/s3_summary{RUN_TAG}.csv", index=False)
    _cols = ["key", "kw", "rho_w", "r_star", "x_star_lo", "x_star_hi", "amp", "amp_lo", "amp_hi", "p_LM", "p_LM_q", "p_LM_amp", "shape", "shape_q",
             "p_left_up", "p_right_down", "d_mono_vs_gam", "d_mono_vs_gam_lo", "d_mono_vs_uni"]
    _t = S3_ALL[S3_ALL.subset == "all"][[c for c in _cols if c in S3_ALL]].copy()
    _t = fmt_sci(_t, ["d_mono_vs_gam", "d_mono_vs_gam_lo", "d_mono_vs_uni"])
    print(_t.round(4).to_string(index=False))
