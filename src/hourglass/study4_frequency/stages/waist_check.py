# Kiểm tra nhanh tầng waist theo truy hồi
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 26 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["waist_check"]) hoặc `python -m hourglass run <experiment> waist_check`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# Kiểm tra nhanh: các model đã có kết quả + tầng waist theo truy hồi
for _k in done_models():
    _run, _meta = load_run(_k)
    _kw, _K1 = waist_layer(_k, _run)
    print(f"{_k:12s} K+1={_K1:3d}  tầng waist (R_pool max) = {_kw} (rho={_kw / (_K1 - 1):.2f})  n_pool={int(_run['n_pool'])}  "
          f"R_pool@waist={_run['POOL_R'][:, _kw].mean():.3f}  tham số không-embedding={_meta.get('n_params_nonemb', 0) / 1e6:.0f}M")
