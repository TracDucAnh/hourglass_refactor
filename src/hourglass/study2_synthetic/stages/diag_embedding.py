# Chẩn đoán U_0/log L ở embedding (Cell C3b)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 29 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["diag_embedding"]) hoặc `python -m hourglass run <experiment> diag_embedding`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C3b — Chẩn đoán nhanh (~1 phút, không train): U_0/log L ở tầng embedding, cách đo CŨ vs MỚI
# Mô hình CHƯA huấn luyện, thế giới của gate seed 0. Kỳ vọng: cũ ≈ 0, mới ≈ 1 (overlap 0% -> chữ cái lộ ngôn ngữ).
# ============================================================
_c = make_cfg("diag", seed=0)
_W = get_world(world_cfg(_c["L"], _c["level"], _c["overlap"], _c["len_scale"], _c["seed"], _c["len_scale_X"], _c["share_X"]))
torch.manual_seed(1000)
_m = build_model("decoder", D_MAIN, "SEM", _W)
_Z, _ = extract_eval(_m, _W["eval_ids"], _W["eval_wid"], desc="[diag] trích eval")
_Z0 = _Z[0].to(DEVICE, torch.float32)                      # [L, N, d] tầng 0 = embedding
_L = _Z0.shape[0]
_old = _standardize(_Z0);           _old = _old - _old.mean(0, keepdim=True)
_new = _standardize_keep_lang(_Z0); _new = _new - _new.mean(0, keepdim=True)
for _name, _Zt in (("CŨ  (trừ tâm từng ngôn ngữ)", _old), ("MỚI (giữ tâm từng ngôn ngữ)", _new)):
    _u, _acc = usable_info_lin(_Zt, SEED)
    print(f"{_name}: U_0/log L = {_u / math.log(_L):+.3f} | accuracy đoán ngôn ngữ = {_acc:.3f} (đoán mò = {1 / _L:.2f})")
del _m, _Z, _Z0, _old, _new
if DEVICE == "cuda":
    torch.cuda.empty_cache()
