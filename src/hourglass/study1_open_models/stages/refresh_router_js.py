# Tính lại J_k của các model MoE (cell cuối notebook)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 21 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["refresh_router_js"]) hoặc `python -m hourglass run <experiment> refresh_router_js`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

def refresh_router_js(ds, key, random_init=False):
    cfg = MODELS[key]; run_id = key + ("__random" if random_init else "")
    d = load_variant(ds, run_id, "base")
    texts = get_texts(ds, N_SENT)
    model, tok = load_model(cfg, random_init)
    try:
        ext = extract_level(model, tok, cfg, texts, t=None)
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()
    JS, JS_raw, js_langs, rmean = router_js(ext["router"], list(texts))
    d.update(JS=JS, JS_raw=JS_raw, JS_langs=js_langs, router_mean=rmean)
    save_variant(ds, run_id, "base", d)

for k in ["olmoe", "qwen15_moe", "dsv2_lite"]:
    refresh_router_js("flores", k)
refresh_router_js("flores", "olmoe", random_init=True)   # control
# NTREX: chỉ olmoe nằm trong NTREX_MODEL_KEYS
for k in ["olmoe"]:
    refresh_router_js("ntrex", k)
refresh_router_js("ntrex", "olmoe", random_init=True)   # control
