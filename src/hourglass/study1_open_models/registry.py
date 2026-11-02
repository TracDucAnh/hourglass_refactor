# Registry 15 model + random-init baseline (Cell A2)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 2 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell A2 — Registry 15 model + random-init baseline theo kiến trúc
# [XM] tên model trên HF cần xác minh trước khi chạy
# ============================================================
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#393b79", "#e6550d", "#31a354", "#756bb1", "#843c39", "#00868b", "#b8860b"]

def _m(key, hf_id, family, label, loader="auto", trc=False, final_norm=None,
       router=False, max_len=512, bs=16, mask_id=None):
    # loader: auto=AutoModel | causal=AutoModelForCausalLM | t5enc=T5EncoderModel
    # final_norm: đường dẫn tới norm cuối (để lấy residual TRƯỚC final norm ở tầng cuối); None nếu không có
    return dict(key=key, id=hf_id, family=family, label=label, loader=loader, trc=trc,
                final_norm=final_norm, router=router, max_len=max_len, bs=bs, mask_id=mask_id)

MODELS = {m["key"]: m for m in [
    # --- encoder (3) + byte-level (1): cùng vẽ trong nhóm "encoder" ---
    _m("mbert",      "bert-base-multilingual-cased", "encoder", "mBERT"),
    _m("xlmr_base",  "FacebookAI/xlm-roberta-base",  "encoder", "XLM-R base"),
    _m("xlmr_large", "FacebookAI/xlm-roberta-large", "encoder", "XLM-R large"),
    _m("byt5_base_enc", "google/byt5-base",          "encoder", "ByT5-base enc (byte)",
       loader="t5enc", final_norm="encoder.final_layer_norm", max_len=1024, bs=8),
    # --- dense decoder (6) ---
    _m("qwen3_0.6b", "Qwen/Qwen3-0.6B-Base", "decoder", "Qwen3-0.6B", final_norm="norm", bs=32),
    _m("qwen3_1.7b", "Qwen/Qwen3-1.7B-Base", "decoder", "Qwen3-1.7B", final_norm="norm", bs=32),
    _m("qwen3_4b",   "Qwen/Qwen3-4B-Base",   "decoder", "Qwen3-4B",   final_norm="norm"),
    _m("qwen3_8b",   "Qwen/Qwen3-8B-Base",   "decoder", "Qwen3-8B",   final_norm="norm"),
    _m("bloom_560m", "bigscience/bloom-560m", "decoder", "BLOOM-560M", final_norm="ln_f", bs=32),
    _m("bloom_1b7",  "bigscience/bloom-1b7",  "decoder", "BLOOM-1.7B", final_norm="ln_f", bs=32),
    # --- MoE (3): router được hook trực tiếp tại gate (layers.N.mlp.gate) ---
    _m("olmoe",      "allenai/OLMoE-1B-7B-0924", "moe", "OLMoE-1B-7B",      final_norm="norm", router=True),
    _m("qwen15_moe", "Qwen/Qwen1.5-MoE-A2.7B",   "moe", "Qwen1.5-MoE-A2.7B", final_norm="norm", router=True),
    _m("dsv2_lite",  "deepseek-ai/DeepSeek-V2-Lite", "moe", "DeepSeek-V2-Lite", final_norm="norm", router=True),
    # --- diffusion (2): LLaDA là AutoModelForCausalLM (remote code), Dream là AutoModel (remote code) ---
    _m("llada_8b", "GSAI-ML/LLaDA-8B-Base", "diffusion", "LLaDA-8B", loader="causal", trc=True,
       final_norm="model.transformer.ln_f", mask_id=126336),
    _m("dream_7b", "Dream-org/Dream-v0-Base-7B", "diffusion", "Dream-7B", trc=True,
       final_norm="model.norm", mask_id=None),     # mask_id=None -> lấy tok.mask_token_id
]}

FAMILIES    = ["encoder", "decoder", "moe", "diffusion"]
FAMILY_FILE = {"encoder": "encoder", "decoder": "decoder_llm", "moe": "moe", "diffusion": "diffusion"}
MODEL_COLOR = {k: PALETTE[i] for i, k in enumerate(MODELS)}

# random-init control: mỗi kiểu kiến trúc một model (trọng số ~ N(0, 0.02^2), tokenizer giữ nguyên)
RANDOM_BASELINE = {"encoder": "xlmr_base", "decoder": "qwen3_1.7b", "moe": "olmoe", "diffusion": "llada_8b"}

def variant_name(t, pool):
    return "base" if t is None else f"t{t:.2f}_{pool}"

def main_variant(key, random_init=False):
    return "base" if MODELS[key]["family"] != "diffusion" else variant_name(0.0, "all")

def plan_levels(cfg, random_init=False):
    """[(mức che t hoặc None, [tên variant tạo ra từ cùng một lượt forward])]"""
    if cfg["family"] != "diffusion":
        return [(None, ["base"])]
    levels = [0.0] if random_init else MASK_LEVELS
    return [(t, [variant_name(t, "all")] + ([variant_name(t, "masked")] if t > 0 else [])) for t in levels]

print(f"{len(MODELS)} models:", {f: sum(c['family'] == f for c in MODELS.values()) for f in FAMILIES})
