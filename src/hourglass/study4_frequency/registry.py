# Registry model core / sweep / 2b local (Cell A2)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 3 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A2 — Registry model (Study 4)
#   core  : XLM-R large, Qwen3-8B (kế hoạch Study 4, bước 2)
#   sweep : thêm các cỡ khác cùng họ để kiểm "r* dịch ra ngoài khi capacity tăng" (Conjecture 2 / S3)
#   local : các model 2b tự huấn luyện (Study 2b), đăng ký bằng register_local_model()
# [XM] tên model trên HF cần xác minh trước khi chạy
# ============================================================
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf",
           "#bcbd22", "#393b79", "#e6550d", "#31a354", "#756bb1", "#843c39", "#00868b", "#b8860b"]

def _m(key, hf_id, family, label, group="core", loader="auto", trc=False, final_norm=None, prefix=None,
       max_len=512, tok_budget=32768, max_bs=256, local=False):
    # final_norm: đường dẫn tới norm cuối trong model (để lấy residual TRƯỚC final norm ở tầng cuối)
    # prefix    : token đặt trước câu (decoder): để token "attention sink" rơi vào prefix, không rơi vào từ đầu câu
    # tok_budget: số token (đã đệm) tối đa mỗi lượt forward -> kích thước batch tự co theo độ dài câu
    return dict(key=key, id=hf_id, family=family, label=label, group=group, loader=loader, trc=trc,
                final_norm=final_norm, prefix=prefix, max_len=max_len, tok_budget=tok_budget, max_bs=max_bs, local=local)

MODELS = {m["key"]: m for m in [
    _m("xlmr_large",  "FacebookAI/xlm-roberta-large", "encoder", "XLM-R large", "core",  tok_budget=65536),
    _m("qwen3_8b",    "Qwen/Qwen3-8B-Base",           "decoder", "Qwen3-8B",    "core",  final_norm="norm", prefix="<|endoftext|>", tok_budget=24576),
    _m("xlmr_base",   "FacebookAI/xlm-roberta-base",  "encoder", "XLM-R base",  "sweep", tok_budget=65536),
    _m("qwen3_0.6b",  "Qwen/Qwen3-0.6B-Base",         "decoder", "Qwen3-0.6B",  "sweep", final_norm="norm", prefix="<|endoftext|>", tok_budget=65536),
    _m("qwen3_1.7b",  "Qwen/Qwen3-1.7B-Base",         "decoder", "Qwen3-1.7B",  "sweep", final_norm="norm", prefix="<|endoftext|>", tok_budget=49152),
    _m("qwen3_4b",    "Qwen/Qwen3-4B-Base",           "decoder", "Qwen3-4B",    "sweep", final_norm="norm", prefix="<|endoftext|>", tok_budget=32768),
]}

def register_local_model(key, path, label, final_norm="norm", prefix=None, family="decoder", tok_budget=65536):
    """Model 2b tự huấn luyện (HF format: config.json + trọng số + tokenizer fast). AutoModel -> LlamaModel nên final_norm='norm'."""
    MODELS[key] = _m(key, path, family, label, "local", final_norm=final_norm, prefix=prefix, tok_budget=tok_budget, local=True)

# Tự đăng ký model 2b: {ROOT}/study3_natural/checkpoints/<tên>/ (bỏ qua các thư mục sinh đôi đơn ngữ: tên bắt đầu bằng "twin")
for _p in sorted(glob.glob(f"{ROOT}/study3_natural/checkpoints/*/")):
    _name = os.path.basename(_p.rstrip("/"))
    if os.path.exists(os.path.join(_p, "config.json")) and not _name.startswith("twin"):
        register_local_model(f"le_{_name}", _p.rstrip("/"), f"2b {_name}")

MODEL_COLOR = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(MODELS)}
FAMILIES = ["encoder", "decoder"]

def scope_models():
    return [k for k in (MODELS if RUN_MODELS is None else RUN_MODELS) if k in MODELS]

print(f"{len(MODELS)} model:", {k: c["group"] for k, c in MODELS.items()})
