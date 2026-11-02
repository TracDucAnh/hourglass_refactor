# Torch, load model pretrained / random-init (Cell B1)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 4 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell B1 — Torch, HF login, load model (pretrained / random-init)
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig, T5EncoderModel
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

FP16_CLIP = 65000.0
SHUF_CHUNK = 100
_LOADERS = {"auto": AutoModel, "causal": AutoModelForCausalLM, "t5enc": T5EncoderModel}

# Model nhạy với padding/batching ở bf16 -> chạy fp32. Các model còn lại giữ bf16.
FP32_MODELS = {"mbert", "byt5_base_enc"}

def _dtype_for(cfg):
    return torch.float32 if cfg["key"] in FP32_MODELS else torch.bfloat16

from hourglass.common.hf import _with_trc_fallback

@torch.no_grad()
def random_init_(model, seed=SEED, std=0.02):
    """Khởi tạo ngẫu nhiên: Linear/Embedding ~ N(0, std^2), bias = 0, các norm = (1, 0)."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev).manual_seed(seed)
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            w = torch.randn(m.weight.shape, generator=g, device=dev, dtype=torch.float32).mul_(std)
            m.weight.copy_(w.to(m.weight.dtype))
            if getattr(m, "bias", None) is not None:
                m.bias.zero_()
        elif getattr(m, "weight", None) is not None and m.weight.ndim == 1 \
                and "norm" in (name + type(m).__name__).lower():
            m.weight.fill_(1.0)
            if getattr(m, "bias", None) is not None:
                m.bias.zero_()

def load_model(cfg, random_init=False):
    tok = _with_trc_fallback(lambda t: AutoTokenizer.from_pretrained(
        cfg["id"], trust_remote_code=t, cache_dir=HF_CACHE_DIR), cfg)
    if tok.pad_token is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
    cls = _LOADERS[cfg["loader"]]
    dt = _dtype_for(cfg)
    extra = {"attn_implementation": "eager"} if "llada" in cfg["id"].lower() else {}   # LLaDA không hỗ trợ SDPA
    if not random_init:
        model = _with_trc_fallback(lambda t: cls.from_pretrained(
            cfg["id"], torch_dtype=dt, trust_remote_code=t, cache_dir=HF_CACHE_DIR, **extra), cfg)
    else:
        conf = _with_trc_fallback(lambda t: AutoConfig.from_pretrained(
            cfg["id"], trust_remote_code=t, cache_dir=HF_CACHE_DIR), cfg)
        if cfg["loader"] == "t5enc":
            model = cls(conf).to(dt)
        else:
            model = cls.from_config(conf, torch_dtype=dt, trust_remote_code=cfg["trc"], **extra)
    # model không có bảng embedding theo từ vựng (get_input_embeddings() ném lỗi) -> bỏ qua resize
    try:
        emb = model.get_input_embeddings()
    except (NotImplementedError, AttributeError):
        emb = None
    if emb is not None and hasattr(emb, "num_embeddings") and len(tok) > emb.num_embeddings:
        model.resize_token_embeddings(len(tok))
    try:
        model.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass
    model.eval().to(DEVICE)
    if random_init:
        random_init_(model)
    print(f"  [dtype] {cfg['key']}: {next(model.parameters()).dtype}")
    return model, tok

from hourglass.common.hf import get_final_norm

_GATE_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.gate$")
def find_router_gates(model):
    gates = []
    for name, mod in model.named_modules():
        m = _GATE_RE.search(name)
        if m and getattr(mod, "weight", None) is not None and mod.weight.ndim == 2:
            gates.append((int(m.group(1)), mod))
    gates.sort(key=lambda x: x[0])
    return gates
