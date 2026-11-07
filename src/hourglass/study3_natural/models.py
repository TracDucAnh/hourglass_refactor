# Mô hình theo định dạng HF: Llama (decoder, diffusion hai chiều) / Mixtral (MoE) (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần torch. Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaForCausalLM, MixtralConfig, MixtralForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP = dict(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else None
print(f"[models] device: {DEVICE}")

def ff_size(d):
    return max(64, int(round(8 * d / 3 / 64)) * 64)

def model_config(family, d, vocab):
    heads = max(1, d // D_HEAD)
    assert d % heads == 0, f"d={d} không chia hết cho {heads} đầu"
    common = dict(vocab_size=vocab, hidden_size=d, num_hidden_layers=N_LAYERS, num_attention_heads=heads,
                  num_key_value_heads=heads, max_position_embeddings=SEQ_LEN, rope_theta=10000.0, tie_word_embeddings=True,
                  pad_token_id=PAD_ID, bos_token_id=BOS_ID, eos_token_id=EOS_ID)
    if family == "moe":   # expert nửa cỡ, top-2 -> số tham số kích hoạt khớp FFN dày của decoder
        return MixtralConfig(intermediate_size=ff_size(d) // 2, num_local_experts=MOE_EXPERTS, num_experts_per_tok=MOE_TOPK,
                             output_router_logits=True, router_aux_loss_coef=MOE_AUX, **common)
    cfg = LlamaConfig(intermediate_size=ff_size(d), **common)
    cfg.hourglass_family = family          # ghi vào config.json; với diffusion, chiều attention do attn_mask() quyết định
    return cfg

SDPA = dict(attn_implementation="sdpa")   # mask boolean 4D của diffusion chỉ đúng với sdpa (eager cộng mask vào logit)

def build_model(family, d, vocab, seed):
    torch.manual_seed(seed)
    cls = MixtralForCausalLM if family == "moe" else LlamaForCausalLM
    cfg = model_config(family, d, vocab); cfg._attn_implementation = "sdpa"
    return cls(cfg).to(DEVICE)

def load_model(path, family):
    cls = MixtralForCausalLM if family == "moe" else LlamaForCausalLM
    return cls.from_pretrained(path, **SDPA).to(DEVICE)

def n_params_nonemb(model):
    return sum(p.numel() for n, p in model.named_parameters() if "embed_tokens" not in n and "lm_head" not in n)

def attn_mask(family, ids):
    """Causal (decoder, MoE): mask 2D bỏ pad, HF tự thêm tính nhân quả. Diffusion: mask 4D (B,1,T,T) cho mọi vị trí
    nhìn mọi token không phải pad -> attention hai chiều trên cùng khung Llama. Mask boolean (True = được nhìn) không phụ
    thuộc kiểu dữ liệu, nên dùng được cả trong autocast bf16."""
    keep = ids != PAD_ID
    if family != "diffusion":
        return keep.long()
    return keep[:, None, None, :].expand(ids.shape[0], 1, ids.shape[1], ids.shape[1])

def hidden_states(model, family, ids):
    """K+1 tensor [B,T,d]: embedding, sau khối 1..K; tầng cuối lấy TRƯỚC final norm (HF trả về bản sau norm)."""
    cap = {}
    h = model.model.norm.register_forward_hook(lambda mod, inp, out: cap.update(pre=inp[0]))
    try:
        out = model(input_ids=ids, attention_mask=attn_mask(family, ids), output_hidden_states=True)
    finally:
        h.remove()
    hs = list(out.hidden_states)
    hs[-1] = cap["pre"]
    return hs
