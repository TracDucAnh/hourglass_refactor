# Torch + mô hình 4 họ TinyLM (Cell B1)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 20 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell B1 — Torch + mô hình 4 họ
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
print("device:", DEVICE)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w

def rope_cache(T, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    fr = torch.outer(torch.arange(T).float(), inv)
    return fr.cos(), fr.sin()                                   # [T, hd/2]

def apply_rope(x, cos, sin):                                    # x: [B,T,H,hd]
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    c = cos[None, :x.shape[1], None, :].to(x.dtype); s = sin[None, :x.shape[1], None, :].to(x.dtype)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

class Attn(nn.Module):
    def __init__(self, d, nh, causal):
        super().__init__()
        self.nh, self.hd, self.causal = nh, d // nh, causal
        self.qkv = nn.Linear(d, 3 * d, bias=False); self.o = nn.Linear(d, d, bias=False)
    def forward(self, x, cos, sin, mask):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).unbind(2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None if self.causal else mask, is_causal=self.causal)
        return self.o(y.transpose(1, 2).reshape(B, T, -1))

class FFN(nn.Module):                                           # SwiGLU
    def __init__(self, d, ff):
        super().__init__()
        self.w1 = nn.Linear(d, ff, bias=False); self.w3 = nn.Linear(d, ff, bias=False); self.w2 = nn.Linear(ff, d, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEFFN(nn.Module):
    """8 expert, top-2, softmax lại trên 2 expert được chọn; expert có d_ff/2 để tham số kích hoạt khớp FFN dày."""
    def __init__(self, d, ff_e, n_exp=8, top_k=2):
        super().__init__()
        self.n_exp, self.top_k = n_exp, top_k
        self.router = nn.Linear(d, n_exp, bias=False)
        self.experts = nn.ModuleList([FFN(d, ff_e) for _ in range(n_exp)])
        self.aux = None
    def forward(self, x):
        B, T, d = x.shape
        xf = x.reshape(-1, d)
        probs = F.softmax(self.router(xf).float(), -1)
        topv, topi = probs.topk(self.top_k, -1)
        topv = topv / topv.sum(-1, keepdim=True)
        out = torch.zeros_like(xf)
        for e in range(self.n_exp):
            rows, slot = (topi == e).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            y = self.experts[e](xf[rows])
            out.index_add_(0, rows, (y * topv[rows, slot].unsqueeze(-1)).to(out.dtype))
        load = F.one_hot(topi, self.n_exp).float().sum(1).mean(0) / self.top_k
        self.aux = self.n_exp * (load * probs.mean(0)).sum()    # loss cân bằng tải (Switch)
        return out.reshape(B, T, d)

class Block(nn.Module):
    def __init__(self, d, nh, ff, moe, causal):
        super().__init__()
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.attn = Attn(d, nh, causal)
        self.ffn = MoEFFN(d, ff // 2) if moe else FFN(d, ff)
    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ffn(self.n2(x))

class TinyLM(nn.Module):
    """family: decoder | moe (causal) ; encoder | diffusion (hai chiều). forward -> (logits, hs, hn):
       hs = [residual sau embedding, sau khối 1, ..., sau khối K] (K+1 tensor, cái cuối là TRƯỚC final norm); hn = sau final norm."""
    def __init__(self, family, d, vocab, n_cls=0, K=K_LAYERS):
        super().__init__()
        self.family, self.d = family, d
        self.causal = family in ("decoder", "moe")
        nh = max(1, d // D_HEAD)
        assert d % nh == 0 and (d // nh) % 2 == 0
        ff = ((int(8 * d / 3) + 7) // 8) * 8
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([Block(d, nh, ff, family == "moe", self.causal) for _ in range(K)])
        self.norm = RMSNorm(d)
        self.cls_head = nn.Linear(d, n_cls) if n_cls else None
        cos, sin = rope_cache(2 * MAX_LEN, d // nh)
        self.register_buffer("cos", cos, persistent=False); self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        scale = 1.0 / math.sqrt(2 * K)
        for b in self.blocks:
            b.attn.o.weight.data.mul_(scale)
            for m in ([b.ffn] if family != "moe" else list(b.ffn.experts)):
                m.w2.weight.data.mul_(scale)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    def forward(self, ids):
        x = self.emb(ids); hs = [x]
        mask = None if self.causal else (ids != PAD)[:, None, None, :]
        for b in self.blocks:
            x = b(x, self.cos, self.sin, mask); hs.append(x)
        hn = self.norm(x)
        return F.linear(hn, self.emb.weight), hs, hn
    def aux_loss(self):
        return sum(b.ffn.aux for b in self.blocks)
    def n_params_nonemb(self):
        return sum(p.numel() for n, p in self.named_parameters() if not n.startswith("emb") and not n.startswith("cls_head"))

def build_model(family, d, task, W):
    meta = W["meta"]
    n_cls = {"SEM": meta["n_cls_sem"], "SURF": meta["n_cls_surf"]}.get(task, 0) if family == "encoder" else 0
    return TinyLM(family, d, meta["vocab_total"], n_cls).to(DEVICE)
