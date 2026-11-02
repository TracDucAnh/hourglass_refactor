# Load model + trích biểu diễn mức từ (Cell B1)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 19 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell B1 — Load model + trích xuất biểu diễn MỨC TỪ theo batch
# Z_k(w): residual stream tầng k (tầng cuối lấy TRƯỚC final norm), trung bình các sub-token chồng lấn với từ w (offset ký tự).
# Tối ưu cho A100 80GB:
#   - chỉ forward các câu có từ cần đo, sắp theo độ dài, batch theo NGÂN SÁCH TOKEN (tok_budget) thay vì số câu cố định;
#   - gộp sub-token -> từ bằng một phép bmm [B,W,S] x [B,S,d] cho mọi tầng (không vòng lặp theo từ);
#   - chỉ lưu các từ thực sự nằm trong cặp từ (NEED), ghi liền khối vào buffer fp16 [K1, n_rows, d] rồi np.save.
# ============================================================
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import transformers

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DT = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
DTYPE  = _DT[EXTRACT_DTYPE] if EXTRACT_DTYPE else (torch.bfloat16 if DEVICE == "cuda" else torch.float32)
torch.manual_seed(SEED); np.random.seed(SEED)
FP16_CLIP = 65000.0

from hourglass.common.hf import _with_trc_fallback

def load_tokenizer(cfg):
    tok = _with_trc_fallback(lambda t: AutoTokenizer.from_pretrained(cfg["id"], trust_remote_code=t, use_fast=True,
                                                                       cache_dir=HF_CACHE_DIR), cfg)
    if not tok.is_fast:
        raise RuntimeError(f"{cfg['id']}: cần tokenizer fast (offset_mapping) để gộp sub-token -> từ.")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    return tok

def load_model(cfg, dtype=None):
    model = _with_trc_fallback(lambda t: AutoModel.from_pretrained(cfg["id"], torch_dtype=(dtype or DTYPE), trust_remote_code=t,
                                                                    cache_dir=HF_CACHE_DIR), cfg)
    model.eval().to(DEVICE)
    emb = model.get_input_embeddings()
    n_total = sum(p.numel() for p in model.parameters())
    n_nonemb = n_total - (emb.weight.numel() if emb is not None else 0)
    print(f"  [model] {cfg['key']}: dtype={next(model.parameters()).dtype}, {n_total / 1e6:.0f}M tham số ({n_nonemb / 1e6:.0f}M không-embedding)")
    return model, int(n_nonemb)

from hourglass.common.hf import get_final_norm

# ---------------- Tokenize + gộp sub-token -> từ ----------------
def tokenize_sentences(tok, cfg, sents):
    enc = tok(list(sents), add_special_tokens=True, return_offsets_mapping=True, truncation=True,
              max_length=cfg["max_len"], padding=False)
    ids, offs = enc["input_ids"], [np.asarray(o, np.int32).reshape(-1, 2) for o in enc["offset_mapping"]]
    if cfg.get("prefix"):
        pid = tok.convert_tokens_to_ids(cfg["prefix"])
        ids = [[pid] + list(x) for x in ids]
        offs = [np.concatenate([np.zeros((1, 2), np.int32), o]) for o in offs]
    return [list(x) for x in ids], offs

def word_token_overlap(offs, ws, we):
    """bool [W,S]: token t chồng lấn từ w (khoảng ký tự [ws,we)) và có độ rộng > 0 (loại token đặc biệt có offset (0,0))."""
    ts, te = offs[:, 0], offs[:, 1]
    return (ts[None, :] < we[:, None]) & (te[None, :] > ws[:, None]) & (te > ts)[None, :]

def plan_language(tok, cfg, sents, wt, need, sent_ids=None, budget=None, max_bs=None):
    """Kế hoạch trích xuất cho một ngôn ngữ: batch, hàng (row) của từng từ trong buffer, số sub-token của MỌI từ (fertility)."""
    budget, max_bs = budget or cfg["tok_budget"], max_bs or cfg["max_bs"]
    ids, offs = tokenize_sentences(tok, cfg, sents)
    N, ptr, nW = len(sents), wt["ptr"], len(wt["norm"])
    keep_s = None if sent_ids is None else set(int(i) for i in sent_ids)
    ntok = np.zeros(nW, np.int32)
    sent_words = [np.zeros(0, np.int64) for _ in range(N)]
    for i in range(N):
        g0, g1 = ptr[i], ptr[i + 1]
        if g1 == g0:
            continue
        ov = word_token_overlap(offs[i], wt["start"][g0:g1], wt["end"][g0:g1])
        ntok[g0:g1] = ov.sum(1)
        if keep_s is not None and i not in keep_s:
            continue
        use = np.where(need[g0:g1] & (ntok[g0:g1] > 0))[0]
        sent_words[i] = g0 + use
    order = sorted((i for i in range(N) if len(sent_words[i])), key=lambda i: len(ids[i]))
    batches, cur, curmax = [], [], 0
    for i in order:
        L = len(ids[i])
        if cur and (max(curmax, L) * (len(cur) + 1) > budget or len(cur) >= max_bs):
            batches.append(cur); cur, curmax = [], 0
        cur.append(i); curmax = max(curmax, L)
    if cur:
        batches.append(cur)
    row_of_g = np.full(nW, -1, np.int64); r = 0
    for b in batches:                                # thứ tự gán hàng = thứ tự ghi (batch, câu trong batch, từ trong câu)
        for i in b:
            for g in sent_words[i]:
                row_of_g[g] = r; r += 1
    n_cut = int(sum(1 for i in range(N) if len(ids[i]) >= cfg["max_len"]))
    return dict(ids=ids, offs=offs, sent_words=sent_words, batches=batches, row_of_g=row_of_g, n_rows=r,
                ntok=ntok, n_trunc=n_cut)

@torch.no_grad()
def extract_language(model, tok, cfg, plan, wt):
    """Trả về (buf fp16 [K1, n_rows, d], thống kê). Từng batch: forward -> bmm gộp sub-token -> ghi liền khối vào buf."""
    stats = dict(n_clip=0, pre_norm_used=False)
    if plan["n_rows"] == 0:
        return None, stats
    state = {"pre_norm": None}
    handles = []
    fn = get_final_norm(model, cfg)
    if fn is not None:
        def _hook(mod, args, kwargs):
            state["pre_norm"] = (args[0] if len(args) else next(iter(kwargs.values()))).detach()
        handles.append(fn.register_forward_pre_hook(_hook, with_kwargs=True))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ptr, start, end = wt["ptr"], wt["start"], wt["end"]
    buf, r0 = None, 0
    try:
        for b in plan["batches"]:
            B = len(b); L = max(len(plan["ids"][i]) for i in b); Wm = max(len(plan["sent_words"][i]) for i in b)
            ids = torch.full((B, L), pad, dtype=torch.long); am = torch.zeros((B, L), dtype=torch.long)
            A = np.zeros((B, Wm, L), np.float32); wmask = np.zeros((B, Wm), bool)
            for r, i in enumerate(b):
                n = len(plan["ids"][i]); gs = plan["sent_words"][i]
                ids[r, :n] = torch.tensor(plan["ids"][i]); am[r, :n] = 1
                ov = word_token_overlap(plan["offs"][i], start[gs], end[gs]).astype(np.float32)
                ov /= ov.sum(1, keepdims=True)                                   # trung bình các sub-token của từ
                A[r, :len(gs), :n] = ov; wmask[r, :len(gs)] = True
            ids, am = ids.to(DEVICE), am.to(DEVICE)
            A, wmask = torch.from_numpy(A).to(DEVICE), torch.from_numpy(wmask).to(DEVICE)
            state["pre_norm"] = None
            out = model(input_ids=ids, attention_mask=am, output_hidden_states=True, return_dict=True)
            hs = list(out.hidden_states)
            pn = state["pre_norm"]
            if pn is not None and tuple(pn.shape) == tuple(hs[-1].shape):
                hs[-1] = pn; stats["pre_norm_used"] = True
            n = int(wmask.sum())
            if buf is None:
                gb = len(hs) * plan["n_rows"] * hs[0].shape[-1] * 2 / 1e9
                buf = torch.empty((len(hs), plan["n_rows"], hs[0].shape[-1]), dtype=torch.float16,
                                  device=DEVICE if gb <= WORD_BUF_GPU_GB else "cpu")
            for k, h in enumerate(hs):
                sel = torch.bmm(A, h.float())[wmask]                               # [n, d], thứ tự (b, w)
                bad = (~torch.isfinite(sel)) | (sel.abs() > FP16_CLIP)
                stats["n_clip"] += int(bad.sum())
                sel = torch.nan_to_num(sel, nan=0.0, posinf=FP16_CLIP, neginf=-FP16_CLIP).clamp_(-FP16_CLIP, FP16_CLIP)
                buf[k, r0:r0 + n] = sel.to(buf.device, torch.float16)
            r0 += n
            del out, hs
    finally:
        for h in handles:
            h.remove()
    assert r0 == plan["n_rows"], (r0, plan["n_rows"])
    return buf, stats

@torch.no_grad()
def sanity_batch_vs_single(model, tok, cfg, sents, wt, need, n=24, tol=0.05, max_frac=0.01):
    """Biểu diễn từ khi batch (có padding) so với chạy từng câu. Lệch hệ thống -> ok=False (padding/mask/offset/số học bf16).
    v5: trả về dict(ok, frac, median, p99, max, n, dtype) — trước đây trả bool và run_model BỎ QUA giá trị trả về."""
    dt = str(next(model.parameters()).dtype)
    ids_ = [i for i in range(len(sents)) if need[wt["ptr"][i]:wt["ptr"][i + 1]].any()][:n]
    if len(ids_) < 2:
        return dict(ok=True, frac=0.0, median=0.0, p99=0.0, max=0.0, n=0, dtype=dt)
    pb = plan_language(tok, cfg, sents, wt, need, sent_ids=ids_, budget=10 ** 7, max_bs=len(ids_))
    ps = plan_language(tok, cfg, sents, wt, need, sent_ids=ids_, budget=10 ** 7, max_bs=1)
    bb, _ = extract_language(model, tok, cfg, pb, wt); bs, _ = extract_language(model, tok, cfg, ps, wt)
    g = np.where((pb["row_of_g"] >= 0) & (ps["row_of_g"] >= 0))[0]
    a = bb[:, torch.from_numpy(pb["row_of_g"][g]).to(bb.device)].float()
    b = bs[:, torch.from_numpy(ps["row_of_g"][g]).to(bs.device)].float()
    rel = ((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)).flatten()
    frac = float((rel > tol).float().mean())
    res = dict(ok=bool(frac <= max_frac), frac=frac, median=float(rel.median()), p99=float(torch.quantile(rel, 0.99)), max=float(rel.max()), n=int(rel.numel()), dtype=dt)
    print(f"  [sanity] batch vs single ({dt}): median={res['median']:.4f}, p99={res['p99']:.4f}, max={res['max']:.4f}; "
          f"{frac:.2%} giá trị > {tol} ->", "OK" if res["ok"] else "CẢNH BÁO: lệch hệ thống")
    return res
