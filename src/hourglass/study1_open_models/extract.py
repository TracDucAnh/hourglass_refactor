# Trích xuất biểu diễn theo tầng (Cell B2)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 5 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell B2 — Trích xuất biểu diễn
# Z_k^l(c): residual stream tầng k (tầng cuối lấy TRƯỚC final norm), mean-pool trên token nội dung (bỏ BOS/CLS/EOS/SEP/PAD)
# Diffusion: che ngẫu nhiên t*100% token nội dung, cùng seed cho mọi ngôn ngữ, trung bình N_MASK_DRAWS lần;
#            pool riêng "all" (mọi token nội dung) và "masked" (chỉ token bị che)
# MoE: hook trực tiếp gate mỗi tầng, lưu softmax router logits pool theo câu
# ============================================================
def pool_tokens(x, mask):
    """x: [B,S,H]; mask: [B,S] bool -> trung bình trên các token có mask (cộng dồn float32)."""
    m = mask.float().to(x.device).unsqueeze(-1)
    return (x.float() * m).sum(1) / m.sum(1).clamp_min(1.0)       # cộng dồn float32 (bản cũ cộng bf16)

def apply_mask(ids, content, idx, t, r, mask_id):
    """Che ~t token nội dung mỗi câu. Seed chỉ phụ thuộc (SEED, chỉ số câu, t, lần che) nên giống nhau ở mọi ngôn ngữ."""
    B, S = ids.shape
    cont = content.cpu().numpy()
    rows = []
    for b in range(B):
        rng = np.random.default_rng([SEED, int(idx[b]), int(round(t * 100)), int(r)])
        u = np.where(cont[b], rng.random(S), 2.0)
        m = u < t
        if not m.any() and cont[b].any():
            m[int(u.argmin())] = True                              # luôn che ít nhất 1 token
        rows.append(m)
    mm = torch.from_numpy(np.stack(rows)).to(ids.device)
    return ids.masked_fill(mm, mask_id), mm

def build_inputs(cfg, enc, ids):
    inp = {"input_ids": ids, "attention_mask": enc["attention_mask"]}
    if cfg["family"] == "encoder" and "token_type_ids" in enc:
        inp["token_type_ids"] = enc["token_type_ids"]
    if "dream" in cfg["id"].lower():            # [XM] theo cách generation_utils của Dream chuẩn bị mask/vị trí
        am = enc["attention_mask"]
        if torch.any(am == 0):
            tok_idx = am.long().cumsum(-1) - 1
            tok_idx.masked_fill_(am == 0, 1)
            inp["attention_mask"] = torch.logical_and(am.unsqueeze(1).unsqueeze(-2), am.unsqueeze(1).unsqueeze(-1))
            inp["position_ids"] = tok_idx
        else:
            inp["attention_mask"] = "full"
    return inp

@torch.no_grad()
def sanity_padding_check(model, tok, cfg, texts_by_lang, n=8, tol=0.05, max_frac=0.01):
    """So biểu diễn batch (có padding) với chạy từng câu (không padding).
    Báo lỗi (trả về False -> run_one chuyển sang batch_size=1) khi quá max_frac số giá trị (tầng x ngôn ngữ x câu) lệch hơn tol.
    Vài ngoại lệ lẻ tẻ được coi là nhiễu số học theo từng câu/tầng, không phải lỗi padding."""
    langs = [l for l in ("eng_Latn", "tha_Thai", "hin_Deva") if l in texts_by_lang]
    sub = {l: texts_by_lang[l][:n] for l in langs}
    t0 = 0.0 if cfg["family"] == "diffusion" else None
    a = extract_level(model, tok, cfg, sub, t=t0, batch_size=n)["all"].float()
    b = extract_level(model, tok, cfg, sub, t=t0, batch_size=1)["all"].float()
    rel = (a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-6)               # [K1, n_lang, n]
    flat = rel.flatten()
    n_bad = int((flat > tol).sum()); frac = n_bad / flat.numel()
    med, p99, worst = flat.median().item(), torch.quantile(flat, 0.99).item(), flat.max().item()
    kw, lw, sw = np.unravel_index(int(rel.argmax()), tuple(rel.shape))
    ok = frac <= max_frac
    print(f"  [sanity] batch vs single: median={med:.4f}, p99={p99:.4f}, tệ nhất={worst:.4f} "
          f"(tầng {kw}, {langs[lw]}, câu {sw}); {n_bad}/{flat.numel()} giá trị > {tol} ({frac:.2%}) ->",
          "OK" if ok else "CẢNH BÁO: sai lệch hệ thống (padding/mask/position ids?) -> sẽ chạy batch_size=1")
    if ok and n_bad:
        print("     (ngoại lệ hiếm lẻ tẻ: coi là nhiễu số học, không phải lỗi padding)")
    return ok

def _to_fp16(x, stats):
    bad = (~torch.isfinite(x)) | (x.abs() > FP16_CLIP)
    stats["n_clip"] += int(bad.sum().item())
    return torch.nan_to_num(x, nan=0.0, posinf=FP16_CLIP, neginf=-FP16_CLIP).clamp(-FP16_CLIP, FP16_CLIP) \
                .to(torch.float16).cpu()

from tqdm.auto import tqdm

@torch.no_grad()
def extract_level(model, tok, cfg, texts_by_lang, t=None, n_draws=1, batch_size=None):
    """Một mức che t (None = không phải diffusion). Trả về buffer fp16 [K1, n_lang, N, H] cho 'all' (và 'masked' nếu t>0)."""
    langs = list(texts_by_lang); nL = len(langs); N = len(texts_by_lang[langs[0]])
    bs, max_len = (batch_size or cfg["bs"]), cfg["max_len"]
    do_mask = (t is not None and t > 0)
    n_draws = n_draws if do_mask else 1
    mask_id = cfg["mask_id"] if cfg["mask_id"] is not None else getattr(tok, "mask_token_id", None)
    if do_mask and mask_id is None:
        raise ValueError("Không xác định được mask token id cho mô hình diffusion.")

    special = {tok.bos_token_id, tok.eos_token_id, tok.cls_token_id, tok.sep_token_id, tok.pad_token_id}
    special.discard(None)
    special_t = torch.tensor(sorted(special), device=DEVICE)

    final_norm = get_final_norm(model, cfg)
    gates = find_router_gates(model) if cfg["router"] else []
    if cfg["router"] and not gates:
        print("  [warn] không tìm thấy router gate (layers.N.mlp.gate) -> bỏ qua J_k")
    state = {"mask": None, "router": {}, "pre_norm": None}
    stats = {"n_trunc": 0, "n_clip": 0, "pre_norm_used": False}
    handles = []
    if final_norm is not None:
        def _norm_hook(mod, args, kwargs):
            x = args[0] if len(args) else next(iter(kwargs.values()))
            state["pre_norm"] = x.detach()
        handles.append(final_norm.register_forward_pre_hook(_norm_hook, with_kwargs=True))
    for gi, (_, gate) in enumerate(gates):
        def _gate_hook(mod, args, kwargs, gi=gi):
            x = args[0] if len(args) else next(iter(kwargs.values()))
            B, S = state["mask"].shape
            p = F.linear(x.reshape(B * S, -1).float(), mod.weight.float()).softmax(-1).reshape(B, S, -1)
            m = state["mask"].float().unsqueeze(-1)
            state["router"][gi] = (p * m).sum(1) / m.sum(1).clamp_min(1.0)
        handles.append(gate.register_forward_pre_hook(_gate_hook, with_kwargs=True))

    buf_all = buf_msk = rbuf = None
    t_lang = time.time()
    try:
        for li, lang in enumerate(langs):
            sents = texts_by_lang[lang]
            order = sorted(range(N), key=lambda i: len(sents[i]))      # gom câu cùng độ dài để giảm padding
            for s in tqdm(range(0, N, bs), desc=f"{lang} ({li + 1}/{nL})", leave=False,
                          disable=(N <= 16)):          # tắt thanh tiến độ ở sanity check (N nhỏ)
                idx = order[s:s + bs]
                batch = [sents[i] for i in idx]
                full = tok(batch, add_special_tokens=True, truncation=False)["input_ids"]
                stats["n_trunc"] += sum(len(x) > max_len for x in full)
                enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
                enc = {k: v.to(DEVICE) for k, v in enc.items()}
                ids = enc["input_ids"]
                content = enc["attention_mask"].bool() & ~torch.isin(ids, special_t)
                state["mask"] = content
                acc_all = acc_msk = None
                for r in range(n_draws):
                    if do_mask:
                        ids_in, mmask = apply_mask(ids, content, idx, t, r, mask_id)
                    else:
                        ids_in, mmask = ids, None
                    state["pre_norm"] = None; state["router"] = {}
                    out = model(**build_inputs(cfg, enc, ids_in), output_hidden_states=True, return_dict=True)
                    hs = list(getattr(out, "hidden_states", None) or [])
                    if not hs:
                        raise RuntimeError(f"{cfg['id']} không trả về hidden_states.")
                    pn = state["pre_norm"]
                    if pn is not None and tuple(pn.shape) == tuple(hs[-1].shape):
                        hs[-1] = pn; stats["pre_norm_used"] = True
                    pooled = torch.stack([pool_tokens(h, content) for h in hs])          # [K1,B,H]
                    acc_all = pooled if acc_all is None else acc_all + pooled
                    if do_mask:
                        pm = torch.stack([pool_tokens(h, mmask) for h in hs])
                        acc_msk = pm if acc_msk is None else acc_msk + pm
                    del out, hs
                idx_t = torch.tensor(idx)
                K1, B, H = acc_all.shape
                if buf_all is None:
                    buf_all = torch.empty((K1, nL, N, H), dtype=torch.float16)
                    if do_mask:
                        buf_msk = torch.empty((K1, nL, N, H), dtype=torch.float16)
                view = buf_all[:, li]; view[:, idx_t] = _to_fp16(acc_all / n_draws, stats)
                if do_mask:
                    view = buf_msk[:, li]; view[:, idx_t] = _to_fp16(acc_msk / n_draws, stats)
                if gates and len(state["router"]) == len(gates):
                    rp = torch.stack([state["router"][g] for g in range(len(gates))])    # [Lm,B,E]
                    if rbuf is None:
                        rbuf = torch.empty((nL, len(gates), N, rp.shape[-1]), dtype=torch.float16)
                    view = rbuf[li]; view[:, idx_t] = rp.to(torch.float16).cpu()
            if N > 16:
                print(f"    {lang}: ok ({time.time() - t_lang:.0f}s)", flush=True)
            t_lang = time.time()
    finally:
        for h in handles:
            h.remove()
    return dict(all=buf_all, masked=buf_msk, router=rbuf, gate_layers=[g[0] for g in gates],
                langs=langs, stats=stats)
