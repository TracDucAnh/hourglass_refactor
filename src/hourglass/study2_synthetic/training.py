# Tác vụ LM / SEM / SURF + huấn luyện + giám sát hội tụ (Cell B2)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 21 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell B2 — Tác vụ (LM / SEM / SURF) và huấn luyện
# ============================================================
# Chuỗi trong pool: [bos, token..., eos, pad...]; n = số token khác pad.
# LM    : mục tiêu ngôn ngữ của họ (decoder/moe: next-token; encoder: MLM 15% theo 80/10/10; diffusion: che liên tục, loss/t).
# SEM   : thêm AUX_FRAC chuỗi mang nhãn phụ = lớp vị ngữ gốc (50 lớp, độc lập ngôn ngữ).
# SURF  : thêm AUX_FRAC chuỗi mang nhãn phụ = (độ dài ký tự, ký tự đầu) của TỪ THỨ HAI (cần thông tin bề mặt).
#   decoder/moe/diffusion: nhãn là token nối sau câu:  <eos> <sem|surf> <y>   (diffusion: <y> luôn bị che)
#   encoder: đầu tuyến tính trên vị trí 0 (<bos>) dự đoán nhãn
# Tổng loss = loss LM trung bình token + AUX_W * CE nhãn phụ (trung bình theo chuỗi có nhãn).
def pool_to_device(W, lang=None):
    sel = None if lang is None else (W["pool_lang"] == lang)
    out = {}
    for k in ("ids", "len", "sem", "surf"):
        a = W["pool_" + k]
        a = a if sel is None else a[sel]
        out[k] = torch.from_numpy(a.astype(np.int32 if k == "ids" else np.int64)).to(DEVICE)
    return out

def _with_label_suffix(ids, n, aux, y, task):
    full = ids.clone()
    bi = aux.nonzero(as_tuple=True)[0]
    if bi.numel():
        full[bi, n[bi]] = SEM_TOK if task == "SEM" else SURF_TOK
        full[bi, n[bi] + 1] = LABEL0 + y[bi]
    return full, bi

def compute_loss(model, pool, idx, family, task, g, V, opts=None):
    """opts (thí nghiệm chẩn đoán điểm neo, lưới mask_ablation): in_mask = tỉ lệ token nội dung ĐẦU VÀO của decoder/MoE bị thay
    bằng [MASK] chung (mục tiêu giữ nguyên); mlm_corrupt = "shuffle": encoder thay [MASK] bằng token khác của CHÍNH câu đó."""
    opts = opts or {}
    ids = pool["ids"][idx].long(); n = pool["len"][idx].long()
    B, T = ids.shape
    dev = ids.device
    if task == "LM":
        aux = torch.zeros(B, dtype=torch.bool, device=dev); y = torch.zeros(B, dtype=torch.long, device=dev)
    else:
        aux = torch.rand(B, device=dev, generator=g) < AUX_FRAC
        y = pool["sem" if task == "SEM" else "surf"][idx]
    if family in ("decoder", "moe"):
        full, bi = _with_label_suffix(ids, n, aux, y, task)
        inp = full[:, :-1]
        if opts.get("in_mask", 0.0) > 0:
            inp = inp.masked_fill((inp >= SYM0) & (torch.rand(inp.shape, device=dev, generator=g) < opts["in_mask"]), MASK)
        logits, _, _ = model(inp)
        tgt = full[:, 1:]
        ce = F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1), reduction="none").view(B, T - 1)
        j = torch.arange(T - 1, device=dev)[None, :]
        is_lab = aux[:, None] & (j == n[:, None]); is_mark = aux[:, None] & (j == (n[:, None] - 1))
        m = (tgt != PAD) & ~is_lab & ~is_mark
        lm = (ce * m).sum() / m.sum().clamp_min(1)
        ax = (ce * is_lab).sum() / is_lab.sum().clamp_min(1)
        return lm + AUX_W * ax, lm.detach()
    if family == "encoder":
        content = ids >= SYM0
        r = torch.rand(ids.shape, device=dev, generator=g)
        sel = content & (r < 0.15)
        none = ~sel.any(1)
        if none.any():
            rows = none.nonzero(as_tuple=True)[0]
            sel[rows, r.masked_fill(~content, 2.0)[rows].argmin(1)] = True
        u = torch.rand(ids.shape, device=dev, generator=g)
        inp = ids.clone()
        m80 = sel & (u < 0.8)
        if opts.get("mlm_corrupt", "mask") == "shuffle":              # không có token chung: lấy token nội dung ngẫu nhiên cùng câu
            order = torch.rand(ids.shape, device=dev, generator=g).masked_fill(~content, 2.0).argsort(1)
            shuf = ids.gather(1, order)                                 # n_content cột đầu = token nội dung của câu, thứ tự ngẫu nhiên
            inp[m80] = torch.cat([shuf[:, :1], shuf[:, :-1]], 1)[m80]  # vị trí nội dung j (1..n_content) lấy shuf[j - 1]
        else:
            inp[m80] = MASK
        m10 = sel & (u >= 0.8) & (u < 0.9)
        inp[m10] = torch.randint(SYM0, V, ids.shape, device=dev, generator=g)[m10]
        logits, _, hn = model(inp)
        lm = F.cross_entropy(logits[sel].float(), ids[sel])
        loss = lm
        if aux.any():
            loss = loss + AUX_W * F.cross_entropy(model.cls_head(hn[:, 0])[aux].float(), y[aux])
        return loss, lm.detach()
    # diffusion (MDLM / LLaDA): che mỗi token nội dung với xác suất t ~ U(0,1), loss CE / t
    full, bi = _with_label_suffix(ids, n, aux, y, task)
    maskable = (ids >= SYM0) | (ids == EOS)
    t = torch.rand(B, device=dev, generator=g).clamp_min(0.05)
    msk = maskable & (torch.rand(ids.shape, device=dev, generator=g) < t[:, None])
    inp = full.clone(); inp[msk] = MASK
    if bi.numel():
        inp[bi, n[bi] + 1] = MASK
    logits, _, _ = model(inp)
    ce = F.cross_entropy(logits.float().reshape(-1, V), full.reshape(-1), reduction="none").view(B, T)
    lm = (ce * msk / t[:, None]).sum() / maskable.sum().clamp_min(1)
    ax = ce[bi, n[bi] + 1].mean() if bi.numel() else ce.new_zeros(())
    return lm + AUX_W * ax, lm.detach()

# ---------- Giám sát hội tụ: val (giữ ra) / trn (pool) với mặt nạ cố định ----------
_VAL_POOLS = {}
def get_val_pool(W, langs=None):
    """Pool giữ ra (tổ hợp R28) dạng {ids,len,sem,surf} trên DEVICE; langs=None -> mọi ngôn ngữ, mỗi ngôn ngữ VAL_N // |langs| câu."""
    L = W["eval_ids"].shape[0]
    langs = list(range(L)) if langs is None else list(langs)
    key = (W["meta"]["id"], tuple(langs), VAL_N)
    if key not in _VAL_POOLS:
        n_per = max(1, VAL_N // len(langs))
        ids = W["eval_ids"][langs][:, :n_per].reshape(-1, MAX_LEN).astype(np.int32)
        t = torch.from_numpy(ids).to(DEVICE)
        z = torch.zeros(t.shape[0], dtype=torch.long, device=DEVICE)
        _VAL_POOLS[key] = dict(ids=t, len=(t != PAD).sum(1).long(), sem=z, surf=z)
    return _VAL_POOLS[key]

@torch.no_grad()
def lm_loss_on(model, family, vpool, V, n_max=None, seed=777, opts=None):
    """Loss LM (không gồm nhãn phụ) của MỌI họ trên pool cho trước. Mặt nạ (encoder/diffusion) dùng generator cố định
    nên các lần gọi so sánh được với nhau. Với decoder/MoE đây là next-token CE thuần."""
    was = model.training; model.eval()
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    N = min(vpool["ids"].shape[0], n_max or VAL_N)
    tot, cnt = 0.0, 0
    for s in range(0, N, VAL_BS):
        idx = torch.arange(s, min(N, s + VAL_BS), device=DEVICE)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            _, lm = compute_loss(model, vpool, idx, family, "LM", g, V, opts)
        tot += float(lm) * len(idx); cnt += len(idx)
    model.train(was)
    return tot / max(cnt, 1)

# ---------- Sàn entropy của ngôn ngữ tổng hợp (decoder/MoE: next-token loss không thể thấp hơn) ----------
_HC = {}
def content_entropy_nats(n=30000, seed=123):
    """H(C) (nats) của nội dung huấn luyện (sau R27/R28), ước lượng Monte Carlo: -E[log p(C)] + log Z, Z = xác suất chấp nhận."""
    if (n, seed) in _HC:
        return _HC[(n, seed)]
    pidx = {k: {c: i for i, c in enumerate(v)} for k, v in POOLS.items()}
    lpk = lambda pool, c: math.log(POOL_PROBS[pool][pidx[pool][c]])
    def lp_np(x, depth):
        _, noun, num, df_, adjs, ext = x
        lp = lpk("N", noun) + math.log(P_PL if num else 1 - P_PL) + math.log(P_DEF if df_ else 1 - P_DEF)
        lp += math.log(P_NADJ[len(adjs)]) + sum(lpk("ADJ", a_) for a_ in adjs)
        if depth < MAX_DEPTH:
            if ext is None: lp += math.log(1 - P_POSS - P_REL)
            elif ext[0] == "POSS": lp += math.log(P_POSS) + lp_np(ext[1], depth + 1)
            else: lp += math.log(P_REL) + lp_cl(ext[1], depth + 1, True)
        return lp
    def lp_cl(S, depth, intr=False):
        _, pred, tense, neg, adv, args = S
        val_ = len(args)
        lp = 0.0 if intr else math.log(P_INTR if val_ == 1 else 1 - P_INTR)
        lp += lpk("V1" if val_ == 1 else "V2", pred) + math.log(P_TENSE[tense]) + math.log(P_NEG if neg else 1 - P_NEG)
        lp += (math.log(P_ADV) + lpk("ADV", adv)) if adv >= 0 else math.log(1 - P_ADV)
        return lp + sum(lp_np(a_, depth) for a_ in args)
    rng = random.Random(seed)
    raw = [sample_clause(rng, 1) for _ in range(n)]
    acc = [c for c in raw if clause_words(c) >= MIN_WORDS and not is_heldout(c)]
    out = -float(np.mean([lp_cl(c, 1) for c in acc])) + math.log(len(acc) / n)
    _HC[(n, seed)] = out
    return out

def entropy_floor(W):
    """Sàn next-token loss (nats/token) của một thế giới = (H(C) + H(lang)) / số token đích trung bình mỗi câu. Xấp xỉ Monte Carlo."""
    sh = np.asarray(W["meta"]["cfg"]["shares"], float); sh = sh / sh.sum()
    h_lang = float(-(sh * np.log(sh)).sum())
    tgt = float((W["pool_len"].astype(np.float64) - 1).mean())
    return (content_entropy_nats() + h_lang) / tgt

def convergence_summary(info, floor=None):
    gap = info["val_loss"] - info["trn_loss"]
    return dict(conv_drop=info["conv_drop"], converged=bool(info["conv_drop"] < CONV_TOL), val_trn_gap=float(gap),
                overfit=bool(gap > OVERFIT_TOL), gap_to_floor=None if floor is None else float(info["val_loss"] - floor))

def train_model(model, family, task, W, steps, seed, lang=None, log=True, desc=None, leave=True, hp=None, val_langs=None):
    """hp: dict từ train_hparams (batch, warmup, lr_scale); None -> mặc định toàn cục HP.
    Đo val/trn ~12 lần trong lúc train. Trả về cả history để kiểm hội tụ; KHÔNG dừng sớm (mọi điều kiện cùng ngân sách)."""
    hp = hp or HP
    BS, LRS, WARM = hp["batch"], hp["lr_scale"], hp["warmup"]
    pool = pool_to_device(W, lang)
    N, V = pool["ids"].shape[0], W["meta"]["vocab_total"]
    vpool = get_val_pool(W, val_langs)
    decay = [p for n_, p in model.named_parameters() if p.ndim >= 2 and not n_.startswith("emb")]
    nodecay = [p for n_, p in model.named_parameters() if not (p.ndim >= 2 and not n_.startswith("emb"))]
    lr_peak = (1e-3 if model.d <= 128 else 6e-4) * LRS            # LRS = (BS/BATCH_REF)**LR_BATCH_EXP (Cell A1)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=lr_peak, betas=(0.9, 0.95), fused=(DEVICE == "cuda"))
    g = torch.Generator(device=DEVICE); g.manual_seed(5000 + seed)
    warm = max(1, min(WARM, steps // 5))
    eval_every = max(1, steps // 12)
    hist, ema, t0 = [], None, time.time()
    model.train()
    bar = tqdm(range(steps), desc=desc or f"train {family}/{task}", unit="step", disable=not log, leave=leave,
               mininterval=1.0, dynamic_ncols=True)
    for step in bar:
        lr = lr_peak * (step + 1) / warm if step < warm else \
            lr_peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))
        for gr in opt.param_groups:
            gr["lr"] = lr
        idx = torch.randint(0, N, (BS,), device=DEVICE, generator=g)
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            loss, lm = compute_loss(model, pool, idx, family, task, g, V, hp)
            if family == "moe":
                loss = loss + MOE_AUX * model.aux_loss()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            v = float(lm)
            ema = v if ema is None else 0.9 * ema + 0.1 * v
            if log:                                  # chỉ cập nhật postfix khi đã sync GPU (mỗi 25 bước), tránh làm chậm vòng lặp
                bar.set_postfix(lm_loss=f"{ema:.3f}", lr=f"{lr:.1e}", bs=BS, refresh=False)
        if (step + 1) % eval_every == 0 or step == steps - 1:
            rec = dict(step=step + 1, seqs=(step + 1) * BS, lr=float(lr), train_ema=float(ema),
                       val=lm_loss_on(model, family, vpool, V, opts=hp), trn=lm_loss_on(model, family, pool, V, opts=hp))
            hist.append(rec)
            if log:
                bar.set_postfix(lm_loss=f"{ema:.3f}", val=f"{rec['val']:.3f}", trn=f"{rec['trn']:.3f}", lr=f"{lr:.1e}", bs=BS, refresh=False)
    bar.close()
    model.eval()
    last = hist[-1]
    ref = [h for h in hist if h["step"] <= 0.85 * steps]            # mốc ~5/6 quá trình (annealing lr đã gần xong)
    ref = ref[-1] if ref else hist[0]
    return dict(final_lm_loss=float(ema), steps=steps, batch=BS, seconds=round(time.time() - t0, 1), history=hist,
                val_loss=float(last["val"]), trn_loss=float(last["trn"]), conv_drop=float(ref["val"] - last["val"]))

@torch.no_grad()
def eval_lm_loss(model, W, n=256):
    """Perplexity (nats/token) trên tập giữ ra cho decoder/moe (kiểm tra học được; họ khác trả None)."""
    if model.family not in ("decoder", "moe"):
        return None
    ids = torch.from_numpy(W["eval_ids"][:, :n].reshape(-1, MAX_LEN).astype(np.int64)).to(DEVICE)
    tot, cnt = 0.0, 0
    for s in range(0, ids.shape[0], EVAL_BS):
        b = ids[s:s + EVAL_BS]
        logits, _, _ = model(b[:, :-1])
        tgt = b[:, 1:]; m = tgt != PAD
        tot += float((F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none") * m.reshape(-1)).sum()); cnt += int(m.sum())
    return tot / max(cnt, 1)
