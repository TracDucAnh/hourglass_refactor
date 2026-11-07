# Dòng token, loss theo họ, vòng huấn luyện, giám sát hội tụ (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần torch. Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.

def data_mult(c, lang):
    """Hệ số dữ liệu của một ngôn ngữ trong run c: 1, hoặc S4_SHARE_MULT cho S4_LANG khi share = "x<m>"."""
    return float(c["share"][1:]) if (c["share"] != "eq" and lang == S4_LANG) else 1.0

class Streams:
    """Các chuỗi SEQ_LEN token của một run. Mỗi ngôn ngữ dùng phần đầu (frac x hệ số) của vùng huấn luyện; đuôi
    VAL_FRAC của corpus là tập val, không bao giờ huấn luyện. Cùng một ngôn ngữ và frac -> cùng dữ liệu ở mọi |L|."""
    def __init__(self, c, epochs):
        self.langs = langs_of(c)
        self.arrs = [load_tokens(c["tok"], l) for l in self.langs]
        idx, self.val = [], []
        for i, (l, a) in enumerate(zip(self.langs, self.arrs)):
            n_val = max(SEQ_LEN * VAL_CHUNKS, int(len(a) * VAL_FRAC))
            n_tr = len(a) - n_val
            use = min(n_tr, int(n_tr * c["frac"] * data_mult(c, l)))
            if c["frac"] * data_mult(c, l) > 1:
                print(f"  [warn] {l}: cần {c['frac'] * data_mult(c, l):.2f} corpus, chỉ có 1.0 -> tăng CORPUS_BYTES_ENG")
            idx += [(i, s) for s in range(0, use - SEQ_LEN, SEQ_LEN)]
            self.val.append([(i, n_tr + s) for s in range(0, n_val - SEQ_LEN, SEQ_LEN)][:VAL_CHUNKS])
        self.idx, self.epochs = np.array(idx, np.int64), epochs
        assert len(self.idx) >= BATCH_SEQS, f"chỉ có {len(self.idx)} chuỗi < BATCH_SEQS={BATCH_SEQS}: tăng dữ liệu hoặc giảm BATCH_SEQS"
        self.steps = max(1, int(len(self.idx) * epochs) // BATCH_SEQS)
        self.tokens_per_lang = {l: int((self.idx[:, 0] == i).sum()) * SEQ_LEN for i, l in enumerate(self.langs)}

    def fetch(self, rows):
        return torch.from_numpy(np.stack([self.arrs[i][s:s + SEQ_LEN].astype(np.int64) for i, s in rows]))

    def batches(self, seed):
        """Xáo lại mỗi epoch (theo seed); dừng sau đúng self.steps lô."""
        rng, done = np.random.default_rng(seed), 0
        while True:
            for b in range(0, len(self.idx) - BATCH_SEQS + 1, BATCH_SEQS):
                if done == self.steps:
                    return
                if b == 0:
                    order = rng.permutation(len(self.idx))
                yield self.fetch(self.idx[order[b:b + BATCH_SEQS]]); done += 1

def lm_loss(model, family, x, g=None):
    """Trả về (loss để lan truyền ngược, loss LM để báo cáo). Decoder / MoE: dự đoán token kế tiếp (MoE cộng loss cân bằng
    tải). Diffusion (MDLM): che mỗi token với xác suất t ~ U(DIFF_T_MIN, 1), loss CE trên token bị che, trọng số 1/t."""
    x = x.to(DEVICE)
    if family in CAUSAL:
        out = model(input_ids=x, labels=x)
        lm = out.loss - (MOE_AUX * out.aux_loss if family == "moe" else 0.0)
        return out.loss, lm.detach()
    B, T = x.shape
    t = torch.rand(B, device=DEVICE, generator=g) * (1 - DIFF_T_MIN) + DIFF_T_MIN
    m = torch.rand(B, T, device=DEVICE, generator=g) < t[:, None]
    m[m.sum(1) == 0, 0] = True
    logits = model(input_ids=x.masked_fill(m, MASK_ID), attention_mask=attn_mask(family, x)).logits
    ce = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), x.reshape(-1), reduction="none").view(B, T)
    loss = (ce * m / t[:, None]).sum() / (B * T)
    return loss, loss.detach()

@torch.no_grad()
def val_loss(model, family, st, langs=None):
    """Loss LM trên tập val của từng ngôn ngữ; mặt nạ diffusion cố định (generator riêng) nên các lần đo so sánh được."""
    was = model.training; model.eval(); out = {}
    for i, l in enumerate(st.langs):
        if langs and l not in langs:
            continue
        g = torch.Generator(device=DEVICE).manual_seed(777)
        tot = []
        for b in range(0, len(st.val[i]), 32):
            with torch.autocast(**AMP) if AMP else torch.no_grad():
                tot.append(float(lm_loss(model, family, st.fetch(st.val[i][b:b + 32]), g)[1]))
        out[l] = float(np.mean(tot))
    model.train(was)
    return out

def lr_at(step, steps, peak):
    warm = max(1, int(steps * WARMUP_FRAC))
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, steps - warm))))

def train_model(model, family, c, st, seed, desc=None):
    """AdamW, warmup rồi cosine về MIN_LR_FRAC; không dừng sớm. Đo val N_EVALS lần; conv_drop = val còn giảm bao nhiêu
    từ mốc ~85% bước tới cuối (như Study 2a)."""
    peak = LR_BY_D.get(c["d"], LR_BY_D[min(LR_BY_D, key=lambda d: abs(d - c["d"]))])
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=peak, betas=(0.9, 0.95))
    g = torch.Generator(device=DEVICE).manual_seed(5000 + seed)
    every, hist, ema, t0 = max(1, st.steps // N_EVALS), [], None, time.time()
    model.train()
    bar = tqdm(st.batches(seed), total=st.steps, desc=desc or "train", unit="step", mininterval=2.0, dynamic_ncols=True)
    for step, x in enumerate(bar):
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, st.steps, peak)
        if AMP:
            with torch.autocast(**AMP):
                loss, lm = lm_loss(model, family, x, g)
        else:
            loss, lm = lm_loss(model, family, x, g)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % 25 == 0 or step == st.steps - 1:
            ema = float(lm) if ema is None else 0.9 * ema + 0.1 * float(lm)
            bar.set_postfix(lm=f"{ema:.3f}", refresh=False)
        if (step + 1) % every == 0 or step == st.steps - 1:
            v = val_loss(model, family, st)
            hist.append(dict(step=step + 1, train_ema=ema, val=float(np.mean(list(v.values()))), val_by_lang=v))
    bar.close(); model.eval()
    ref = [h for h in hist if h["step"] <= 0.85 * st.steps] or hist[:1]
    conv_drop = float(ref[-1]["val"] - hist[-1]["val"])
    return dict(steps=st.steps, seconds=round(time.time() - t0, 1), history=hist, val_loss=hist[-1]["val"],
                val_by_lang=hist[-1]["val_by_lang"], conv_drop=conv_drop, converged=bool(conv_drop < CONV_TOL),
                tokens_per_lang=st.tokens_per_lang, lr_peak=peak)
