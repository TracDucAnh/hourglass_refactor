# Trích biểu diễn FLORES (token cuối / trung bình), M1-M2, logit-lens (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần torch. Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
READOUTS = ("last", "mean")

_ALNUM = {}
def _has_alnum(tok_key):
    """has[id] = token có ít nhất một chữ / số (không phải dấu câu, khoảng trắng hay token đặc biệt)."""
    if tok_key not in _ALNUM:
        tok = get_tokenizer(tok_key)
        _ALNUM[tok_key] = np.array([i >= len(SPECIALS) and any(ch.isalnum() for ch in tok.decode([i])) for i in range(len(tok))])
    return _ALNUM[tok_key]

def eval_seqs(tok_key, langs):
    """FLORES devtest -> ({lang: [ [BOS] + token ]}, {lang: [vị trí đọc "token cuối"]}). Vị trí đọc là token cuối CÓ
    chữ / số: câu tự nhiên thường kết thúc bằng dấu câu, mà biểu diễn của dấu chấm ở tầng embedding giống hệt nhau trong
    một ngôn ngữ (G ở tầng 0 không xác định). Với họ causal, vị trí này vẫn đã thấy toàn bộ nội dung câu."""
    texts, tok, has = get_flores("devtest", langs), get_tokenizer(tok_key), _has_alnum(tok_key)
    seqs = {l: [[BOS_ID] + t[:SEQ_LEN - 1] for t in tok(v, add_special_tokens=False)["input_ids"]] for l, v in texts.items()}
    last = {l: [max([i for i, t in enumerate(s) if has[t]], default=len(s) - 1) for s in v] for l, v in seqs.items()}
    return seqs, last

def _pad(batch):
    ids = torch.full((len(batch), max(map(len, batch))), PAD_ID, dtype=torch.long)
    for i, s in enumerate(batch):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(DEVICE), torch.tensor([len(s) for s in batch], device=DEVICE)

@torch.no_grad()
def extract(model, family, seqs, last, bs=64):
    """-> (Z_last, Z_mean) [K1, L, N, d] float16. Token cuối = vị trí `last` (token cuối có chữ / số; với họ causal là
    vị trí đã thấy cả nội dung câu); trung bình = trung bình các token (bỏ BOS), như Study 1."""
    langs = list(seqs); N, d, K1 = len(seqs[langs[0]]), model.config.hidden_size, N_LAYERS + 1
    Zl, Zm = (np.zeros((K1, len(langs), N, d), np.float16) for _ in range(2))
    model.eval()
    for li, l in enumerate(langs):
        for b in range(0, N, bs):
            ids, lens = _pad(seqs[l][b:b + bs])
            with torch.autocast(**AMP) if AMP else torch.no_grad():
                H = torch.stack(hidden_states(model, family, ids)).float()          # [K1, B, T, d]
            pos = torch.arange(ids.shape[1], device=DEVICE)[None]
            m = ((pos >= 1) & (pos < lens[:, None])).float()
            lp = torch.tensor(last[l][b:b + bs], device=DEVICE)
            Zl[:, li, b:b + len(lens)] = H[:, torch.arange(len(lens), device=DEVICE), lp].cpu().numpy()
            Zm[:, li, b:b + len(lens)] = ((H * m[None, :, :, None]).sum(2) / m.sum(1)[None, :, None]).cpu().numpy()
    return Zl, Zm

@torch.no_grad()
def lens_mass(model, family, seqs, tok_lang, n_sub=200, bs=16):
    """Logit-lens (S4): M[k, b, l] = khối lượng xác suất token kế tiếp rơi vào bộ token riêng của ngôn ngữ l khi đọc câu
    ngôn ngữ b, áp final norm + lm_head lên residual ở tầng k; chuẩn hoá trên các token có chủ (bỏ token dùng chung)."""
    langs = list(seqs); L = len(langs)
    onehot = torch.zeros(len(tok_lang), L + 1, device=DEVICE)
    onehot[torch.arange(len(tok_lang)), torch.as_tensor(np.where(tok_lang < 0, L, tok_lang), device=DEVICE).long()] = 1
    M = np.zeros((N_LAYERS + 1, L, L), np.float32)
    for bi, l in enumerate(langs):
        acc, cnt = torch.zeros(N_LAYERS + 1, L + 1, device=DEVICE), 0.0
        for b in range(0, min(n_sub, len(seqs[l])), bs):
            ids, lens = _pad(seqs[l][b:b + bs])
            m = ((torch.arange(ids.shape[1], device=DEVICE)[None] >= 1) & (torch.arange(ids.shape[1], device=DEVICE)[None] < lens[:, None])).float()
            with torch.autocast(**AMP) if AMP else torch.no_grad():
                hs = hidden_states(model, family, ids)
            for k, h in enumerate(hs):
                p = torch.softmax(model.lm_head(model.model.norm(h)).float(), -1)               # [B, T, V]
                acc[k] += ((p @ onehot) * m[..., None]).sum((0, 1))
            cnt += float(m.sum())
        own = acc[:, :L] / acc[:, :L].sum(1, keepdim=True).clamp_min(1e-12)
        M[:, bi] = own.cpu().numpy()
    return M

def measure_run(model, c, variant):
    """Joint: D, G, CKA, A [cặp, K1] cho cả hai cách đọc (+ logit-lens với họ causal sau huấn luyện).
    Twin: biểu diễn của chính ngôn ngữ nó [K1, N, d] cho cả hai cách đọc (để tính A_indep khi phân tích)."""
    langs = langs_of(c)
    seqs, last = eval_seqs(c["tok"], langs)
    Zl, Zm = extract(model, c["family"], seqs, last)
    if c["kind"] == "twin":
        return dict(Z_last=Zl[:, 0], Z_mean=Zm[:, 0])
    out = {}
    for r, Z in zip(READOUTS, (Zl, Zm)):
        mets, pairs = pair_metrics(Z)
        out.update({f"{k}_{r}": v for k, v in mets.items()})
    out["pairs"] = np.array(pairs, np.int16)
    if variant == "trained" and c["family"] in CAUSAL:
        out["lens"] = lens_mass(model, c["family"], seqs, token_language(c["tok"], langs))
    return out
