# Chẩn đoán hubness CSLS (Cell E1)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 44 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell E1 — CHẨN ĐOÁN thiên lệch hubness của CSLS: R theo log-hạng với cosine / CSLS-theo-loại / CSLS-theo-instance (cách cũ)
# Cần cache biểu diễn mức từ (đặt KEEP_WORD_CACHE = True ở Cell A1 trước khi chạy model). Nếu "instance" thấp hơn rõ rệt ở các bin đầu so với "cos"/"type"
# thì nhánh trái cũ là artifact của CSLS-theo-instance. Giá trị = RR trung bình theo instance (phía a->b) trong từng khoảng log10-hạng của từ nguồn.
# ============================================================
@torch.no_grad()
def diag_hub_modes(key, layer=None, pair_names=("eng-fra", "eng-rus", "eng-vie", "eng-zho"), bins=(0, 2, 3, 4, 5, 9)):
    cfg = MODELS[key]; run, meta = load_run(key); kw, K1 = waist_layer(key, run); layer = kw if layer is None else layer
    tok = load_tokenizer(cfg)
    plans = {l: plan_language(tok, cfg, TEXTS[MEAS][l], WT[MEAS][l], NEED[l]) for l in LANGS}
    prow = pair_rows(ITEMS, plans)
    mms = [np.load(cache_path(key, l), mmap_mode="r") for l in LANGS]
    Xs = [torch.from_numpy(np.array(m[layer])).to(DEVICE).float() for m in mms]
    n_all = sum(x.shape[0] for x in Xs); mu = sum(x.sum(0) for x in Xs) / n_all
    sd = (sum(((x - mu) ** 2).sum(0) for x in Xs) / n_all).sqrt().clamp_min(1e-6)
    Xs = [(x - mu) / sd for x in Xs]; Xs = [x - x.mean(0, keepdim=True) for x in Xs]
    out, labels = {}, [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    for name in pair_names:
        pi = PAIR_NAME.index(name); ia, ib = PAIRS[pi]; s, e, ra, rb, v = prow[pi]
        ga, gb = ITEMS["wa"][s:e][v], ITEMS["wb"][s:e][v]
        Xa = F.normalize(Xs[ia][torch.from_numpy(ra[v]).to(DEVICE)], dim=1); Xb = F.normalize(Xs[ib][torch.from_numpy(rb[v]).to(DEVICE)], dim=1)
        C = Xa @ Xb.T
        ta = torch.from_numpy(TID[LANGS[ia]][ga]).to(DEVICE); tb = torch.from_numpy(TID[LANGS[ib]][gb]).to(DEVICE)
        la = LANGS[ia]; lr = np.array([RANKS[la].log_rank(WT[MEAS][la]["norm"][g])[0] for g in ga]); bi = np.digitize(lr, bins[1:-1])
        for mode in ("cos", "type", "instance"):
            rr = csls_mrr(C, ta, tb, mode=mode)[0].cpu().numpy()
            out[(name, mode)] = [float(rr[bi == i].mean()) if (bi == i).any() else np.nan for i in range(len(labels))]
    return pd.DataFrame(out, index=labels).T.round(3)

# Ví dụ (sau khi chạy model với KEEP_WORD_CACHE = True):  diag_hub_modes("qwen3_1.7b")
