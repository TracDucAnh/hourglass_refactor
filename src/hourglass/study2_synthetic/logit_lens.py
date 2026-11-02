# Logit-lens: ngôn ngữ trục (S4) (Cell B5)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 24 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell B5 — Logit-lens: ai là "ngôn ngữ trục" (pivot) ở tầng giữa? (S4, chỉ decoder, overlap 0%)
# ============================================================
# Với câu thuộc ngôn ngữ b, áp final norm + ma trận unembedding lên residual của MỖI tầng, rồi tính khối lượng xác suất
# của token kế tiếp rơi vào bộ token của từng ngôn ngữ l (token -> ngôn ngữ nhờ tok_lang, hợp lệ khi overlap 0%).
# M[k, b, l] = khối lượng trung bình (đã chuẩn hóa trên các token có chủ). Ngôn ngữ trục = l có
# pivot_score(l) = mean_{b != l} M[k, b, l] lớn nhất quanh tầng giữa.
@torch.no_grad()
def logit_lens_mass(model, ids_all, tok_lang, n_sub=300, bs=None, progress=True):
    bs = bs or min(EVAL_BS, n_sub)
    L = ids_all.shape[0]
    n_use = min(n_sub, ids_all.shape[1])
    tl = torch.from_numpy(tok_lang.astype(np.int64)).to(DEVICE)
    onehot = torch.zeros(len(tl), L, device=DEVICE)
    own = tl >= 0
    onehot[own.nonzero(as_tuple=True)[0], tl[own]] = 1.0
    K1 = len(model.blocks) + 1
    # tích lũy trên GPU, chỉ chuyển về CPU một lần ở cuối (tránh K1 x số batch lần đồng bộ)
    M = torch.zeros((K1, L, L), dtype=torch.float64, device=DEVICE); C = torch.zeros((L,), dtype=torch.float64, device=DEVICE)
    jobs = [(b, s) for b in range(L) for s in range(0, n_use, bs)]
    for b, s in tqdm(jobs, desc="logit-lens", unit="batch", leave=False, disable=not progress, dynamic_ncols=True):
        ids = torch.from_numpy(ids_all[b, s:min(s + bs, n_use)].astype(np.int64)).to(DEVICE)
        _, hs, _ = model(ids)
        valid = (ids[:, :-1] != PAD) & (ids[:, 1:] >= SYM0)              # vị trí mà token kế tiếp là ký hiệu nội dung
        for k, h in enumerate(hs):
            p = F.softmax(F.linear(model.norm(h[:, :-1]), model.emb.weight).float(), -1)
            mass = p @ onehot
            mass = mass / mass.sum(-1, keepdim=True).clamp_min(1e-9)
            M[k, b] += (mass * valid.unsqueeze(-1)).sum((0, 1)).double()
        C[b] += valid.sum().double()
    M, C = M.cpu().numpy(), C.cpu().numpy()
    return (M / np.maximum(C[None, :, None], 1)).astype(np.float32)
