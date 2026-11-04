# Trích biểu diễn gộp theo từ (Cell B3)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 22 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell B3 — Trích biểu diễn (gộp theo từ)
# ============================================================
def _pool_words(hs, wid):
    """hs: K+1 tensor [B,T,d]; wid: [B,T] (-1 = không thuộc từ). Trả về (câu [K1,B,d], từ [K1,B,W,d], token cuối [K1,B,d]).
    Mỗi từ = trung bình sub-token của nó; câu = trung bình các từ (mỗi từ trọng số như nhau).
    Token cuối = trạng thái ở token nội dung cuối câu: với họ causal đây là vị trí DUY NHẤT đã thấy cả câu (trung bình theo từ
    trộn các tiền tố khác nhau khi trật tự từ khác nhau -> thiên lệch có lợi cho họ hai chiều)."""
    H = torch.stack(hs).float()
    W_ = int(wid.max().item()) + 1 if (wid >= 0).any() else 1
    oh = (F.one_hot(wid.clamp_min(0), W_) * (wid >= 0).unsqueeze(-1)).float()
    cnt = oh.sum(1)
    wv = torch.einsum("btw,kbtd->kbwd", oh, H) / cnt.clamp_min(1)[None, :, :, None]
    valid = (cnt > 0).float()
    sent = (wv * valid[None, :, :, None]).sum(2) / valid.sum(1).clamp_min(1)[None, :, None]
    pos = torch.arange(wid.shape[1], device=wid.device)
    li = torch.where(wid >= 0, pos, -1).max(1).values.clamp_min(0)
    last = H[:, torch.arange(H.shape[1], device=H.device), li]
    return sent, wv, last

@torch.no_grad()
def extract_eval(model, ids_all, wid_all, t=None, n_draws=1, pick=None, bs=None, desc=None, progress=True, last=False):
    """ids_all, wid_all: np [L,N,T]. Trả về Z [K1,L,N,d] fp16 (câu) và Zp (vector từ `pick` = chỉ số từ mỗi nội dung) nếu có.
    last=True: trả thêm Zl [K1,L,N,d] (trạng thái ở token nội dung cuối), tức (Z, Zp, Zl).
    t>0 (diffusion): che ~t token trong từ; cùng vector ngẫu nhiên theo chỉ số nội dung cho mọi ngôn ngữ.
    bs: kích thước batch (mặc định EVAL_BS). Kết quả không phụ thuộc bs (mỗi câu được xử lý độc lập)."""
    model.eval()
    bs = bs or EVAL_BS
    L, N, T = ids_all.shape
    K1 = len(model.blocks) + 1
    Z = torch.empty((K1, L, N, model.d), dtype=torch.float16)
    Zp = torch.empty((K1, L, N, model.d), dtype=torch.float16) if pick is not None else None
    Zl = torch.empty((K1, L, N, model.d), dtype=torch.float16) if last else None
    do_mask = t is not None and t > 0
    nd = n_draws if do_mask else 1
    if do_mask:
        U = np.random.default_rng(SEED + int(round(t * 1000))).random((N, nd, T)).astype(np.float32)
    clip = lambda x: torch.nan_to_num(x, nan=0.0, posinf=6e4, neginf=-6e4).clamp(-6e4, 6e4).half().cpu()
    jobs = [(l, s) for l in range(L) for s in range(0, N, bs)]            # (ngôn ngữ, đầu batch)
    for l, s in tqdm(jobs, desc=desc or "trích biểu diễn", unit="batch", leave=False, disable=not progress, dynamic_ncols=True):
        sl = slice(s, min(N, s + bs))
        ids = torch.from_numpy(ids_all[l, sl].astype(np.int64)).to(DEVICE)
        wid = torch.from_numpy(wid_all[l, sl].astype(np.int64)).to(DEVICE)
        acc_s = acc_p = acc_l = None
        for r in range(nd):
            inp = ids
            if do_mask:
                u = torch.from_numpy(U[sl, r]).to(DEVICE).masked_fill(wid < 0, 2.0)
                m = u < t
                rows = (~m.any(1)).nonzero(as_tuple=True)[0]
                if rows.numel():
                    m[rows, u[rows].argmin(1)] = True
                inp = ids.masked_fill(m, MASK)
            _, hs, _ = model(inp)
            sent, wv, lt = _pool_words(hs, wid)
            acc_s = sent if acc_s is None else acc_s + sent
            acc_l = lt if acc_l is None else acc_l + lt
            if pick is not None:
                pk = torch.from_numpy(pick[sl].astype(np.int64)).to(DEVICE).clamp_min(0)
                pv = wv[:, torch.arange(wv.shape[1], device=DEVICE), pk.clamp_max(wv.shape[2] - 1)]
                acc_p = pv if acc_p is None else acc_p + pv
        Z[:, l, sl] = clip(acc_s / nd)
        if pick is not None:
            Zp[:, l, sl] = clip(acc_p / nd)
        if last:
            Zl[:, l, sl] = clip(acc_l / nd)
    return (Z, Zp, Zl) if last else (Z, Zp)
