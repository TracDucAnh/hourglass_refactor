# Truy hồi bản dịch M3 theo tầng (Cell B2)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 20 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell B2 — Truy hồi bản dịch (M3) cho từng từ, theo TẦNG, trên GPU
# Với mỗi cặp (a,b) và mỗi tầng k: ma trận cosine [n,n] giữa các cặp từ -> CSLS -> hạng của bản dịch đúng -> 1/hạng (RR).
#   - ứng viên = mọi cặp từ của cặp ngôn ngữ đó; ứng viên có CÙNG dạng từ đích với đáp án đúng KHÔNG bị tính là đối thủ
#     (nếu không, "the" xuất hiện 500 lần sẽ tự chặn chính nó);
#   - tiền xử lý giống M1/M3 của Study 1: chuẩn hoá từng chiều (chung mọi ngôn ngữ) rồi trừ trung bình từng ngôn ngữ.
# Ngoài RR trên toàn bộ ứng viên (dùng cho phân tích theo từ), còn tính RR trên tập ứng viên CỐ ĐỊNH n=POOL (rút mẫu con, POOL_REPS lần)
# để so sánh GIỮA CÁC CẶP ngôn ngữ (Mantel): RR phụ thuộc kích thước tập ứng viên nên không so thẳng được giữa các cặp.
# ============================================================
def _compact_ids(t):
    u, inv = torch.unique(t, return_inverse=True)
    return inv, int(u.numel())

@torch.no_grad()
def _hub_terms(C, ta, tb, k, mode, chunk=RETR_CHUNK):
    """Số hạng hubness r_T(x), r_S(y) của CSLS.
    "instance": trung bình top-k trên mọi instance (cách cũ). Từ phổ biến có rất nhiều instance gần giống nhau -> trông như hub -> bị phạt oan.
    "type"    : mỗi LOẠI từ đối diện chỉ được đếm MỘT lần (lấy cosine lớn nhất trong loại) rồi mới lấy top-k.
    "cos"     : không hiệu chỉnh (trả về None)."""
    n = C.shape[0]; kk = min(k, n)
    if mode == "cos":
        return None, None
    if mode == "instance":
        return C.topk(kk, dim=1).values.mean(1), C.topk(kk, dim=0).values.mean(0)
    ib, Tb = _compact_ids(tb); ia, Ta = _compact_ids(ta)
    ra = torch.empty(n, device=C.device, dtype=C.dtype); rb = torch.empty(n, device=C.device, dtype=C.dtype)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        Cb = torch.full((e - s, Tb), -2.0, device=C.device, dtype=C.dtype).scatter_reduce(
            1, ib[None, :].expand(e - s, n), C[s:e], reduce="amax", include_self=True)           # [hàng, loại đích]
        ra[s:e] = Cb.topk(min(kk, Tb), dim=1).values.mean(1)
        Ca = torch.full((Ta, e - s), -2.0, device=C.device, dtype=C.dtype).scatter_reduce(
            0, ia[:, None].expand(n, e - s), C[:, s:e], reduce="amax", include_self=True)        # [loại nguồn, cột]
        rb[s:e] = Ca.topk(min(kk, Ta), dim=0).values.mean(0)
        del Cb, Ca
    return ra, rb

@torch.no_grad()
def csls_mrr(C, ta, tb, k=CSLS_K, chunk=RETR_CHUNK, mode=None, acc=None, ia=None, ib=None, which="strict"):
    """C: cosine [n,n] (hàng = từ ở ngôn ngữ a, cột = từ ở b; đáp án đúng nằm trên đường chéo).
    ta, tb: id dạng từ (long [n]). mode mặc định = HUB_MODE.
    which = "strict": ứng viên CÙNG dạng từ đích với đáp án bị che (cách cũ) -> trả (rr_ab, rr_ba).
    which = "valid" : (v5) ứng viên là MỌI bản dịch hợp lệ đã quan sát của từ nguồn bị che (vd. the->le thì la, les không còn là "sai"),
                      cần acc [Ta,Tb] bool (loại nguồn x loại đích; True = cặp đã được gióng hàng + xác nhận trong ITEMS) và ia, ib = id nén của từng hàng trong acc -> trả (rr_ab, rr_ba).
    which = "both"  : trả (rr_ab, rr_ba, rr_ab_valid, rr_ba_valid) từ CÙNG một ma trận S (rẻ hơn chạy hai lần).
    Vì (ta_i, tb_i) luôn nằm trong acc nên mặt nạ "valid" là tập cha của mặt nạ "strict": RR_valid >= RR_strict."""
    mode = mode or HUB_MODE
    n = C.shape[0]
    ra, rb = _hub_terms(C, ta, tb, k, mode)
    S = C if ra is None else 2.0 * C - ra[:, None] - rb[None, :]
    dg = S.diagonal()
    want_s, want_v = which in ("strict", "both"), which in ("valid", "both")
    if want_v and (acc is None or ia is None or ib is None):
        raise ValueError("which='valid'/'both' cần acc, ia, ib (đặt MASK_VALID_TRANSLATIONS=True)")
    rk = {nm: (torch.empty(n, device=C.device), torch.empty(n, device=C.device)) for nm, w in (("s", want_s), ("v", want_v)) if w}
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        gt_ab = S[s:e] > dg[s:e, None]                      # a_i truy hồi trong tập b_j
        gt_ba = S[:, s:e] > dg[None, s:e]                   # b_j truy hồi trong tập a_i
        if want_s:
            rk["s"][0][s:e] = 1 + (gt_ab & (tb[s:e, None] != tb[None, :])).sum(1)
            rk["s"][1][s:e] = 1 + (gt_ba & (ta[:, None] != ta[None, s:e])).sum(0)
        if want_v:
            rk["v"][0][s:e] = 1 + (gt_ab & ~acc[ia[s:e]][:, ib]).sum(1)        # acc[ia_i, ib_j]  [e-s, n]
            rk["v"][1][s:e] = 1 + (gt_ba & ~acc[:, ib[s:e]][ia]).sum(0)        # acc[ia_i, ib_j]  [n, e-s]
        del gt_ab, gt_ba
    out = []
    for nm in ("s", "v"):
        if nm in rk:
            out += [1.0 / rk[nm][0], 1.0 / rk[nm][1]]
    return tuple(out)

def type_ids(wt):
    """id số của dạng từ trong một ngôn ngữ (theo bảng từ devtest)."""
    _, inv = np.unique(np.asarray(wt["norm"], dtype=object).astype(str), return_inverse=True)
    return inv.astype(np.int64)

def cache_path(key, lang):
    return f"{WORK_DIR}/cache/{key}/{SHORT[lang]}.npy"

def pair_rows(items, plans, valid_only=True):
    """Chỉ số hàng buffer của từng cặp từ (a,b) + mặt nạ hợp lệ (cả hai từ có sub-token và được trích)."""
    out = []
    for pi, (ia, ib) in enumerate(PAIRS):
        s, e = items["ptr"][pi], items["ptr"][pi + 1]
        ra = plans[LANGS[ia]]["row_of_g"][items["wa"][s:e]]
        rb = plans[LANGS[ib]]["row_of_g"][items["wb"][s:e]]
        out.append((s, e, ra, rb, (ra >= 0) & (rb >= 0)))
    return out

@torch.no_grad()
def compute_retrieval(key, items, plans, tid):
    """Đọc cache từng tầng, tính RR cho mọi cặp ngôn ngữ. Trả về dict mảng (xem cuối hàm)."""
    mms = [np.load(cache_path(key, l), mmap_mode="r") for l in LANGS]
    K1 = mms[0].shape[0]
    prow = pair_rows(items, plans)
    nv = [int(v.sum()) for *_, v in prow]
    big = [x for x in nv if x >= MIN_PAIR_ITEMS]                 # cặp quá nhỏ (Dice) không được kéo n_pool xuống
    n_pool = int(min(POOL_N, min(big) if big else min(nv)))
    if n_pool < 100:
        print(f"  [warn] n_pool={n_pool} quá nhỏ: có cặp ngôn ngữ chỉ còn {min(nv)} cặp từ hợp lệ")
    off = np.concatenate([[0], np.cumsum(nv)])
    RR_AB = np.zeros((K1, off[-1]), np.float16); RR_BA = np.zeros((K1, off[-1]), np.float16)
    RR_ABV = np.zeros((K1, off[-1]), np.float16) if MASK_VALID_TRANSLATIONS else None; RR_BAV = np.zeros((K1, off[-1]), np.float16) if MASK_VALID_TRANSLATIONS else None
    POOL_R = np.zeros((len(PAIRS), K1), np.float32)
    # dữ liệu theo cặp (cố định qua các tầng)
    pd_ = []
    for pi, (ia, ib) in enumerate(PAIRS):
        s, e, ra, rb, v = prow[pi]
        g = torch.Generator().manual_seed(SEED + pi)
        sub = [torch.randperm(nv[pi], generator=g)[:n_pool].to(DEVICE) for _ in range(POOL_REPS)]
        ga, gb = items["wa"][s:e][v], items["wb"][s:e][v]
        ta_np, tb_np = tid[LANGS[ia]][ga], tid[LANGS[ib]][gb]
        acc = ia_c = ib_c = None
        if MASK_VALID_TRANSLATIONS:          # (v5) acc[loại nguồn, loại đích] = True nếu cặp đó đã được gióng hàng + xác nhận ở BẤT KỲ đâu trong ITEMS của cặp ngôn ngữ này
            _, ia_n = np.unique(ta_np, return_inverse=True); _, ib_n = np.unique(tb_np, return_inverse=True)
            ia_n, ib_n = ia_n.astype(np.int64).reshape(-1), ib_n.astype(np.int64).reshape(-1)
            acc = torch.zeros((int(ia_n.max()) + 1, int(ib_n.max()) + 1), dtype=torch.bool)
            acc[torch.from_numpy(ia_n), torch.from_numpy(ib_n)] = True
            acc, ia_c, ib_c = acc.to(DEVICE), torch.from_numpy(ia_n).to(DEVICE), torch.from_numpy(ib_n).to(DEVICE)
        pd_.append(dict(ra=torch.from_numpy(ra[v]).to(DEVICE), rb=torch.from_numpy(rb[v]).to(DEVICE),
                        ta=torch.from_numpy(ta_np).to(DEVICE), tb=torch.from_numpy(tb_np).to(DEVICE), sub=sub, acc=acc, ia=ia_c, ib=ib_c))
    t0 = time.time()
    for k in range(K1):
        Xs = [torch.from_numpy(np.array(m[k])).to(DEVICE).float() for m in mms]     # mặt phẳng tầng k liền khối
        n_all = sum(x.shape[0] for x in Xs)
        mu = sum(x.sum(0) for x in Xs) / n_all
        sd = (sum(((x - mu) ** 2).sum(0) for x in Xs) / n_all).sqrt().clamp_min(1e-6)
        Xs = [(x - mu) / sd for x in Xs]
        Xs = [x - x.mean(0, keepdim=True) for x in Xs]
        for pi, (ia, ib) in enumerate(PAIRS):
            d = pd_[pi]
            Xa = F.normalize(Xs[ia][d["ra"]], dim=1); Xb = F.normalize(Xs[ib][d["rb"]], dim=1)
            C = Xa @ Xb.T
            if d["acc"] is None:
                rab, rba = csls_mrr(C, d["ta"], d["tb"])
            else:
                rab, rba, rabv, rbav = csls_mrr(C, d["ta"], d["tb"], acc=d["acc"], ia=d["ia"], ib=d["ib"], which="both")
                RR_ABV[k, off[pi]:off[pi + 1]] = rabv.cpu().numpy().astype(np.float16)
                RR_BAV[k, off[pi]:off[pi + 1]] = rbav.cpu().numpy().astype(np.float16)
            RR_AB[k, off[pi]:off[pi + 1]] = rab.cpu().numpy().astype(np.float16)
            RR_BA[k, off[pi]:off[pi + 1]] = rba.cpu().numpy().astype(np.float16)
            s_acc = 0.0
            for idx in d["sub"]:               # R_pool (Mantel) dùng cùng biến thể RR_VARIANT với phân tích chính
                if d["acc"] is not None and RR_VARIANT == "valid":
                    r1, r2 = csls_mrr(C[idx][:, idx], d["ta"][idx], d["tb"][idx], acc=d["acc"], ia=d["ia"][idx], ib=d["ib"][idx], which="valid")
                else:
                    r1, r2 = csls_mrr(C[idx][:, idx], d["ta"][idx], d["tb"][idx])
                s_acc += 0.5 * float(r1.mean() + r2.mean())
            POOL_R[pi, k] = s_acc / len(d["sub"])
            del C, Xa, Xb
        del Xs
        if k % 6 == 0 or k == K1 - 1:
            print(f"    tầng {k + 1}/{K1}: R_pool trung bình = {POOL_R[:, k].mean():.3f}  ({time.time() - t0:.0f}s)", flush=True)
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
    valid_idx = np.concatenate([np.arange(s, e)[v] for s, e, _, _, v in prow]) if off[-1] else np.zeros(0, np.int64)
    res = dict(RR_AB=RR_AB, RR_BA=RR_BA, POOL_R=POOL_R, valid_idx=valid_idx, pair_ptr=off, n_pool=np.int64(n_pool), K1=np.int64(K1))
    if MASK_VALID_TRANSLATIONS:
        res.update(RR_AB_V=RR_ABV, RR_BA_V=RR_BAV)
    return res
