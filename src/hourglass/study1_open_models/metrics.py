# Số đo M1–M4 + J_k (Cell B3)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 6 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell B3 — Số đo M1–M4 cho một buffer [K1, n_lang, N, H]
#   M1: D_k, G_k (+ g theo từng content cho bootstrap)   M2: CKA không chệch, Procrustes-CV A_k
#   M3: MRR truy hồi CSLS R_k                            M4: usable information U_k (probe tuyến tính)
# ============================================================
def usable_info(Zt, seed, lams=(1e-4, 1e-3, 1e-2, 1e-1), iters=60):
    """Zt: [L,N,H] đã trừ trung bình theo content. Probe softmax đoán ngôn ngữ; chia train/val/test theo content.
    U = log|L| - CE_test (nats), chọn lambda (L2) theo CE trên val."""
    L, N, H = Zt.shape
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g).to(Zt.device)
    te = perm[: N // 2]; rest = perm[N // 2:]
    n_va = max(1, len(rest) // 5); va, trn = rest[:n_va], rest[n_va:]
    def flat(c):
        return Zt[:, c].reshape(-1, H), torch.arange(L, device=Zt.device).repeat_interleave(len(c))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = flat(trn), flat(va), flat(te)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr, Xva, Xte = [(x - mu) / sd for x in (Xtr, Xva, Xte)]
    best = None
    with torch.enable_grad():
        for lam in lams:
            W = torch.zeros(H, L, device=Zt.device, requires_grad=True)
            b = torch.zeros(L, device=Zt.device, requires_grad=True)
            opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(Xtr @ W + b, ytr) + 0.5 * lam * W.square().sum()
                loss.backward()
                return loss
            opt.step(closure)
            with torch.no_grad():
                ce_va = F.cross_entropy(Xva @ W + b, yva).item()
                if best is None or ce_va < best[0]:
                    lo = Xte @ W + b
                    best = (ce_va, F.cross_entropy(lo, yte).item(), (lo.argmax(1) == yte).float().mean().item())
    return math.log(L) - best[1], best[2]

@torch.no_grad()
def compute_metrics(buf, n_shuffles=N_SHUFFLES, seed=SEED):
    K1, nL, N, H = buf.shape
    pairs = list(itertools.combinations(range(nL), 2)); P = len(pairs)
    nan = lambda *s: np.full(s, np.nan, np.float32)
    G, D, Dpi, CKA, A, R = (nan(P, K1) for _ in range(6))
    g_content = nan(P, K1, N)
    rr_content = np.full((P, K1, N), np.nan, np.float16)
    sig2 = np.zeros(K1, np.float32)
    U, Uacc = nan(PROBE_SEEDS, K1), nan(PROBE_SEEDS, K1)

    perm0 = np.random.default_rng(seed).permutation(N)                       # chia nửa content cho Procrustes-CV
    tr = torch.tensor(np.sort(perm0[: N // 2]), device=DEVICE); te = torch.tensor(np.sort(perm0[N // 2:]), device=DEVICE)
    d_pca = int(min(D_PCA, len(tr) - 1, H))
    idxN = torch.arange(N, device=DEVICE)
    gen = torch.Generator(device=DEVICE)

    for k in range(K1):
        Z = buf[k].to(DEVICE, torch.float32)                                  # [L,N,H]
        Z = (Z - Z.mean((0, 1), keepdim=True)) / Z.std(dim=(0, 1), keepdim=True, correction=0).clamp_min(1e-6)
        Z = Z - Z.mean(1, keepdim=True)                                       # trừ trung bình từng ngôn ngữ (sigma^2 như paper)
        nrm2 = Z.square().sum(-1)                                             # [L,N]
        s2 = float(nrm2.mean()); sig2[k] = s2
        if s2 <= 1e-12:
            continue

        # ---- CKA không chệch (HSIC unbiased, kernel tuyến tính) ----
        Kg = torch.bmm(Z, Z.transpose(1, 2))
        Kg.diagonal(dim1=1, dim2=2).zero_()
        ksum, krow = Kg.sum((1, 2)), Kg.sum(2)
        def hsic(a, b):
            t1 = (Kg[a] * Kg[b]).sum()
            t2 = ksum[a] * ksum[b] / ((N - 1) * (N - 2))
            t3 = 2.0 / (N - 2) * (krow[a] * krow[b]).sum()
            return (t1 + t2 - t3) / (N * (N - 3))
        self_h = [hsic(l, l) for l in range(nL)]

        # ---- PCA mỗi ngôn ngữ (fit trên nửa train) cho Procrustes-CV ----
        torch.manual_seed(seed + k)
        Pp = []
        for l in range(nL):
            Xtr = Z[l][tr] - Z[l][tr].mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(Xtr, q=d_pca, center=False, niter=3)
            Pl = Z[l] @ V[:, :d_pca]
            Pp.append(Pl / (Pl[tr].norm() / math.sqrt(len(tr))).clamp_min(1e-8))
        Pp = torch.stack(Pp)

        for pi, (a, b) in enumerate(pairs):
            Za, Zb = Z[a], Z[b]
            Gm = Za @ Zb.T                                                    # [N,N]
            # M1: khoảng cách cùng content vs. khoảng cách khi ghép ngẫu nhiên (1000 hoán vị, qua ma trận Gram)
            d_real = (Za - Zb).square().sum(-1)
            gen.manual_seed(seed + 7919 * (pi * K1 + k))
            acc, done = torch.zeros((), device=DEVICE), 0
            while done < n_shuffles:
                m = min(SHUF_CHUNK, n_shuffles - done)
                perm = torch.argsort(torch.rand(m, N, device=DEVICE, generator=gen), dim=1)
                d_perm = nrm2[a][None, :] + nrm2[b][perm] - 2.0 * Gm[idxN[None, :], perm]
                acc += d_perm.mean(1).sum(); done += m
            d_shuf = float(acc) / n_shuffles
            if d_shuf > 1e-12:
                D[pi, k] = float(d_real.mean()) / s2
                Dpi[pi, k] = d_shuf / s2
                G[pi, k] = 1.0 - D[pi, k] / Dpi[pi, k]
                g_content[pi, k] = (1.0 - d_real / d_shuf).cpu().numpy()
            # M2: CKA và Procrustes-CV
            CKA[pi, k] = float(hsic(a, b) / (self_h[a] * self_h[b]).clamp_min(1e-12).sqrt())
            Uu, _, Vh = torch.linalg.svd(Pp[a][tr].T @ Pp[b][tr])
            Rm = Uu @ Vh
            err = (Pp[b][te] - Pp[a][te] @ Rm).square().sum() / Pp[b][te].square().sum().clamp_min(1e-12)
            A[pi, k] = float(1.0 - err)
            # M3: MRR truy hồi bản dịch bằng CSLS (trung bình hai chiều a->b, b->a)
            C = Gm / (nrm2[a].sqrt().clamp_min(1e-8)[:, None] * nrm2[b].sqrt().clamp_min(1e-8)[None, :])
            kk = min(CSLS_K, N)
            Sc = 2 * C - C.topk(kk, dim=1).values.mean(1)[:, None] - C.topk(kk, dim=0).values.mean(0)[None, :]
            dg = Sc.diagonal()
            rank_ab = 1 + (Sc > dg[:, None]).sum(1); rank_ba = 1 + (Sc > dg[None, :]).sum(0)
            rr = 0.5 * (1.0 / rank_ab + 1.0 / rank_ba)
            R[pi, k] = float(rr.mean()); rr_content[pi, k] = rr.cpu().numpy().astype(np.float16)
            del Gm, C, Sc

        # ---- M4: usable information về ngôn ngữ sau khi trừ phần content giải thích được ----
        Zt = Z - Z.mean(0, keepdim=True)
        for si in range(PROBE_SEEDS):
            U[si, k], Uacc[si, k] = usable_info(Zt, seed + si)
        del Z, Kg, Pp, Zt
        if k % 8 == 0 or k == K1 - 1:
            print(f"    layer {k + 1}/{K1}  G_mean={np.nanmean(G[:, k]):.3f}  CKA={np.nanmean(CKA[:, k]):.3f}")
            torch.cuda.empty_cache()
    pi_a = np.array([p[0] for p in pairs]); pi_b = np.array([p[1] for p in pairs])
    return dict(G=G, D=D, Dpi=Dpi, CKA=CKA, A=A, R=R, sig2=sig2, U=U, Uacc=Uacc,
                g_content=g_content, rr_content=rr_content, pair_a=pi_a, pair_b=pi_b)

@torch.no_grad()
def router_js(rbuf, langs, eng="eng_Latn"):
    """J_k theo từng content. Trả về:
       JS     : entropy-normalised JS (D_H-JS, Bandarkar et al. A.4), thuộc [0,1]  -> dùng để vẽ
       JS_raw : JS thô (nats), giữ để đối chiếu."""
    ie = langs.index(eng); others = [i for i in range(len(langs)) if i != ie]
    nL, Lm, N, E = rbuf.shape
    logE = math.log(E)
    JS     = np.zeros((len(others), Lm, N), np.float32)
    JS_raw = np.zeros_like(JS)
    for l in range(Lm):
        Pm = rbuf[:, l].to(DEVICE).float().clamp_min(1e-12)
        Pm = Pm / Pm.sum(-1, keepdim=True)                      # [nL, N, E]
        H  = -(Pm * Pm.log()).sum(-1)                           # [nL, N] entropy mỗi câu
        pe = Pm[ie]
        for oi, i in enumerate(others):
            p = Pm[i]; mid = 0.5 * (p + pe)
            js = 0.5 * (p * (p / mid).log()).sum(-1) + 0.5 * (pe * (pe / mid).log()).sum(-1)
            F_ = (logE - 0.5 * (H[i] + H[ie])).clamp_min(1e-8)  # F = log E - H_avg
            JS_raw[oi, l] = js.cpu().numpy()
            JS[oi, l]     = (js / F_).cpu().numpy()
    return JS, JS_raw, np.array([langs[i] for i in others]), rbuf.float().mean(2).numpy()
