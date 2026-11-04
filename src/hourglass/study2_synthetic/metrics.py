# Số đo M1–M4 (Cell B4)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 23 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell B4 — Số đo M1–M4 cho một buffer Z [K1, L, N, d]
#   M1: D_k, G_k (kỳ vọng chính xác), g theo từng nội dung (cho bootstrap)    M2: CKA không chệch, Procrustes-CV A_k
#   M4: usable information U_k (probe tuyến tính và MLP, chia nhóm theo nội dung)
#   S3: G theo bin hạng danh từ (rank_alignment)
# ============================================================
METRIC_VERSION = 3   # v3: họ causal (decoder, MoE) đo ở token nội dung cuối; S3 lưu ở khoá G_rank_w (đường cơ sở trong bin). v2: probe U giữ lại tâm từng ngôn ngữ. v1 trừ mất tâm -> U ≈ 0 (lỗi đo).

def _standardize_keep_lang(Z):
    """Z [L,N,d]: chuẩn hóa từng chiều trên (L,N) nhưng KHÔNG trừ trung bình từng ngôn ngữ.
    Dùng cho probe U: dấu hiệu 'đây là ngôn ngữ nào' nằm chủ yếu ở tâm của mỗi ngôn ngữ; trừ tâm đi thì probe tuyến tính mù."""
    return (Z - Z.mean((0, 1), keepdim=True)) / Z.std(dim=(0, 1), keepdim=True, correction=0).clamp_min(1e-6)

def _standardize(Z):
    """Z [L,N,d]: chuẩn hóa từng chiều trên (L,N), rồi trừ trung bình từng ngôn ngữ (σ² như paper)."""
    Z = (Z - Z.mean((0, 1), keepdim=True)) / Z.std(dim=(0, 1), keepdim=True, correction=0).clamp_min(1e-6)
    return Z - Z.mean(1, keepdim=True)

def _probe_split(Zt, seed):
    L, N, H = Zt.shape
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(N, generator=g).to(Zt.device)
    te = perm[: N // 2]; rest = perm[N // 2:]
    n_va = max(1, len(rest) // 5); va, trn = rest[:n_va], rest[n_va:]
    flat = lambda c: (Zt[:, c].reshape(-1, H), torch.arange(L, device=Zt.device).repeat_interleave(len(c)))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = flat(trn), flat(va), flat(te)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    return [(x - mu) / sd for x in (Xtr, Xva, Xte)], (ytr, yva, yte), L

def usable_info_lin(Zt, seed, lams=(1e-4, 1e-3, 1e-2, 1e-1), iters=60):
    """Zt [L,N,d] đã trừ trung bình theo nội dung. Probe softmax đoán ngôn ngữ; U = log|L| - CE_test (nats)."""
    (Xtr, Xva, Xte), (ytr, yva, yte), L = _probe_split(Zt, seed)
    H, best = Xtr.shape[1], None
    with torch.enable_grad():
        for lam in lams:
            W_ = torch.zeros(H, L, device=Zt.device, requires_grad=True); b_ = torch.zeros(L, device=Zt.device, requires_grad=True)
            opt = torch.optim.LBFGS([W_, b_], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe")
            def closure():
                opt.zero_grad()
                loss = F.cross_entropy(Xtr @ W_ + b_, ytr) + 0.5 * lam * W_.square().sum()
                loss.backward(); return loss
            opt.step(closure)
            with torch.no_grad():
                ce_va = F.cross_entropy(Xva @ W_ + b_, yva).item()
                if best is None or ce_va < best[0]:
                    lo = Xte @ W_ + b_
                    best = (ce_va, F.cross_entropy(lo, yte).item(), (lo.argmax(1) == yte).float().mean().item())
    return math.log(L) - best[1], best[2]

def usable_info_mlp(Zt, seed, hidden=128, steps=150):
    (Xtr, Xva, Xte), (ytr, yva, yte), L = _probe_split(Zt, seed)
    torch.manual_seed(int(seed))
    net = nn.Sequential(nn.Linear(Xtr.shape[1], hidden), nn.GELU(), nn.Linear(hidden, L)).to(Zt.device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    best = None
    with torch.enable_grad():
        for it in range(steps):
            opt.zero_grad(); F.cross_entropy(net(Xtr), ytr).backward(); opt.step()
            if it % 10 == 9:
                with torch.no_grad():
                    cv = F.cross_entropy(net(Xva), yva).item()
                    if best is None or cv < best[0]:
                        best = (cv, F.cross_entropy(net(Xte), yte).item())
    return math.log(L) - best[1]

@torch.no_grad()
def compute_metrics(Z, seed=SEED, probes=True, desc="số đo theo tầng", progress=True):
    K1, L, N, d = Z.shape
    pairs = list(itertools.combinations(range(L), 2)); P = len(pairs)
    nan = lambda *s: np.full(s, np.nan, np.float32)
    G, D, CKA, A = nan(P, K1), nan(P, K1), nan(P, K1), nan(P, K1)
    g_content = nan(K1, N); sig2 = np.zeros(K1, np.float32)
    U_lin, U_mlp, U_acc = nan(K1), nan(K1), nan(K1)
    perm0 = np.random.default_rng(seed).permutation(N)
    tr = torch.tensor(np.sort(perm0[: N // 2]), device=DEVICE); te = torch.tensor(np.sort(perm0[N // 2:]), device=DEVICE)
    d_pca = int(min(D_PCA, len(tr) - 1, d))
    for k in tqdm(range(K1), desc=desc, unit="tầng", leave=False, disable=not progress, dynamic_ncols=True):
        Zraw = Z[k].to(DEVICE, torch.float32)
        Zk = _standardize(Zraw)          # D, G, CKA, A: vẫn trừ tâm từng ngôn ngữ (σ² như paper)
        nrm2 = Zk.square().sum(-1)
        s2 = float(nrm2.mean()); sig2[k] = s2
        if s2 <= 1e-12:
            continue
        Kg = torch.bmm(Zk, Zk.transpose(1, 2)); Kg.diagonal(dim1=1, dim2=2).zero_()
        ksum, krow = Kg.sum((1, 2)), Kg.sum(2)
        def hsic(a, b):
            t1 = (Kg[a] * Kg[b]).sum()
            t2 = ksum[a] * ksum[b] / ((N - 1) * (N - 2))
            t3 = 2.0 / (N - 2) * (krow[a] * krow[b]).sum()
            return (t1 + t2 - t3) / (N * (N - 3))
        self_h = [hsic(l, l) for l in range(L)]
        torch.manual_seed(seed + k)
        Pp = []
        for l in range(L):
            Xtr = Zk[l][tr] - Zk[l][tr].mean(0, keepdim=True)
            _, _, V = torch.pca_lowrank(Xtr, q=d_pca, center=False, niter=3)
            Pl = Zk[l] @ V[:, :d_pca]
            Pp.append(Pl / (Pl[tr].norm() / math.sqrt(len(tr))).clamp_min(1e-8))
        Pp = torch.stack(Pp)
        gsum = torch.zeros(N, device=DEVICE)
        for pi, (a, b) in enumerate(pairs):
            d_real = (Zk[a] - Zk[b]).square().sum(-1)
            d_shuf = nrm2[a].mean() + nrm2[b].mean()          # = E_π ||za_i - zb_π(i)||^2 chính xác (trung bình ngôn ngữ = 0)
            D[pi, k] = float(d_real.mean() / s2)
            G[pi, k] = float(1.0 - d_real.mean() / d_shuf)
            gsum += 1.0 - d_real / d_shuf
            CKA[pi, k] = float(hsic(a, b) / (self_h[a] * self_h[b]).clamp_min(1e-12).sqrt())
            Uu, _, Vh = torch.linalg.svd(Pp[a][tr].T @ Pp[b][tr])
            err = (Pp[b][te] - Pp[a][te] @ (Uu @ Vh)).square().sum() / Pp[b][te].square().sum().clamp_min(1e-12)
            A[pi, k] = float(1.0 - err)
        g_content[k] = (gsum / P).cpu().numpy()
        del Kg, Pp
        if probes:
            Zt = _standardize_keep_lang(Zraw)            # U: giữ tâm từng ngôn ngữ ...
            Zt = Zt - Zt.mean(0, keepdim=True)           # ... nhưng bỏ phần nội dung (trung bình theo nội dung qua các ngôn ngữ)
            r = [usable_info_lin(Zt, seed + si) for si in range(PROBE_SEEDS)]
            U_lin[k] = float(np.mean([x[0] for x in r])); U_acc[k] = float(np.mean([x[1] for x in r]))
            U_mlp[k] = float(np.mean([usable_info_mlp(Zt, seed + si) for si in range(PROBE_SEEDS)]))
    return dict(G=G, D=D, CKA=CKA, A=A, g_content=g_content.astype(np.float16), sig2=sig2,
                U_lin=U_lin, U_mlp=U_mlp, U_acc=U_acc,
                pair_a=np.array([p[0] for p in pairs]), pair_b=np.array([p[1] for p in pairs]))

@torch.no_grad()
def rank_alignment(Zn, ranks):
    """S3: G theo bin hạng danh từ. Zn [K1,L,Np,d]. G_bin = 1 - mean_bin(d_real) / E_pi[d], pi = hoán vị ngẫu nhiên TRONG bin
    (kỳ vọng chính xác: mean_bin|za|^2 + mean_bin|zc|^2 - 2 mu_a.mu_c), trung bình qua cặp ngôn ngữ. Đường cơ sở trong bin tránh G
    âm giả do các bin khác nhau về độ dài từ / độ lớn vector (bản cũ chia cho đường cơ sở của toàn tập dò). Bin < 2 mẫu: NaN."""
    K1, L, Np, d = Zn.shape
    pairs = list(itertools.combinations(range(L), 2))
    b = torch.from_numpy(np.clip(np.floor(np.log2(ranks)).astype(np.int64), 0, N_RANK_BINS - 1)).to(DEVICE)
    oh = F.one_hot(b, N_RANK_BINS).float()                      # [Np, B]
    cnt = oh.sum(0)
    out = np.full((K1, N_RANK_BINS), np.nan, np.float32)
    for k in range(K1):
        Zk = _standardize(Zn[k].to(DEVICE, torch.float32))
        nrm2 = Zk.square().sum(-1)                              # [L, Np]
        if float(nrm2.mean()) <= 1e-12:
            continue
        sq = nrm2 @ oh / cnt.clamp_min(1)                         # [L, B]   mean_bin |z|^2
        mu = torch.einsum("lnd,nb->lbd", Zk, oh) / cnt.clamp_min(1)[None, :, None]
        acc = torch.zeros(N_RANK_BINS, device=DEVICE)
        for a, c in pairs:
            real = ((Zk[a] - Zk[c]).square().sum(-1) @ oh) / cnt.clamp_min(1)
            base = sq[a] + sq[c] - 2 * (mu[a] * mu[c]).sum(-1)
            acc += 1.0 - real / base.clamp_min(1e-12)
        out[k] = (acc / len(pairs)).cpu().numpy()
        out[k][cnt.cpu().numpy() < 2] = np.nan
    return out, cnt.cpu().numpy()
