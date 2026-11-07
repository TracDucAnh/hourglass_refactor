# Số đo M1-M2 theo tầng bằng numpy: D, G, CKA, A (Procrustes-CV), đặc trưng eo (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
# A của mô hình chung (đo khi chạy) và A_indep giữa hai twin (đo khi phân tích) dùng CÙNG hàm procrustes_score và
# cùng cách chia nửa huấn luyện / nửa kiểm tra, nên ΔA = A_joint - A_indep không lệch vì khác thủ tục.

def split_halves(N):
    perm = np.random.default_rng(SEED).permutation(N)
    return np.sort(perm[: N // 2]), np.sort(perm[N // 2:])

def std_joint(Zk):
    """Zk [L,N,d]: chuẩn hoá từng chiều trên (L,N) rồi trừ tâm của từng ngôn ngữ (như Study 1, 2a)."""
    Z = (Zk - Zk.mean((0, 1), keepdims=True)) / Zk.std((0, 1), keepdims=True).clip(1e-6)
    return Z - Z.mean(1, keepdims=True)

def lang_std(z):
    """z [N,d] của MỘT ngôn ngữ (hoặc một twin): chuẩn hoá từng chiều trong chính không gian đó."""
    z = z.astype(np.float64)
    return (z - z.mean(0)) / z.std(0).clip(1e-6)

def pca_unit(z, tr, q=None):
    """Chiếu lên q thành phần chính (ước lượng trên nửa huấn luyện), chuẩn hoá về độ lớn RMS = 1."""
    q = min(q or D_PCA, len(tr) - 1, z.shape[1])
    X = z[tr] - z[tr].mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    P = z @ Vt[:q].T
    return P / (np.sqrt((P[tr] ** 2).sum(1).mean()) + 1e-8)

def procrustes_score(Pa, Pb, tr, te):
    """A = 1 - lỗi tương đối khi quay Pa khớp Pb (quay ước lượng trên tr, chấm trên te)."""
    U, _, Vt = np.linalg.svd(Pa[tr].T @ Pb[tr])
    err = ((Pb[te] - Pa[te] @ (U @ Vt)) ** 2).sum() / max((Pb[te] ** 2).sum(), 1e-12)
    return float(1.0 - err)

def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((np.linalg.norm(X.T @ Y) ** 2) / max(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y), 1e-12))

def pair_metrics(Z):
    """Z [K1,L,N,d] -> {D, G, CKA, A: [cặp, K1]} và danh sách cặp. G = 1 - khoảng cách thật / kỳ vọng khi ghép ngẫu nhiên."""
    K1, L, N, d = Z.shape
    pairs = list(itertools.combinations(range(L), 2))
    tr, te = split_halves(N)
    out = {m: np.full((len(pairs), K1), np.nan, np.float32) for m in ("D", "G", "CKA", "A")}
    for k in range(K1):
        Zk = std_joint(Z[k].astype(np.float64))
        n2 = (Zk ** 2).sum(-1); s2 = n2.mean()
        if s2 < 1e-12:
            continue
        P = [pca_unit(lang_std(Z[k, l]), tr) for l in range(L)]
        for pi, (a, b) in enumerate(pairs):
            dr = ((Zk[a] - Zk[b]) ** 2).sum(-1).mean()
            out["D"][pi, k] = dr / s2
            out["G"][pi, k] = 1.0 - dr / (n2[a].mean() + n2[b].mean())
            out["CKA"][pi, k] = linear_cka(Zk[a], Zk[b])
            out["A"][pi, k] = procrustes_score(P[a], P[b], tr, te)
    return out, pairs

def twin_A(Za, Zb):
    """A_indep[k] giữa hai mô hình độc lập: Za, Zb [K1,N,d] (cùng nội dung, mỗi mô hình một ngôn ngữ)."""
    tr, te = split_halves(Za.shape[1])
    return np.array([procrustes_score(pca_unit(lang_std(Za[k]), tr), pca_unit(lang_std(Zb[k]), tr), tr, te)
                     for k in range(Za.shape[0])], np.float32)

def curve_features(G):
    """G [K1] (trung bình qua cặp) -> k*, rho*, S_edge (đỉnh trừ đầu mút cao hơn), w (độ rộng), G*, đỉnh nằm trong thân?"""
    G = np.nan_to_num(np.asarray(G, np.float64))
    K = len(G) - 1; k = int(G.argmax())
    S = G[k] - 0.5 * (G[0] + G[-1])
    return dict(k_star=k, rho_star=k / K, S_edge=float(G[k] - max(G[0], G[-1])), w=float((G >= G[k] - 0.25 * S).mean()),
                G_peak=float(G[k]), G_first=float(G[0]), G_last=float(G[-1]), interior=bool(0 < k < K))
