# GAM P-spline, bootstrap trên GPU, đơn điệu / unimodal, Mantel (Cell C4)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 27 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C4 — Công cụ thống kê: GAM (P-spline), bootstrap theo từ XẾP THEO BATCH TRÊN GPU, mô hình đơn điệu / unimodal, Mantel
#   R_{l,w}(k) = s(log-hạng) + hiệu ứng từ loại (UPOS) + độ dài + log sub-token/từ + hiệu ứng ngôn ngữ + nhiễu
# GAM = B-spline bậc 3 với phạt sai phân bậc 2 (P-spline, Eilers & Marx) giải dạng đóng nên bootstrap 1000 lần x mọi tầng
# chỉ là một phép nhân ma trận W[B,n] @ X⊗X[n,p^2] và một lệnh torch.linalg.solve theo lô.
# ============================================================
from scipy.interpolate import BSpline
from sklearn.isotonic import IsotonicRegression

def bspline_basis(x, lo, hi, n_knots=N_KNOTS, degree=3):
    inner = np.linspace(lo, hi, n_knots + 2)
    t = np.r_[[lo] * degree, inner, [hi] * degree]
    return BSpline.design_matrix(np.clip(x, lo, hi), t, degree).toarray()          # [n, n_knots+4]

def diff_penalty(nb, order=2):
    D = np.diff(np.eye(nb), order, axis=0)
    return D.T @ D

class GamDesign:
    """Ma trận thiết kế: [spline(log-hạng) | UPOS | len_z | nsub_z | pmix_z | trans_z | ngôn ngữ]. Không có intercept riêng (spline đã cộng thành 1).
    v5: từng nhóm covariate bật/tắt riêng (use_len, use_nsub, use_pmix, use_trans) để chạy GAM không hiệu chỉnh (chỉ spline) so với chấm thô;
    thêm trans_z = z-score entropy bản dịch (đa dạng bản dịch của từ)."""
    def __init__(self, df, use_upos=True, use_lang=True, ref_upos="NOUN", n_knots=N_KNOTS, use_len=True, use_nsub=True, use_pmix=None, use_trans=None):
        x = df.log_rank.values
        self.lo, self.hi = np.quantile(x, GAM_QUANT[0]), np.quantile(x, GAM_QUANT[1])
        self.n_knots = n_knots
        self.upos_lv = [u for u in sorted(df.upos.unique()) if u != ref_upos] if use_upos else []
        self.lang_lv = sorted(df.lang.unique(), key=lambda s: [SHORT[l] for l in LANGS].index(s))[1:] if use_lang else []
        self.use_len, self.use_nsub = bool(use_len), bool(use_nsub)
        self.len_mu, self.len_sd = df.len.mean(), max(df.len.std(), 1e-6)
        ln = np.log(df.nsub.clip(lower=1)); self.ns_mu, self.ns_sd = ln.mean(), max(ln.std(), 1e-6)
        self.ref_upos = ref_upos
        want_pm = USE_PMIX if use_pmix is None else use_pmix
        self.use_pmix = bool(want_pm and "pmix" in df.columns and df.pmix.std() > 1e-9)
        if self.use_pmix:
            self.pm_mu, self.pm_sd = df.pmix.mean(), max(df.pmix.std(), 1e-6)
        want_tr = USE_TRANS_ENT if use_trans is None else use_trans
        self.use_trans = bool(want_tr and "trans_ent" in df.columns and df.trans_ent.std() > 1e-9)
        if self.use_trans:
            self.tr_mu, self.tr_sd = df.trans_ent.mean(), max(df.trans_ent.std(), 1e-6)
        self.names = ([f"upos:{u}" for u in self.upos_lv] + (["len_z"] if self.use_len else []) + (["nsub_z"] if self.use_nsub else [])
                      + (["pmix_z"] if self.use_pmix else []) + (["trans_z"] if self.use_trans else []) + [f"lang:{l}" for l in self.lang_lv])
    def par(self, df):
        cols = [(df.upos.values == u).astype(float) for u in self.upos_lv]
        if self.use_len:
            cols.append((df.len.values - self.len_mu) / self.len_sd)
        if self.use_nsub:
            cols.append((np.log(df.nsub.clip(lower=1).values) - self.ns_mu) / self.ns_sd)
        if self.use_pmix:
            cols.append((df.pmix.values - self.pm_mu) / self.pm_sd)
        if self.use_trans:
            cols.append((df.trans_ent.values - self.tr_mu) / self.tr_sd)
        cols += [(df.lang.values == l).astype(float) for l in self.lang_lv]
        return np.stack(cols, 1) if cols else np.zeros((len(df), 0))
    def matrix(self, df):
        S = bspline_basis(df.log_rank.values, self.lo, self.hi, self.n_knots)
        self.nb = S.shape[1]
        return np.concatenate([S, self.par(df)], 1)
    def penalty(self, p):
        P = np.zeros((p, p)); P[:self.nb, :self.nb] = diff_penalty(self.nb)
        return P + 1e-8 * np.eye(p)
    def grid_basis(self, G=200):
        g = np.linspace(self.lo, self.hi, G)
        return g, bspline_basis(g, self.lo, self.hi, self.n_knots)

def gcv_lambda(X, y, P, grid=LAMBDA_GRID):
    """λ theo GCV trên một tầng (thường là tầng waist); dùng chung cho mọi tầng."""
    XtX, Xty, n, p = X.T @ X, X.T @ y, len(y), X.shape[1]
    best = None
    for lam in grid:
        A = XtX + lam * P
        beta = np.linalg.solve(A, Xty)
        tr = np.trace(np.linalg.solve(A, XtX))
        rss = float(((y - X @ beta) ** 2).sum())
        gcv = n * rss / (n - tr) ** 2
        if best is None or gcv < best[0]:
            best = (gcv, float(lam), float(tr))
    return best[1], best[2]

@torch.no_grad()
def gam_bootstrap(X, Y, P, lam, B=N_BOOT, chunk=BOOT_CHUNK, seed=SEED, device=DEVICE, cl=None):
    """X [n,p], Y [n,K] (K tầng). Trả về beta_obs [p,K], beta_boot [B,p,K] (float32).
    cl=None: bootstrap theo từ (mỗi lần rút n dòng có hoàn lại, độc lập) -> trọng số nguyên W[b,n].
    cl=[n] id cụm (v5): bootstrap THEO CỤM — rút lại các cụm khái niệm có hoàn lại và mọi dòng của một cụm vào/ra mẫu cùng nhau
    (cùng một khái niệm xuất hiện ở nhiều ngôn ngữ nên các dòng không độc lập). X'WX = W @ vec(x x'), X'Wy = W @ vec(x y').
    λ giữ cố định (xem LAM_SENS ở cell C5 để kiểm độ nhạy theo λ)."""
    Xt = torch.from_numpy(X).float().to(device); Yt = torch.from_numpy(np.ascontiguousarray(Y)).float().to(device)
    n, p = Xt.shape; K = Yt.shape[1]
    Xo = (Xt[:, :, None] * Xt[:, None, :]).reshape(n, p * p)
    XY = (Xt[:, :, None] * Yt[:, None, :]).reshape(n, p * K)
    Pm = torch.from_numpy(lam * P).to(device).double()
    def solve(W):
        A = (W @ Xo).reshape(-1, p, p).double() + Pm
        R = (W @ XY).reshape(-1, p, K).double()
        return torch.linalg.solve(A, R).float()
    if cl is not None:
        _, cl_c = np.unique(np.asarray(cl), return_inverse=True)
        cl_t = torch.from_numpy(cl_c.astype(np.int64)).to(device); Cn = int(cl_c.max()) + 1
    beta_obs = solve(torch.ones(1, n, device=device))[0]
    g = torch.Generator(device=device).manual_seed(seed)
    out = []
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        if cl is None:
            idx = torch.randint(0, n, (m, n), generator=g, device=device)
            W = torch.zeros(m, n, device=device).scatter_add_(1, idx, torch.ones(m, n, device=device))
        else:
            idx = torch.randint(0, Cn, (m, Cn), generator=g, device=device)
            M = torch.zeros(m, Cn, device=device).scatter_add_(1, idx, torch.ones(m, Cn, device=device))
            W = M[:, cl_t]
        out.append(solve(W).cpu())
    del Xo, XY
    return beta_obs.cpu(), torch.cat(out)

def prep_main(df, R, drop_unseen=None, quant=X_QUANT, drop_hapax=False):
    """Dữ liệu cho GAM: bỏ từ unseen (hạng giả V+1) và cắt đuôi theo phân vị của trục x bằng cách LOẠI BỎ (không clip) -> không có khối điểm dồn ở biên spline.
    drop_hapax (v5): bỏ thêm các từ chỉ xuất hiện 1 lần trong FineWeb-2 (chung một hạng trung bình = một 'khối' dựng đứng ở biên phải)."""
    drop_unseen = DROP_UNSEEN if drop_unseen is None else drop_unseen
    m = df.seen.values.copy() if drop_unseen else np.ones(len(df), bool)
    if drop_hapax and "fw_count" in df.columns:
        m &= (df.fw_count.values >= 2)
    x = df.log_rank.values
    lo, hi = np.quantile(x[m], quant)
    m &= (x >= lo) & (x <= hi)
    return df[m].reset_index(drop=True), R[m]

def classify_shape(p_left_up, p_left_down, p_right_down, p_right_up, a=0.05):
    """Phân loại hình dạng theo hạng từ các p một phía của độ dốc hai đầu (hạng tăng = từ hiếm hơn)."""
    lu, ld, rd, ru = p_left_up < a, p_left_down < a, p_right_down < a, p_right_up < a
    if lu and rd: return "inverted-U"
    if ld and rd: return "decreasing in rank (A2-like)"
    if lu and ru: return "increasing in rank (frequent words less aligned)"
    if ld and ru: return "U-shaped"
    if lu: return "head rises, tail flat"
    if rd: return "tail falls, head flat"
    if ld: return "head falls, tail flat"
    if ru: return "head flat, tail rises"
    return "flat / inconclusive"

def shape_stats(grid, f, head_hi=None, tail_lo=None):
    """f [..., G] -> hình dạng theo log10 hạng.
    slope_left/right = hệ số OLS trên 20% ngoài cùng của LƯỚI x (tiêu chí cũ; lưới đều theo x, KHÔNG theo khối lượng dữ liệu).
    amp = đỉnh − max(trung bình đầu trái, trung bình đầu phải) trên 10% ngoài cùng của lưới: ≈ 0 với đường đơn điệu.
    (v5) Nếu truyền head_hi / tail_lo (phân vị EDGE_Q của dữ liệu): slope_left_q/right_q và amp_q dùng cửa sổ định nghĩa theo PHÂN VỊ DỮ LIỆU
    (đầu = phần x <= head_hi, đuôi = phần x >= tail_lo; tối thiểu 5 điểm lưới) — không phụ thuộc việc lưới kéo dài tới đâu.
    amp_mean_ends = công thức cũ (luôn > 0 kể cả khi đơn điệu) — chỉ để đối chiếu, KHÔNG dùng để kết luận."""
    G = f.shape[-1]; i_e, i_s = max(2, int(0.1 * G)), max(5, int(0.2 * G))
    k = f.argmax(-1)
    gr = torch.as_tensor(grid, dtype=f.dtype)
    def ols(sl):
        xs = gr[sl] - gr[sl].mean()
        return (f[..., sl] * xs).sum(-1) / (xs ** 2).sum()
    left, right = f[..., :i_e].mean(-1), f[..., -i_e:].mean(-1)
    ymax = f.max(-1).values
    out = dict(x_star=gr[k], amp=ymax - torch.maximum(left, right), amp_mean_ends=ymax - 0.5 * (left + right),
               slope_left=ols(slice(0, i_s)), slope_right=ols(slice(G - i_s, G)),
               interior=((k > 0.05 * G) & (k < 0.95 * G)).float(), y_star=ymax)
    if head_hi is None or tail_lo is None:
        out.update(slope_left_q=out["slope_left"], slope_right_q=out["slope_right"], amp_q=out["amp"])
    else:
        gnp = np.asarray(grid)
        nl, nr = max(5, int((gnp <= head_hi).sum())), max(5, int((gnp >= tail_lo).sum()))
        hl, hr = f[..., :nl].mean(-1), f[..., G - nr:].mean(-1)
        out.update(slope_left_q=ols(slice(0, nl)), slope_right_q=ols(slice(G - nr, G)), amp_q=ymax - torch.maximum(hl, hr))
    return out

def gam_analysis(df, R, layers=None, use_upos=True, use_lang=True, B=N_BOOT, seed=SEED, lam=None, G=200, k_ref=None, covars="full", cluster=None):
    """Khớp GAM + bootstrap cho các tầng `layers` (mặc định mọi tầng). Trả về dict kết quả + bảng theo tầng.
    covars: "full" (UPOS + độ dài + nsub + pmix + trans_ent + ngôn ngữ) | "upos_pmix" (bỏ nsub/độ dài/ngôn ngữ) | "none" (chỉ spline: đường trung bình làm mượt, so thẳng với chấm thô).
    cluster: None -> theo CLUSTER_BOOT; True = bootstrap theo cụm khái niệm (cột df.concept), False = theo từng dòng.
    p một phía dùng hiệu chỉnh cộng một: p = (1 + #{replicate vi phạm H0}) / (B + 1) -> không bao giờ báo 0.000 (sàn = 1/(B+1))."""
    layers = list(range(R.shape[1])) if layers is None else list(layers)
    kw_des = {"full": dict(use_upos=use_upos, use_lang=use_lang),
              "upos_pmix": dict(use_upos=True, use_lang=False, use_len=False, use_nsub=False, use_pmix=True, use_trans=False),
              "none": dict(use_upos=False, use_lang=False, use_len=False, use_nsub=False, use_pmix=False, use_trans=False)}[covars]
    des = GamDesign(df, **kw_des)
    X = des.matrix(df); P = des.penalty(X.shape[1]); nb = des.nb
    Y = R[:, layers].astype(np.float64)
    if lam is None:
        kk = layers.index(k_ref) if (k_ref is not None and k_ref in layers) else int(np.argmax(Y.mean(0)))
        lam, edf = gcv_lambda(X, Y[:, kk], P)
    cl = None
    if (CLUSTER_BOOT if cluster is None else cluster) and "concept" in df.columns:
        c = df.concept.values.astype(np.int64); neg = c < 0
        cl = np.where(neg, c.max() + 1 + np.cumsum(neg), c)           # dòng không có đối ứng tiếng Anh: mỗi dòng một cụm riêng
    beta_o, beta_b = gam_bootstrap(X, Y, P, lam, B=B, seed=seed, cl=cl)
    grid, Bg = des.grid_basis(G)
    Bg_t = torch.from_numpy(Bg).float()
    xbar = torch.from_numpy(X[:, nb:].mean(0)).float()
    def curves(beta):                                  # beta [..., p, K] -> f [..., K, G] (đã cộng phần tham số trung bình)
        f = torch.einsum("gb,...bk->...kg", Bg_t, beta[..., :nb, :])
        off = torch.einsum("q,...qk->...k", xbar, beta[..., nb:, :]) if X.shape[1] > nb else 0
        return f + (off[..., None] if X.shape[1] > nb else 0)
    f_o, f_b = curves(beta_o), curves(beta_b)           # [K,G], [B,K,G]
    hh, tl = (float(v) for v in np.quantile(df.log_rank.values, EDGE_Q))
    so, sb = shape_stats(grid, f_o, hh, tl), shape_stats(grid, f_b, hh, tl)
    ci = lambda t: torch.quantile(t, torch.tensor([0.025, 0.975]), dim=0).numpy()
    rows = []
    for j, k in enumerate(layers):
        r = dict(layer=k, lam=lam)
        for nm in ("x_star", "amp", "amp_mean_ends", "amp_q", "slope_left", "slope_right", "slope_left_q", "slope_right_q", "y_star"):
            lo, hi = ci(sb[nm][:, j]); r[nm] = float(so[nm][j]); r[nm + "_lo"], r[nm + "_hi"] = float(lo), float(hi)
        Bn = int(sb["interior"].shape[0]); pv = lambda c, Bn=Bn: float((1.0 + float(c)) / (Bn + 1.0))
        r["B"] = Bn; r["n_clusters"] = int(len(np.unique(cl))) if cl is not None else int(len(df))
        r["p_interior"] = float(sb["interior"][:, j].mean())
        r["p_not_interior"] = pv((sb["interior"][:, j] == 0).sum())
        sl, sr = sb["slope_left"][:, j], sb["slope_right"][:, j]
        r["p_left_up"]    = pv((sl <= 0).sum())    # H0: đầu trái KHÔNG tăng theo hạng (nhỏ => các từ phổ biến nhất kém căn chỉnh hơn từ hạng giữa)
        r["p_left_down"]  = pv((sl >= 0).sum())
        r["p_right_down"] = pv((sr >= 0).sum())    # H0: đầu phải KHÔNG giảm theo hạng (nhỏ => đuôi hiếm kém căn chỉnh)
        r["p_right_up"]   = pv((sr <= 0).sum())
        # Lind–Mehlum (union–intersection), tiêu chí CŨ (độ dốc trên 20% lưới x): đầu trái tăng VÀ đầu phải giảm VÀ đỉnh nằm trong thân; p = max các p một phía
        r["p_LM"] = max(r["p_left_up"], r["p_right_down"], r["p_not_interior"])
        r["shape"] = classify_shape(r["p_left_up"], r["p_left_down"], r["p_right_down"], r["p_right_up"])
        # (v5) cùng cấu trúc nhưng cửa sổ đầu/đuôi theo PHÂN VỊ DỮ LIỆU (EDGE_Q)
        slq, srq = sb["slope_left_q"][:, j], sb["slope_right_q"][:, j]
        r["p_left_up_q"], r["p_left_down_q"] = pv((slq <= 0).sum()), pv((slq >= 0).sum())
        r["p_right_down_q"], r["p_right_up_q"] = pv((srq >= 0).sum()), pv((srq <= 0).sum())
        r["p_LM_q"] = max(r["p_left_up_q"], r["p_right_down_q"], r["p_not_interior"])
        r["shape_q"] = classify_shape(r["p_left_up_q"], r["p_left_down_q"], r["p_right_down_q"], r["p_right_up_q"])
        # (v5) tiêu chí "đỉnh cao hơn CẢ HAI đầu": amp_q > 0 (H0: amp_q <= 0) và đỉnh nằm trong thân
        r["p_amp"] = pv((sb["amp_q"][:, j] <= 0).sum())
        r["p_LM_amp"] = max(r["p_amp"], r["p_not_interior"])
        rows.append(r)
    return dict(table=pd.DataFrame(rows), grid=grid, f_obs=f_o.numpy(), f_boot=f_b.numpy().astype(np.float16), layers=layers,
                names=des.names, nb=nb, beta_obs=beta_o.numpy(), beta_boot=beta_b.numpy(), design=des, lam=lam, X=X, y=Y, P=P)

# ---------------- Mô hình đơn điệu / unimodal (so với GAM bằng CV) ----------------
def unimodal_fit(x, y, nbins=60):
    """Hồi quy unimodal (tăng rồi giảm) bình phương tối thiểu trên các bin phân vị. Trả về hàm dự đoán."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
    bi = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    nb = len(edges) - 1
    w = np.bincount(bi, minlength=nb).astype(float); m = np.bincount(bi, weights=y, minlength=nb) / np.maximum(w, 1)
    ok = w > 0; xs = np.arange(nb)
    best = None
    for md in range(nb):
        l = IsotonicRegression(increasing=True).fit(xs[:md + 1][ok[:md + 1]], m[:md + 1][ok[:md + 1]], sample_weight=w[:md + 1][ok[:md + 1]]).predict(xs[:md + 1])
        if md + 1 < nb and ok[md + 1:].any():
            r = IsotonicRegression(increasing=False).fit(xs[md + 1:][ok[md + 1:]], m[md + 1:][ok[md + 1:]], sample_weight=w[md + 1:][ok[md + 1:]]).predict(xs[md + 1:])
            r = np.minimum(r, l[-1])                     # giữ unimodal: nhánh phải không vượt đỉnh trái
        else:
            r = np.zeros(0)
        f = np.concatenate([l, r]); sse = float((w * (m - f) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, f)
    f = best[1]
    return lambda xn: f[np.clip(np.searchsorted(edges, xn, side="right") - 1, 0, nb - 1)]

def cv_shapes(X, nb, P, lam, y, x, folds=CV_FOLDS, seed=SEED):
    """CV theo K phần: sai số bình phương trên dòng test của 4 mô hình, cùng phần tham số (UPOS/len/nsub/ngôn ngữ) lấy từ GAM huấn luyện:
       const (không phụ thuộc hạng) | mono (đơn điệu tốt nhất, tăng hoặc giảm) | unimodal | gam (tự do)."""
    n = len(y); rng = np.random.default_rng(seed); fold = rng.permutation(n) % folds
    se = {k: np.zeros(n) for k in ("const", "mono", "unimodal", "gam")}
    for f in range(folds):
        tr, te = fold != f, fold == f
        A = X[tr].T @ X[tr] + lam * P
        beta = np.linalg.solve(A, X[tr].T @ y[tr])
        par_tr, par_te = X[tr][:, nb:] @ beta[nb:], X[te][:, nb:] @ beta[nb:]
        ys = y[tr] - par_tr
        se["gam"][te] = (y[te] - X[te] @ beta) ** 2
        se["const"][te] = (y[te] - par_te - ys.mean()) ** 2
        inc = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x[tr], ys)
        dec = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(x[tr], ys)
        mono = inc if ((ys - inc.predict(x[tr])) ** 2).sum() <= ((ys - dec.predict(x[tr])) ** 2).sum() else dec
        se["mono"][te] = (y[te] - par_te - mono.predict(x[te])) ** 2
        se["unimodal"][te] = (y[te] - par_te - unimodal_fit(x[tr], ys)(x[te])) ** 2
    return se

def paired_boot_ci(d, B=2000, seed=SEED):
    rng = np.random.default_rng(seed); n = len(d)
    m = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))
