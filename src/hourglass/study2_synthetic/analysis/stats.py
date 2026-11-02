# Tiện ích thống kê (Cell D2)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 36 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D2 — Tiện ích thống kê (Wilcoxon chính xác, TOST, JT, bước-so-với-tuyến-tính, Holm)
# ============================================================
def holm(ps):
    ps = np.asarray(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j]); out[j] = min(1.0, run)
    return out

def wilcoxon_p(x, y=None, alt="greater"):
    x = np.asarray(x, float) if y is None else np.asarray(x, float) - np.asarray(y, float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.all(x == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x, alternative=alt).pvalue)         # chính xác khi n nhỏ và không có số 0
    except Exception:
        return np.nan

def tost_p(x, margin):
    """TOST cho 'trung bình ≈ 0 trong ±margin' (p nhỏ = bằng nhau theo nghĩa thực dụng)."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 3:
        return np.nan
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(n)
    if se == 0:
        return 0.0 if abs(m) < margin else 1.0
    return float(max(1 - stats.t.cdf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1)))

def jonckheere_terpstra(groups):
    """groups xếp theo thứ tự tăng giả thuyết (nhóm sau lớn hơn). Trả về (J, z, p một phía 'tăng')."""
    g = [np.asarray(x, float)[np.isfinite(x)] for x in groups]
    J = 0.0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            d = g[j][None, :] - g[i][:, None]
            J += float((d > 0).sum() + 0.5 * (d == 0).sum())
    n = np.array([len(x) for x in g]); N = n.sum()
    mu = (N ** 2 - (n ** 2).sum()) / 4.0
    var = (N ** 2 * (2 * N + 3) - (n ** 2 * (2 * n + 3)).sum()) / 72.0
    z = (J - mu) / math.sqrt(var) if var > 0 else np.nan
    return J, float(z), float(1 - stats.norm.cdf(z))

def _aicc(sse, n, k):
    return n * math.log(max(sse, 1e-12) / n) + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)

def step_vs_linear(x, y):
    """x (vd. log2 d), y: quan sát cấp seed. So mô hình bậc thang một điểm gãy với đường thẳng bằng AICc.
    Trả về (ΔAICc = AICc_tuyến_tính - AICc_bậc_thang, chỉ số điểm gãy trong các x duy nhất)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(y); x, y = x[ok], y[ok]; n = len(y)
    if n < 8 or len(np.unique(x)) < 3:
        return np.nan, -1
    b, a = np.polyfit(x, y, 1)
    aic_lin = _aicc(float(np.square(y - (a + b * x)).sum()), n, 3)
    ux = np.unique(x); best = (np.inf, -1)
    for i in range(len(ux) - 1):
        lo = x <= ux[i]
        sse = float(np.square(y[lo] - y[lo].mean()).sum() + np.square(y[~lo] - y[~lo].mean()).sum())
        if sse < best[0]:
            best = (sse, i)
    return aic_lin - _aicc(best[0], n, 4), best[1]

def tcrit_band(M):
    """M [R,K] -> (mean, lo, hi) CI95 (t) qua R run (seed)."""
    M = np.asarray(M, float); R = M.shape[0]
    m = np.nanmean(M, 0)
    if R < 2:
        return m, m, m
    h = stats.t.ppf(0.975, R - 1) * np.nanstd(M, 0, ddof=1) / math.sqrt(R)
    return m, m - h, m + h

def sel(df, **kw):
    out = df
    for k, v in kw.items():
        out = out[out[k].isin(v)] if isinstance(v, (list, tuple, set)) else out[out[k] == v]
    return out

def _has(x):
    """pandas đổi None thành NaN: NaN không phải 'có giá trị' (vd. share_X, len_scale_X của lưới không dùng)."""
    return x is not None and not (isinstance(x, float) and math.isnan(x))

def _res(hyp, verdict, evidence, table=None):
    return dict(hyp=hyp, verdict=verdict, evidence=evidence, table=table)

NO_DATA = "chưa có dữ liệu"
