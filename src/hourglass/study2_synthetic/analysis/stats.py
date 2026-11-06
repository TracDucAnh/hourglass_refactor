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
def wilcoxon_p(x, y=None, alt="greater"):
    x = np.asarray(x, float) if y is None else np.asarray(x, float) - np.asarray(y, float)
    x = x[np.isfinite(x)]
    if len(x) < 4 or np.all(x == 0):
        return np.nan
    try:
        return float(stats.wilcoxon(x, alternative=alt).pvalue)         # chính xác khi n nhỏ và không có số 0
    except Exception:
        return np.nan

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

def _res(hyp, verdict, evidence, table=None):
    return dict(hyp=hyp, verdict=verdict, evidence=evidence, table=table)

NO_DATA = "chưa có dữ liệu"
