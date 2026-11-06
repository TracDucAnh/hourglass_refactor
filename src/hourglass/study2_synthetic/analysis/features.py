# Tải kết quả + đặc trưng waist (Cell D1)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 35 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D1 — Tải kết quả + đặc trưng waist cho từng run
# ============================================================
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore", category=RuntimeWarning)

FAMILIES = ["decoder", "encoder", "moe", "diffusion"]
TASKS = ["LM", "SEM", "SURF"]
LEVELS = ("M0", "M2", "M4")
WAIST_S_FLOOR = 0.05      # cận dưới CI95 của edge depth phải vượt ngưỡng này
CKA_SLACK = 1             # đỉnh CKA được phép nằm ngoài dải waist của G tối đa 1 tầng
INCLUDE_PILOT = False     # lưới pilot_conv chỉ để chọn ngân sách huấn luyện, KHÔNG đưa vào kiểm định giả thuyết
RHO_MARGIN = 0.10         # SESOI cho vị trí waist

def _rj(p):
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)

def _ld(rid, name):
    p = f"{RUN_DIR}/{rid}/{name}.npz"
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def scan_runs():
    rows = []
    for rid in sorted(os.listdir(RUN_DIR)):
        m = _rj(f"{RUN_DIR}/{rid}/meta.json")
        if m and m.get("status") == "ok":
            c = m["cfg"]
            if m.get("train_version") != TRAIN_VERSION:
                continue                                  # giao thức v1 (750 bước, chưa hội tụ) -> không dùng
            if c.get("grid") == "pilot_conv" and not INCLUDE_PILOT:
                continue
            rows.append(dict(rid=rid, **c, seconds=m["seconds"], loss=m["final_lm_loss"], params=m["params_nonemb"],
                             val_loss=m.get("val_loss"), conv_drop=m.get("conv_drop"), converged=m.get("converged"),
                             fert_mean=float(np.mean(m["fertility"])), fert_X=float(m["fertility"][0]),
                             fert_list=m["fertility"]))
    return pd.DataFrame(rows)

_CV = {}
def cv(rid, variant="trained"):
    """Đường cong (trung bình qua cặp ngôn ngữ) của một run/variant, có cache."""
    key = (rid, variant)
    if key not in _CV:
        d = _ld(rid, variant)
        if d is None:
            _CV[key] = None
        else:
            o = dict(G=np.nanmean(d["G"], 0), D=np.nanmean(d["D"], 0), CKA=np.nanmean(d["CKA"], 0), A=np.nanmean(d["A"], 0),
                     gc=d["g_content"].astype(np.float32), U=d["U_lin"], Um=d["U_mlp"], Apair=d["A"], Gpair=d["G"])
            if "G_rank_w" in d:                        # S3 bản đường cơ sở trong bin; run cũ (khoá G_rank) không dùng cho S3
                o["Grank"], o["rank_counts"] = d["G_rank_w"], d["rank_counts"]
            _CV[key] = o
    return _CV[key]

def boot_G(gc, n_boot=None, seed=SEED, chunk=100):
    """gc [K1,N] (g theo nội dung, đã trung bình qua cặp) -> các đường G bootstrap theo nội dung [B,K1]."""
    n_boot = n_boot or N_BOOT
    K1, N = gc.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, K1), np.float32)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        out[s:s + m] = gc[:, rng.integers(0, N, size=(m, N))].mean(-1).T
    return out

def curve_features(Gm):
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0)
    K1 = Gm.shape[-1]
    ks = Gm.argmax(-1)
    Gk = np.take_along_axis(Gm, ks[..., None], -1)[..., 0]
    S = Gk - 0.5 * (Gm[..., 0] + Gm[..., -1])
    w = (Gm >= (Gk - 0.25 * S)[..., None]).mean(-1)
    return ks, ks / (K1 - 1), S, w

def edge_depth(Gm):
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0)
    return Gm.max(-1) - np.maximum(Gm[..., 0], Gm[..., -1])

def _slope(y, x):
    return float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan

def waist_from_curves(Gm, Gb, cka):
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0); K1 = len(Gm)
    k, rho, S, w = (x[0] for x in curve_features(Gm[None])); k = int(k)
    kb, rb, Sb, wb = curve_features(Gb)
    ci = lambda x: tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))
    Se, Se_ci = float(edge_depth(Gm)), ci(edge_depth(Gb))
    band = Gm >= (Gm[k] - 0.25 * S)
    k_cka = int(np.nanargmax(cka)) if np.isfinite(cka).any() else -1
    cka_in = bool(k_cka >= 0 and band[max(0, k_cka - CKA_SLACK): k_cka + CKA_SLACK + 1].any())
    interior = bool(0 < k < K1 - 1)
    x = np.arange(K1)
    return dict(K=K1 - 1, k_star=k, rho_star=float(rho), rho_lo=ci(rb)[0], rho_hi=ci(rb)[1], S=float(S), w=float(w),
                S_edge=Se, S_edge_lo=Se_ci[0], S_edge_hi=Se_ci[1], interior=interior,
                G_at_kstar=float(Gm[k]), G_first=float(Gm[0]), G_last=float(Gm[-1]),
                k_star_cka=k_cka, cka_in_band=cka_in, has_waist_curve=bool(interior and Se_ci[0] > WAIST_S_FLOOR and cka_in),
                slope_left=_slope(Gm[:k + 1], x[:k + 1]), slope_right=_slope(Gm[k:], x[k:]))

_FEAT = {}
def feat(rid, variant="trained"):
    key = (rid, variant)
    if key not in _FEAT:
        c = cv(rid, variant)
        _FEAT[key] = None if c is None else waist_from_curves(c["G"], boot_G(c["gc"]), c["CKA"])
    return _FEAT[key]

def build_feature_table(save=True):
    runs = scan_runs()
    if runs.empty:
        return runs
    rows = []
    for r in runs.itertuples():
        f, fi = feat(r.rid, "trained"), feat(r.rid, "init")
        if f is None:
            continue
        c = cv(r.rid, "trained")
        row = {**r._asdict(), **f}
        row["S_edge_init"] = fi["S_edge"] if fi else np.nan
        row["S_edge_init_hi"] = fi["S_edge_hi"] if fi else np.nan
        row["above_init"] = bool(f["S_edge_lo"] > fi["S_edge_hi"]) if fi else True
        row["has_waist"] = bool(f["has_waist_curve"] and row["above_init"])
        row["U0_over_logL"] = float(c["U"][0] / math.log(r.L))
        row["U_at_kstar_over_logL"] = float(c["U"][f["k_star"]] / math.log(r.L))
        row["Umlp_at_kstar_over_logL"] = float(c["Um"][f["k_star"]] / math.log(r.L))
        row["ln_params"] = math.log(r.params)
        rows.append(row)
    df = pd.DataFrame(rows).drop(columns=["Index"], errors="ignore")
    if save and len(df):
        df.drop(columns=["fert_list"]).to_csv(f"{ANALYSIS_DIR}/features_all_runs.csv", index=False)
    return df
