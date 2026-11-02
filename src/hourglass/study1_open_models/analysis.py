# Bootstrap CI, đặc trưng waist, bảng P1 / S6 (Cell D1)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 13 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell D1 — Phân tích từ /mnt/cl (không cần GPU): đường cong + CI, đặc trưng waist, bảng P1 / S6
# ============================================================
import pandas as pd
warnings.filterwarnings("ignore", category=RuntimeWarning)       # nanmean của cột toàn NaN

def boot_curves(g, n_boot=N_BOOT, seed=SEED, chunk=50):
    """g: [P,K1,N] giá trị theo content. Trung bình theo cặp -> [K1,N], rồi bootstrap theo content (95% CI như paper).
    Trả về (đường cong quan sát [K1], các đường bootstrap [B,K1])."""
    gm = np.nanmean(g.astype(np.float32), axis=0)
    K1, N = gm.shape
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, K1), np.float32)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        idx = rng.integers(0, N, size=(m, N))
        out[s:s + m] = gm[:, idx].mean(-1).T
    return gm.mean(1), out

# ---- Tiêu chí waist (P1) ----
WAIST_S_FLOOR = 0.05   # độ sâu tối thiểu của đỉnh so với đầu mút cao hơn (đơn vị G, G in [0,1]); cận dưới CI95 phải vượt ngưỡng này
CKA_SLACK     = 1      # đỉnh CKA được phép nằm ngoài dải waist của G tối đa 1 tầng

def curve_features(Gm):
    """Gm [..., K1] -> (k*, rho*, S, w) theo M1: k*=argmax, S=G_k* - (G_0+G_K)/2, w = tỉ lệ tầng có G >= G_k* - 0.25 S.
    Lưu ý: S theo paper luôn >= 0 (max >= trung bình hai đầu), kể cả với đường đơn điệu hoặc nhiễu,
    nên S chỉ để báo cáo; kết luận 'có waist' dùng edge_depth() bên dưới."""
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0)
    K1 = Gm.shape[-1]
    ks = Gm.argmax(-1)
    Gk = np.take_along_axis(Gm, ks[..., None], -1)[..., 0]
    S = Gk - 0.5 * (Gm[..., 0] + Gm[..., -1])
    w = (Gm >= (Gk - 0.25 * S)[..., None]).mean(-1)
    return ks, ks / (K1 - 1), S, w

def edge_depth(Gm):
    """Độ sâu thật của waist = đỉnh cao hơn CẢ HAI đầu mút bao nhiêu (đơn vị G).
    = 0 nếu đỉnh nằm ở tầng đầu/cuối (đường đơn điệu); nhỏ nếu đỉnh chỉ nhô lên vì nhiễu."""
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0)
    return Gm.max(-1) - np.maximum(Gm[..., 0], Gm[..., -1])

_CURVE_CACHE = {}
def get_curves(ds, run_id, variant):
    """Mỗi metric -> (mean, lo, hi). G và R: CI 95% bootstrap theo content. D, CKA, A: mean ± std qua các cặp ngôn ngữ.
    U: mean ± std qua các lần chia train/test."""
    ck = (ds_tag(ds), run_id, variant)
    if ck in _CURVE_CACHE:
        return _CURVE_CACHE[ck]
    d = load_variant(ds, run_id, variant)
    if d is None:
        return None
    nL = len(d["langs"]); out = {}
    Gm, Gb = boot_curves(d["g_content"])
    out["G"] = (Gm, *np.percentile(Gb, [2.5, 97.5], axis=0)); out["Gboot"] = Gb
    Rm, Rb = boot_curves(d["rr_content"])
    out["R"] = (Rm, *np.percentile(Rb, [2.5, 97.5], axis=0))
    for name in ("D", "CKA", "A"):
        m, s = np.nanmean(d[name], 0), np.nanstd(d[name], 0)
        out[name] = (m, m - s, m + s)
    U = d["U"] / math.log(nL)
    m, s = U.mean(0), U.std(0)
    out["U"] = (m, m - s, m + s)
    out["K1"] = len(Gm)
    _CURVE_CACHE[ck] = out
    return out

def waist_from_curves(Gm, Gb, cka):
    """Gm [K1]: đường G quan sát; Gb [B,K1]: các đường G bootstrap theo content; cka [K1]: CKA trung bình theo cặp.
    Có waist (chưa tính đối chứng random-init) khi đủ CẢ BA điều kiện:
      (1) đỉnh G nằm trong thân (không ở tầng đầu/cuối);
      (2) cận dưới CI95 của edge_depth > WAIST_S_FLOOR (đỉnh cao hơn cả hai đầu một cách có ý nghĩa, không phải nhiễu);
      (3) đỉnh CKA rơi vào dải waist của G (cùng định nghĩa với w, nới thêm CKA_SLACK tầng) -> hai số đo cùng chỉ một vùng."""
    Gm = np.nan_to_num(np.asarray(Gm, np.float64), nan=0.0); K1 = len(Gm)
    k, rho, S, w = (x[0] for x in curve_features(Gm[None]))
    k = int(k)
    kb, rb, Sb, wb = curve_features(Gb)
    ci = lambda x: tuple(float(v) for v in np.percentile(x, [2.5, 97.5]))
    S_ci = ci(Sb)
    Se = float(edge_depth(Gm)); Se_ci = ci(edge_depth(Gb))
    band = Gm >= (Gm[k] - 0.25 * S)                                   # dải waist của G
    k_cka = int(np.nanargmax(cka)) if np.isfinite(cka).any() else -1
    cka_in_band = bool(k_cka >= 0 and band[max(0, k_cka - CKA_SLACK): k_cka + CKA_SLACK + 1].any())
    interior = bool(0 < k < K1 - 1)
    has_curve = bool(interior and Se_ci[0] > WAIST_S_FLOOR and cka_in_band)
    return dict(K=K1 - 1, k_star=k, rho_star=float(rho), rho_lo=ci(rb)[0], rho_hi=ci(rb)[1],
                S=float(S), S_lo=S_ci[0], S_hi=S_ci[1], w=float(w), w_lo=ci(wb)[0], w_hi=ci(wb)[1],
                S_edge=Se, S_edge_lo=Se_ci[0], S_edge_hi=Se_ci[1], interior=interior,
                G_at_kstar=float(Gm[k]), G_first=float(Gm[0]), G_last=float(Gm[-1]),
                k_star_cka=k_cka, dk_G_vs_CKA=abs(k - k_cka), d_rho_G_vs_CKA=abs(k - k_cka) / (K1 - 1),
                cka_in_band=cka_in_band, has_waist_curve=has_curve, has_waist=has_curve)

def waist_features(ds, run_id, variant):
    c = get_curves(ds, run_id, variant)
    if c is None:
        return None
    return waist_from_curves(c["G"][0], c["Gboot"], c["CKA"][0])

def all_runs(ds):
    """[(run_id, key, random_init)] cho các run đã có file kết quả chính."""
    out = []
    for key in MODELS:
        for rnd in (False, True):
            rid = key + ("__random" if rnd else "")
            if os.path.exists(variant_path(ds, rid, main_variant(key, rnd))):
                out.append((rid, key, rnd))
    return out

def feature_table(ds, save=True):
    """Mỗi run một dòng. has_waist cuối cùng = has_waist_curve (xem waist_from_curves) VÀ vượt null random-init cùng họ kiến trúc
    (cận dưới CI95 của S_edge > cận trên CI95 của S_edge ở random-init; nếu chưa chạy random-init thì bỏ qua điều kiện này).
    Với chính các dòng random-init, has_waist chỉ là kiểm tra lành mạnh: kỳ vọng False."""
    rows = []
    for rid, key, rnd in all_runs(ds):
        f = waist_features(ds, rid, main_variant(key, rnd))
        cfg = MODELS[key]
        rows.append(dict(dataset=ds_tag(ds), family=cfg["family"], run=rid, label=cfg["label"] + (" (random)" if rnd else ""),
                         random_init=rnd, variant=main_variant(key, rnd), **f))
    df = pd.DataFrame(rows)
    if len(df):
        null_hi = {fam: float(g.S_edge_hi.max()) for fam, g in df[df.random_init].groupby("family")}
        df["S_edge_random_hi"] = df.family.map(null_hi)
        df["above_random"] = [None if (r.random_init or pd.isna(r.S_edge_random_hi)) else bool(r.S_edge_lo > r.S_edge_random_hi)
                              for r in df.itertuples()]
        ok_null = df.above_random.map(lambda v: True if v is None else bool(v))
        df["has_waist"] = (df.has_waist_curve & ok_null).astype(bool)
    if save and len(df):
        df.to_csv(f"{ANALYSIS_DIR}/features_{ds_tag(ds)}.csv", index=False)
        p1 = {}
        for fam in FAMILIES:
            sub = df[(df.family == fam) & (~df.random_init)]
            rnd_ = df[(df.family == fam) & (df.random_init)]
            p1[fam] = dict(n_models=int(len(sub)), n_with_waist=int(sub.has_waist.sum()),
                           models_without_waist=sub.loc[~sub.has_waist, "label"].tolist(),
                           random_S=(None if rnd_.empty else float(rnd_.S.iloc[0])),
                           random_S_ci=(None if rnd_.empty else [float(rnd_.S_lo.iloc[0]), float(rnd_.S_hi.iloc[0])]),
                           random_S_edge=(None if rnd_.empty else float(rnd_.S_edge.iloc[0])),
                           random_S_edge_ci=(None if rnd_.empty else [float(rnd_.S_edge_lo.iloc[0]), float(rnd_.S_edge_hi.iloc[0])]),
                           random_has_waist=(None if rnd_.empty else bool(rnd_.has_waist.iloc[0])))
        write_json(f"{ANALYSIS_DIR}/p1_summary_{ds_tag(ds)}.json", p1)
    return df

def diffusion_s6_table(ds, save=True):
    """S6: mức che đổi các tầng cuối nhưng không đổi k*. So sánh rho* ở mỗi mức che với t=0 (biên TOST |drho| <= 0.10)."""
    rows = []
    for key, cfg in MODELS.items():
        if cfg["family"] != "diffusion":
            continue
        base = waist_features(ds, key, variant_name(0.0, "all"))
        for t in MASK_LEVELS:
            for pool in ("all", "masked"):
                if pool == "masked" and t == 0:
                    continue
                f = waist_features(ds, key, variant_name(t, pool))
                if f is None:
                    continue
                rows.append(dict(dataset=ds_tag(ds), model=cfg["label"], t=t, pool=pool, k_star=f["k_star"],
                                 rho_star=f["rho_star"], rho_lo=f["rho_lo"], rho_hi=f["rho_hi"], S=f["S"], G_last=f["G_last"],
                                 d_rho_vs_t0=(f["rho_star"] - base["rho_star"]) if base else np.nan,
                                 d_Glast_vs_t0=(f["G_last"] - base["G_last"]) if base else np.nan,
                                 k_star_stable=(abs(f["rho_star"] - base["rho_star"]) <= 0.10) if base else None))
    df = pd.DataFrame(rows)
    if save and len(df):
        df.to_csv(f"{ANALYSIS_DIR}/diffusion_masklevels_{ds_tag(ds)}.csv", index=False)
    return df

def export_curves_csv(ds):
    """Bảng dạng dài (mỗi dòng = một tầng của một run/variant) để vẽ lại bằng công cụ khác."""
    rows = []
    for key, cfg in MODELS.items():
        for rnd in (False, True):
            rid = key + ("__random" if rnd else "")
            for _, vs in plan_levels(cfg, rnd):
                for v in vs:
                    c = get_curves(ds, rid, v)
                    if c is None:
                        continue
                    K1 = c["K1"]
                    for k in range(K1):
                        r = dict(dataset=ds_tag(ds), run=rid, family=cfg["family"], label=cfg["label"], random_init=rnd,
                                 variant=v, layer=k, rho=k / (K1 - 1))
                        for name in ("G", "D", "CKA", "A", "R", "U"):
                            m, lo, hi = c[name]
                            r.update({f"{name}_mean": m[k], f"{name}_lo": lo[k], f"{name}_hi": hi[k]})
                        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(f"{ANALYSIS_DIR}/curves_{ds_tag(ds)}.csv", index=False)
    return df

def print_p1_summary(df):
    cols = ["family", "label", "k_star", "rho_star", "S", "S_edge", "S_edge_lo", "w", "k_star_cka", "cka_in_band", "above_random", "has_waist"]
    print(df[cols].round(3).to_string(index=False))
    print(f"  (waist = đỉnh G trong thân + S_edge_lo > {WAIST_S_FLOOR} + đỉnh CKA trong dải waist của G (±{CKA_SLACK} tầng) + vượt random-init)")
    for fam in FAMILIES:
        sub = df[(df.family == fam) & (~df.random_init)]
        if len(sub):
            print(f"  {fam:10s}: {int(sub.has_waist.sum())}/{len(sub)} model có waist")
        rnd_ = df[(df.family == fam) & (df.random_init)]
        if len(rnd_) and bool(rnd_.has_waist.iloc[0]):
            print(f"  [!] random-init của họ {fam} cũng ra waist -> kiểm tra lại (null không sạch)")
