# Capacity: đỉnh r* theo cỡ model (Cell C7)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 35 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C7 — Capacity: đỉnh r* có dịch ra ngoài khi model lớn hơn không? + bảng kết luận S3
# Conjecture 2: r*(µ) = (cδ/µb)^{1/α}; capacity tăng -> µ giảm -> r* tăng, tức log r* tăng theo log B.
# CHÚ Ý: r* chỉ có nghĩa khi đường cong THỰC SỰ có đỉnh. Với đường phẳng / đơn điệu, argmax chỉ là một điểm tùy ý hoặc nằm ở biên.
# v5 (sửa theo rà soát):
#   - argmax dính biên (0.6B ở cận dưới lưới, 8B/4B có CI chạm cận trên): thêm cờ `at_boundary` và thống kê KHÔNG dính biên `centroid`
#     = trọng tâm của (f − min f) trên lưới x (không phải đỉnh; chỉ là vị trí khối lượng của đường cong, dùng kèm các tiêu chí đỉnh);
#   - thêm tiêu chí đỉnh thứ hai peak_ok_amp (đỉnh cao hơn cả hai đầu, p_LM_amp) và tập con interior_peak_only (loại model có đỉnh dính biên);
#   - Kendall τ=1.0 với n=4 có p tối thiểu 2/4! = 0.083 > 0.05 -> không thể "có ý nghĩa": cột kendall_p_floor ghi p tối thiểu đạt được theo n;
#   - thêm slope gộp có hiệu ứng cố định theo họ (6 model) + số cặp cùng họ có cùng dấu (concordant_pairs / n_pairs);
#   - CI lấy từ bootstrap theo cụm của TỪNG model: là sai số lấy mẫu từ, KHÔNG phải CI qua các model; với n nhỏ phải đọc cùng p_floor và leave-one-out.
# ============================================================
def capacity_table(S3_ALL=None):
    ok, ok_amp, shp = {}, {}, {}
    if S3_ALL is not None and len(S3_ALL):
        a = S3_ALL[S3_ALL.subset == "all"]
        ok = {k: bool(p < 0.05) for k, p in zip(a.key, a.p_LM)}; shp = dict(zip(a.key, a["shape"]))
        if "p_LM_amp" in a:
            ok_amp = {k: bool(p < 0.05) for k, p in zip(a.key, a.p_LM_amp)}
    rows = []
    for k in done_models():
        p = f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p); meta = read_json(f"{run_dir(k)}/meta.json", {})
        g = z["grid"]; xb = z["x_star_boot"]; lo_, hi_ = np.percentile(xb, [2.5, 97.5]); span = float(g[-1] - g[0])
        at_b = bool(lo_ <= g[0] + 0.02 * span or hi_ >= g[-1] - 0.02 * span)          # CI của đỉnh chạm biên lưới
        cent = float(np.median(z["centroid_boot"])) if "centroid_boot" in z.files else np.nan
        rows.append(dict(key=k, label=MODELS[k]["label"], family=meta.get("family"), n_params=meta.get("n_params_nonemb"),
                         x_star=float(np.median(xb)), x_obs=float(g[z["f_boot_w"].astype(np.float32).mean(0).argmax()]), centroid=cent,
                         kw=int(z["kw"]), K1=int(z["K1"]), peak_ok=ok.get(k, True), peak_ok_amp=ok_amp.get(k, ok.get(k, True)),
                         at_boundary=at_b, numerics_flag=bool(meta.get("numerics_flag", False)), shape=shp.get(k, "")))
    return pd.DataFrame(rows)

def _slope_stats(g, n_boot, rng, xcol="x_obs", bootkey="x_star_boot"):
    lb = np.log10(g.n_params.values.astype(float)); xo = g[xcol].values.astype(float)
    xs = [np.load(f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz")[bootkey] for k in g.key]
    sl = np.array([np.polyfit(lb, [x[rng.integers(0, len(x))] for x in xs], 1)[0] for _ in range(n_boot)])
    tau, ptau = kendalltau(lb, xo); n = len(g)
    loo = [np.polyfit(np.delete(lb, i), np.delete(xo, i), 1)[0] for i in range(n)] if n >= 4 else [np.nan]
    return dict(n_models=n, slope=float(np.polyfit(lb, xo, 1)[0]), slope_lo=float(np.percentile(sl, 2.5)), slope_hi=float(np.percentile(sl, 97.5)),
                kendall_tau=float(tau), kendall_p=float(ptau), kendall_p_floor=float(min(1.0, 2.0 / math.factorial(n))),
                slope_loo_min=float(np.nanmin(loo)), slope_loo_max=float(np.nanmax(loo)))

def _pooled_fe(g, n_boot, rng, xcol, bootkey):
    """Slope gộp qua các họ model, trừ trung bình TRONG từng họ (hiệu ứng cố định theo họ) + số cặp cùng họ có cùng dấu (Δx, Δlog B)."""
    lb = np.log10(g.n_params.values.astype(float)); fam = g.family.values; xo = g[xcol].values.astype(float)
    xs = [np.load(f"{ANALYSIS_DIR}/s3_{k}{RUN_TAG}.npz")[bootkey] for k in g.key]
    def fe_slope(x):
        num = den = 0.0
        for f in np.unique(fam):
            m = fam == f
            if m.sum() < 2:
                continue
            lc, xc = lb[m] - lb[m].mean(), x[m] - x[m].mean()
            num += float((lc * xc).sum()); den += float((lc ** 2).sum())
        return num / den if den > 0 else np.nan
    sl = np.array([fe_slope(np.array([x[rng.integers(0, len(x))] for x in xs])) for _ in range(n_boot)])
    conc = tot = 0
    for f in np.unique(fam):
        for i, j in itertools.combinations(np.where(fam == f)[0], 2):
            tot += 1; conc += int(np.sign(xo[i] - xo[j]) == np.sign(lb[i] - lb[j]))
    return dict(n_models=len(g), slope=float(fe_slope(xo)), slope_lo=float(np.nanpercentile(sl, 2.5)), slope_hi=float(np.nanpercentile(sl, 97.5)),
                concordant_pairs=conc, n_pairs=tot)

def capacity_slopes(cap, n_boot=1000, seed=SEED):
    rng = np.random.default_rng(seed); out = []
    for stat, xcol, bk in (("peak_argmax", "x_obs", "x_star_boot"), ("centroid", "centroid", "centroid_boot")):
        if xcol not in cap or cap[xcol].isna().all():
            continue
        c_ = cap.dropna(subset=["n_params", xcol])
        for fam, g in c_.groupby("family"):
            subs = [("all_models", g)]
            if stat == "peak_argmax":                    # tập con theo tiêu chí đỉnh chỉ có nghĩa với thống kê đỉnh
                subs += [("peak_ok_only", g[g.peak_ok.astype(bool)]), ("peak_ok_amp_only", g[g.peak_ok_amp.astype(bool)]),
                         ("interior_peak_only", g[~g.at_boundary.astype(bool)])]
            for nm, gg in subs:
                if len(gg) < 3 or (nm != "all_models" and len(gg) == len(g)):
                    continue
                st = _slope_stats(gg, n_boot, rng, xcol, bk); st.update(family=fam, stat=stat, subset=nm, outward=bool(st["slope_lo"] > 0)); out.append(st)
        if c_.family.nunique() >= 1 and len(c_) >= 4:
            st = _pooled_fe(c_, n_boot, rng, xcol, bk); st.update(family="all", stat=stat, subset="pooled_within_family", outward=bool(st["slope_lo"] > 0)); out.append(st)
    cols = ["family", "stat", "subset", "n_models", "slope", "slope_lo", "slope_hi", "kendall_tau", "kendall_p", "kendall_p_floor",
            "slope_loo_min", "slope_loo_max", "concordant_pairs", "n_pairs", "outward"]
    return pd.DataFrame(out, columns=cols)

def verdict_table(S3_ALL):
    """Kết luận S3 trên từng model, báo SONG SONG ba tiêu chí áp dụng đồng loạt cho mọi model (không chọn tiêu chí sau khi xem kết quả):
      inverted_U      : tiêu chí CŨ  = p_LM < 0.05 (độ dốc trên 20% ngoài cùng của lưới x) và mô hình đơn điệu kém GAM có ý nghĩa
      inverted_U_q    : cùng cấu trúc, nhưng đầu/đuôi theo PHÂN VỊ DỮ LIỆU (EDGE_Q)                          -> p_LM_q
      inverted_U_amp  : đỉnh cao hơn cả hai đầu (amp_q > 0) và đỉnh nằm trong thân                           -> p_LM_amp
    A2_refuted: đầu trái TĂNG theo hạng (từ phổ biến nhất kém căn chỉnh hơn từ hạng giữa) -> không thể là 'căn chỉnh tăng đơn điệu theo tần suất'.
    Cột *_bare: GAM chỉ spline (không covariate) — nếu đỉnh chỉ có ở *_bare mà mất ở GAM đầy đủ (hoặc ngược lại) thì kết luận phụ thuộc vào hiệu chỉnh covariate."""
    rows = []
    for k, g in S3_ALL.groupby("key"):
        a = g[g.subset == "all"].iloc[0]; c = g[g.subset == "content"]; b = g[g.subset == "gam_bare"]
        b = None if b.empty else b.iloc[0]
        gv = lambda r, col: (float(r[col]) if (r is not None and col in r and pd.notna(r[col])) else np.nan)
        gam_ok = bool(a.d_mono_vs_gam_lo > 0)
        inv_u = bool(a.p_LM < 0.05 and gam_ok)
        inv_q = bool(gv(a, "p_LM_q") < 0.05 and gam_ok); inv_a = bool(gv(a, "p_LM_amp") < 0.05 and gam_ok)
        c_u = None if c.empty else bool(c.iloc[0].p_LM < 0.05)
        nf = a.get("numerics_flag", False); nf = bool(nf) if pd.notna(nf) else False
        rows.append(dict(key=k, shape=a["shape"], shape_q=a.get("shape_q", ""), shape_bare=(b["shape"] if b is not None else ""),
                         r_star=10 ** a.x_star, r_star_lo=10 ** a.x_star_lo, r_star_hi=10 ** a.x_star_hi,
                         p_LM=a.p_LM, p_LM_q=gv(a, "p_LM_q"), p_LM_amp=gv(a, "p_LM_amp"), p_LM_amp_bare=gv(b, "p_LM_amp"),
                         amp=a.amp, amp_lo=a.amp_lo, amp_hi=a.amp_hi, amp_q=gv(a, "amp_q"), amp_q_lo=gv(a, "amp_q_lo"),
                         p_left_up=a.p_left_up, p_right_down=a.p_right_down, p_left_up_q=gv(a, "p_left_up_q"), p_right_down_q=gv(a, "p_right_down_q"),
                         d_mono_minus_gam=a.d_mono_vs_gam, d_mono_ci_lo=a.d_mono_vs_gam_lo,
                         peak_ok=bool(a.p_LM < 0.05), inverted_U=inv_u, inverted_U_q=inv_q, inverted_U_amp=inv_a,
                         A2_refuted=bool(a.p_left_up < 0.05), A2_refuted_q=bool(gv(a, "p_left_up_q") < 0.05),
                         content_only_inverted_U=c_u, S3_supported=bool(inv_u and (c_u is None or c_u)),
                         numerics_flag=nf, n_clip=a.get("n_clip", np.nan)))
    return pd.DataFrame(rows)
