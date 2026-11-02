# Bảng mức độ bằng chứng (Cell C7b)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 37 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["evidence"]) hoặc `python -m hourglass run <experiment> evidence`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C7b — (v5b) BẢNG MỨC ĐỘ BẰNG CHỨNG: báo cáo độ lớn + độ bất định + độ lệch giữa các biến thể, KHÔNG gán nhãn qua/trượt
#   Dữ liệu thật không sạch (tokenizer, số học bf16, từ điển, hapax...), nên câu hỏi hợp lý là "bướu lớn cỡ nào, ổn định đến đâu" chứ không phải "qua cửa p<0.05 hay không".
#   Mỗi model: (1) độ lớn bướu amp_q, độ tăng đầu trái, độ giảm đuôi phải, vị trí đỉnh r*, lợi ích CV của GAM so với đơn điệu — kèm CI;
#              (2) độ lệch giữa các biến thể (GAM chỉ-spline, strict/valid, không hapax, λ khác, leave-one-language-out): min/median/max và tỉ lệ biến thể còn dạng tăng-rồi-giảm.
#   Đọc kèm: amp_q LUÔN >= 0 theo cách tính (đỉnh >= trung bình cửa sổ) nên đừng đọc "CI không chứa 0" như bằng chứng; hãy đọc ĐỘ LỚN và so với độ lệch giữa các biến thể.
# ============================================================
EVID_VARIANTS = ("gam_bare", "gam_upos_pmix", "rr_strict", "rr_valid", "no_hapax", "lam_x0.1", "lam_x10")

def _num(d, c):
    return pd.to_numeric(d[c], errors="coerce") if c in d else pd.Series(np.nan, index=d.index)

def evidence_table(S3_ALL, variants=EVID_VARIANTS):
    rows = []
    for k, g in S3_ALL.groupby("key"):
        a = g[g.subset == "all"]
        if a.empty:
            continue
        a = a.iloc[0]
        v = g[g.subset.isin(variants) | (g.subset.str.startswith("loo_") & (g.subset != "loo_problem_langs"))]
        pool = pd.concat([a.to_frame().T, v], ignore_index=True)                         # phân tích chính + mọi biến thể/LOO
        amp, sl, sr, xs = _num(pool, "amp_q"), _num(pool, "slope_left_q"), _num(pool, "slope_right_q"), _num(pool, "x_star")
        rows.append(dict(
            key=k, n_words=int(a.n_words), n_variants=len(pool) - 1,
            bump=a.get("amp_q"), bump_lo=a.get("amp_q_lo"), bump_hi=a.get("amp_q_hi"),
            rise_left=a.get("slope_left_q"), rise_lo=a.get("slope_left_q_lo"), rise_hi=a.get("slope_left_q_hi"),
            fall_right=a.get("slope_right_q"), fall_lo=a.get("slope_right_q_lo"), fall_hi=a.get("slope_right_q_hi"),
            r_star=10 ** a.x_star, r_star_lo=10 ** a.x_star_lo, r_star_hi=10 ** a.x_star_hi, p_interior=a.get("p_interior"),
            cv_gain_vs_mono=a.get("d_mono_vs_gam"), cv_gain_lo=a.get("d_mono_vs_gam_lo"), cv_gain_hi=a.get("d_mono_vs_gam_hi"),
            bump_min=amp.min(), bump_med=amp.median(), bump_max=amp.max(),
            frac_rise_then_fall=float(((sl > 0) & (sr < 0)).mean()),                      # tỉ lệ cấu hình (điểm ước lượng) có đầu trái tăng VÀ đuôi phải giảm
            log10_rstar_range=float(xs.max() - xs.min())))
    return pd.DataFrame(rows)

def evidence_text(r):
    f = lambda x, p=3: "nan" if pd.isna(x) else f"{x:+.{p}f}"
    return (f"{r.key}: bướu {r.bump:.3f} [{r.bump_lo:.3f}, {r.bump_hi:.3f}] (giữa các biến thể {r.bump_min:.3f}–{r.bump_max:.3f}); "
            f"độ dốc đầu trái {f(r.rise_left)}, đuôi phải {f(r.fall_right)}; {r.frac_rise_then_fall:.0%} cấu hình giữ dạng tăng-rồi-giảm; "
            f"r*≈{r.r_star:.0f}, đỉnh dao động {r.log10_rstar_range:.2f} dex giữa các biến thể")

if "S3_ALL" in globals() and len(S3_ALL):
    EVID = evidence_table(S3_ALL); EVID.to_csv(f"{ANALYSIS_DIR}/s3_evidence{RUN_TAG}.csv", index=False)
    print(EVID.round(3).to_string(index=False)); print()
    for _, _r in EVID.iterrows():
        print(" -", evidence_text(_r))
