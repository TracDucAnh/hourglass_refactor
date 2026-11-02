# Độ bền theo lựa chọn ngôn ngữ (Cell C5b)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 32 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["robustness"]) hoặc `python -m hourglass run <experiment> robustness`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C5b — Độ bền theo lựa chọn ngôn ngữ (chạy sau cell C5): kết luận S3 có phụ thuộc vào jpn / zho / swh không?
#   Nguồn: các tập con loo_<ngôn ngữ> (leave-one-language-out), loo_problem_langs (bỏ cùng lúc PROBLEM_LANGS), ext_tier_only, all_pairs trong s3_summary.
#   same_shape / same_peak_verdict: so với tập con "all" (phân tích chính). d_log10_r_star chỉ có nghĩa khi CẢ HAI đường cong có đỉnh (peak_ok).
#   Kết luận "không cần thay ngôn ngữ" chỉ khi robust_to_lang_choice = True ở mọi model; ngược lại xem ngôn ngữ nào làm đổi kết luận (cột flips).
# ============================================================
def robustness_table(S3_ALL, ref="all"):
    rows = []
    for k, g in S3_ALL.groupby("key"):
        a = g[g.subset == ref]
        if a.empty:
            continue
        a = a.iloc[0]
        for _, r in g[g.subset.str.startswith(("loo_", "ext_tier", "all_pairs"))].iterrows():
            both_peak = bool(r.p_LM < 0.05 and a.p_LM < 0.05)
            rows.append(dict(key=k, subset=r.subset, n_words=int(r.n_words), shape=r["shape"], same_shape=bool(r["shape"] == a["shape"]),
                             peak_ok=bool(r.p_LM < 0.05), same_peak_verdict=bool((r.p_LM < 0.05) == (a.p_LM < 0.05)), p_LM=float(r.p_LM),
                             p_left_up=float(r.p_left_up), p_right_down=float(r.p_right_down), r_star=float(10 ** r.x_star),
                             d_log10_r_star=float(r.x_star - a.x_star) if both_peak else np.nan))
    return pd.DataFrame(rows)

def robustness_summary(rt):
    rows = []
    for k, g in rt.groupby("key"):
        loo = g[g.subset.str.startswith("loo_") & (g.subset != "loo_problem_langs")]
        prob = g[g.subset == "loo_problem_langs"]
        flips = loo[~(loo.same_shape & loo.same_peak_verdict)].subset.str.replace("loo_", "", regex=False).tolist()
        dr = loo.d_log10_r_star.abs()
        rows.append(dict(key=k, n_loo=len(loo), n_same_shape=int(loo.same_shape.sum()), n_same_peak_verdict=int(loo.same_peak_verdict.sum()),
                         max_abs_d_log10_r_star=float(dr.max()) if dr.notna().any() else np.nan,
                         argmax_lang=(loo.loc[dr.idxmax(), "subset"].replace("loo_", "") if dr.notna().any() else ""),
                         problem_langs_same=(bool(prob.iloc[0].same_shape and prob.iloc[0].same_peak_verdict) if len(prob) else None),
                         flips=",".join(flips),
                         robust_to_lang_choice=bool(len(loo) and len(flips) == 0 and (prob.empty or (prob.iloc[0].same_shape and prob.iloc[0].same_peak_verdict)))))
    return pd.DataFrame(rows)

if "S3_ALL" in globals() and len(S3_ALL):
    ROBUST = robustness_table(S3_ALL); ROBUST_SUM = robustness_summary(ROBUST)
    ROBUST.to_csv(f"{ANALYSIS_DIR}/s3_robustness{RUN_TAG}.csv", index=False); ROBUST_SUM.to_csv(f"{ANALYSIS_DIR}/s3_robustness_summary{RUN_TAG}.csv", index=False)
    print(ROBUST_SUM.round(3).to_string(index=False)); print()
    print(ROBUST[ROBUST.subset.isin(["ext_tier_only", "all_pairs", "loo_problem_langs", "loo_jpn", "loo_zho", "loo_swh"])].round(3).to_string(index=False))
