# Chẩn đoán bổ sung v5 (Cell E2)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 45 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["diag_extra"]) hoặc `python -m hourglass run <experiment> diag_extra`; cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell E2 — (v5) CHẨN ĐOÁN bổ sung sau khi chạy lại (chỉ đọc kết quả; không cần GPU)
#   diag_mask_effect : mặt nạ "mọi bản dịch hợp lệ" nâng R bao nhiêu, theo khoảng hạng? (kiểm A: từ chức năng/phổ biến bị phạt oan vì nhiều dạng dịch)
#   diag_tie_block   : khối hapax / hạng bằng nhau ở biên phải chiếm bao nhiêu dữ liệu, và cửa sổ đuôi (EDGE_Q) có gồm trọn khối đó không? (kiểm C)
#   diag_criteria    : cùng một model, các tiêu chí/biến thể song song (GAM đầy đủ, chỉ spline, không nsub/độ dài/ngôn ngữ, strict, không hapax, λ khác) (kiểm B, C, E)
#   diag_numerics    : sanity batch-vs-single (bf16 và fp32), n_clip theo model (kiểm G)
# ============================================================
def _main_pairs():
    pair_muse = [pi for pi in range(P_) if PAIR_STATS.confirm_src.iloc[pi] in MAIN_SRCS]
    return pair_muse if (MAIN_PAIRS == "muse" and 0 < len(pair_muse) < P_) else None

def diag_mask_effect(key, nb=8):
    run, _ = load_run(key); kw, K1 = waist_layer(key, run)
    if "RR_AB_V" not in run:
        print("[skip] không có biến thể 'valid' (MASK_VALID_TRANSLATIONS=False khi chạy)"); return None
    d1, R1 = word_table(key, _main_pairs(), variant="strict"); d2, R2 = word_table(key, _main_pairs(), variant="valid")
    x = d1.log_rank.values; edges = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1))); bi = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    rows = [dict(bin=f"{edges[i]:.2f}-{edges[i + 1]:.2f}", n=int((bi == i).sum()), R_strict=R1[bi == i, kw].mean(), R_valid=R2[bi == i, kw].mean(),
                 gain=(R2[bi == i, kw] - R1[bi == i, kw]).mean(), trans_ent=d1.trans_ent.values[bi == i].mean(),
                 func_share=d1.upos.isin(FUNC_UPOS).values[bi == i].mean()) for i in range(len(edges) - 1)]
    out = pd.DataFrame(rows); print(f"[{key}] tầng {kw}: tác động của mặt nạ 'bản dịch hợp lệ' theo log10-hạng"); print(out.round(4).to_string(index=False)); return out

def diag_tie_block(key):
    d0, R0 = word_table(key, _main_pairs()); d, R = prep_main(d0, R0)
    x = d.log_rank.values; at_max = x >= x.max() - 1e-9; tl = float(np.quantile(x, EDGE_Q[1]))
    print(f"[{key}] {len(d):,} dạng từ sau prep_main | trùng hạng ở biên phải: {at_max.mean():.1%} | từ có fw_count==1: {(d.fw_count == 1).mean():.1%} | fw_count<=3: {(d.fw_count <= 3).mean():.1%}")
    print(f"   cửa sổ đuôi (x >= phân vị {EDGE_Q[1]:.0%} = {tl:.2f}) gồm {(x >= tl).sum():,} từ, trong đó {(at_max & (x >= tl)).sum() / max((x >= tl).sum(), 1):.1%} là khối hạng-bằng-nhau ở biên")
    print(d.assign(at_max=at_max).groupby("lang").at_max.mean().round(3).to_dict())

def diag_criteria(S3_ALL, subsets=("all", "gam_bare", "gam_upos_pmix", "rr_strict", "rr_valid", "no_hapax", "lam_x0.1", "lam_x10")):
    cols = ["key", "subset", "n_words", "shape", "shape_q", "p_LM", "p_LM_q", "p_LM_amp", "amp", "amp_lo", "amp_hi", "x_star", "p_right_down", "p_right_down_q"]
    t = S3_ALL[S3_ALL.subset.isin(subsets)][[c for c in cols if c in S3_ALL]].copy()
    t["subset"] = pd.Categorical(t.subset, [s for s in subsets if s in set(t.subset)], ordered=True)
    t = t.sort_values(["key", "subset"]); print(t.round(3).to_string(index=False)); return t

def diag_numerics():
    rows = []
    for k in done_models():
        m = read_json(f"{run_dir(k)}/meta.json", {}); s = m.get("sanity") or {}; s32 = m.get("sanity_fp32") or {}
        rows.append(dict(key=k, dtype=m.get("dtype"), n_clip=m.get("n_clip"), numerics_flag=m.get("numerics_flag"),
                         max_rel_diff=max([v.get("max", np.nan) for v in s.values()], default=np.nan),
                         frac_gt_tol=max([v.get("frac", np.nan) for v in s.values()], default=np.nan),
                         fp32_ok=(None if not s32 else all(v.get("ok") for v in s32.values()))))
    t = pd.DataFrame(rows); print(t.round(4).to_string(index=False)); return t

if "S3_ALL" in globals() and len(S3_ALL):
    _ = diag_numerics(); print()
    for _k in done_models():
        diag_tie_block(_k); _ = diag_mask_effect(_k); print()
    _ = diag_criteria(S3_ALL)
