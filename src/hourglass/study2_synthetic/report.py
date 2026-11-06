# Báo cáo kết luận (Cell D6)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 40 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell D6 — Báo cáo: bảng kết luận từng giả thuyết + số liệu chính
# ============================================================
def build_report(df, results):
    L_ = []; w = L_.append
    w(f"# Báo cáo Study 2a — ngôn ngữ tổng hợp ({datetime.now():%Y-%m-%d %H:%M})\n")
    w("> Mọi kết luận chỉ ở **quy mô nhỏ, trong ngôn ngữ tổng hợp** (d = 192, K = 8). Kết quả âm chưa bác bỏ LE-D ở quy mô lớn. "
      "Phạm vi 2a: P1 (có waist), P2 (waist theo tác vụ), S3 (chữ U ngược theo tần suất, bổ trợ).\n")
    cov = df.groupby("grid").size().to_dict() if len(df) else {}
    w("## Phạm vi dữ liệu\n"); w("| lưới | run hoàn thành |\n|---|---|")
    for g_, n_ in cov.items():
        w(f"| {g_} | {n_} |")
    w("\n## Kết luận theo giả thuyết\n"); w("| giả thuyết | kết luận | bằng chứng |\n|---|---|---|")
    for r in results:
        w(f"| **{r['hyp']}** | {r['verdict']} | {r['evidence']} |")
    if len(df):
        w("\n## Waist theo (họ, mức, tác vụ), overlap 0%\n")
        g = sel(df, overlap=0, grid=["p1p2", "moe_diff", "gate_g2", "smoke"]).groupby(["family", "level", "task"]).agg(
            n=("rid", "size"), frac_waist=("has_waist", "mean"), rho=("rho_star", "median"), S_edge=("S_edge", "median"), width=("w", "median"),
            U0=("U0_over_logL", "median"), U_waist=("U_at_kstar_over_logL", "median")).round(3).reset_index()
        try:
            w(g.to_markdown(index=False))
        except Exception:                                   # chưa cài tabulate
            w("```\n" + g.to_string(index=False) + "\n```")
        if (df.U0_over_logL < 0.9).any():
            w("\n> ⚠ Có run với U₀/log|L| < 0.9 ở tầng embedding: ngôn ngữ chưa đọc ra hoàn toàn từ chữ (kiểm tra giao diện/overlap).")
    if "converged" in df.columns and (df.converged == False).any():
        w(f"\n> ⚠ {int((df.converged == False).sum())}/{len(df)} run chưa đạt tiêu chí hội tụ (val còn giảm >= {CONV_TOL} nats ở ~1/6 bước cuối): "
          "waist của các run này có thể là artifact của giai đoạn huấn luyện; xem cột `conv_drop` trong features_all_runs.csv.")
    w("\n## Hình\n")
    for f_ in sorted(os.listdir(FIG_DIR)):
        w(f"- `figures/{f_}`")
    w("\n## Lưu ý đọc kết quả\n- P1 dựa trên ƯỚC LƯỢNG qua seed: ΔS = S_edge(trained) - S_edge(init) với CI95, độ sâu tương đối S/G*, vị trí ρ* "
      "(xem table_P1.csv). Cột frac_waist / frac_runs_study1 là tỉ lệ run qua tiêu chí từng run của Study 1 (đỉnh G trong thân, "
      "CI dưới của edge depth > 0.05, CKA khớp, vượt `init`) - chỉ để đối chiếu, không dùng để kết luận.\n"
      "- P2 theo đúng phát biểu của paper: SEM so với SURF ở M4 (tới 12 seed), SURF hẹp hơn (độ rộng) hoặc eo dịch (vị trí), "
      "bác nếu eo cố định; ước lượng ghép theo seed với CI95. Dòng role = mô tả trong table_P2.csv (độ sâu, cặp có LM, M0/M2) "
      "không dùng cho kết luận P2.\n"
      "- Hình hourglass lý tưởng của Corollary 1 (quay lại H(Λ) ở tầng cuối) hiếm khi đạt: so G_K với G_0 trong bảng đặc trưng.\n"
      "- Biểu diễn câu: họ causal (decoder, MoE) lấy ở token nội dung cuối; họ hai chiều (encoder, diffusion) trung bình theo từ.\n"
      "- D_anchor là chẩn đoán (lưới mask_ablation, anchors), không phải giả thuyết của paper; P1/P2/S3 không dùng các lưới này.\n")
    txt = "\n".join(L_)
    with open(f"{ANALYSIS_DIR}/report_study2a.md", "w", encoding="utf-8") as f:
        f.write(txt)
    pd.DataFrame([dict(hypothesis=r["hyp"], verdict=r["verdict"], evidence=r["evidence"]) for r in results]).to_csv(f"{ANALYSIS_DIR}/hypotheses_summary.csv", index=False)
    for r in results:
        if r["table"] is not None and len(r["table"]):
            r["table"].to_csv(f"{ANALYSIS_DIR}/table_{r['hyp']}.csv", index=False)
    return txt

def regenerate_all():
    df = build_feature_table()
    if df.empty:
        print("Chưa có run nào hoàn thành trong", RUN_DIR); return None, []
    res = run_all_tests(df)
    make_all_figures(df)
    txt = build_report(df, res)
    print(txt)
    return df, res
