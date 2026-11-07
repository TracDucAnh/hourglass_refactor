# Bản đồ thí nghiệm: giả thuyết ↔ study ↔ mã ↔ config ↔ kết quả

Theo *Implementation Guides* (kế hoạch LE-D, mục 1, 3, 4) và mục 5 của paper. PDF nằm trong `docs/` nhưng không
được commit (xem `.gitignore`).

## Study trong repo

| Study (paper) | Package | Notebook gốc | Experiment (`configs/experiment/`) | Kết quả thô → bảng → hình |
|---|---|---|---|---|
| Study 1 — hourglass ở 4 họ mô hình mở (Algorithm 1) | `src/hourglass/study1_open_models/` | `study-1.ipynb` | `study1`, `study1_gate_g1`, `study1_quick` | `outputs/study1/` → `results/study1/` → `paper/figures/study1/` |
| Study 2a — ngôn ngữ tổng hợp (Algorithm 2; gồm 3a, 3d) | `src/hourglass/study2_synthetic/` | `study-2a.ipynb` | `study2a`, `study2a_smoke`, `study2a_pilot_conv`, `study2a_gate_g2`, `study2a_grids`, `study2a_{decoder,encoder,moe,diffusion}` | `outputs/study2a/` → `results/study2a/` → `paper/figures/study2a/` |
| Study 4 — tần suất, loại hình (Algorithm 5) | `src/hourglass/study4_frequency/` | `study-4.ipynb` | `study4`, `study4_pilot` | `outputs/study4/` → `results/study4/` → `paper/figures/study4/` |

**Study 2b** ở `src/hourglass/study3_natural/` (không có notebook; xem `docs/studies/study2b.md`).
**Chưa có trong repo**: Study 2c (dùng lại twin của 2b), Study 3 (probe XNLI/UPOS, chiếu bỏ, 3b/3c), Study 5 (dự báo P4). Study 4 tự đăng ký các model 2b
nếu có checkpoint ở `{ROOT}/study3_natural/checkpoints/<tên>/`.

Tên package theo kế hoạch (mục 3): `common/` ← các study; các study không import lẫn nhau. `study4_frequency`
(thay vì `study4_analysis` của kế hoạch) vì notebook Study 4 tự trích biểu diễn từ model, không chỉ đọc kết quả.

## Giả thuyết → nơi kiểm định

| | Phép thử (kế hoạch mục 1) | Chạy ở | Mã kiểm định | Đầu ra chính |
|---|---|---|---|---|
| P1 | Hourglass ở 4 họ | Study 1; 2a (`p1p2`, `moe_diff`) | S1: `analysis.py` (`feature_table`); 2a: `analysis/hypotheses.py` (`test_P1`) | `results/study1/features_*.csv`, `p1_summary_*.json`; `results/study2a/table_P1.csv` |
| P2 | 3 tác vụ LM / SEM / SURF | 2a (`p1p2`) | `test_P2` | `results/study2a/hypotheses_summary.csv` |
| P3 | Cặp sinh đôi đơn ngữ ΔA, theo \|L\| và d | 2b (`p3`, `p3_families`) | `study3_natural/analysis.py` (`test_P3`) | `results/study2b/table_P3.csv` |
| P4 | Dự báo cấu hình giữ ra | Study 5 (chưa có); đã bỏ khỏi 2a | — | — |
| S1 | Probe chuyển giao | Study 3 (chưa có) | — | — |
| S2 | Lưới capacity × L | 2b (`s2`) | `test_S2` | `results/study2b/table_S2.csv` |
| S3 | Căn chỉnh theo log-hạng | 2a (tập dò S3, bổ trợ, tiêu chí nới); Study 4 | 2a: `test_S3`; S4: `analysis/s3.py`, `analysis/shape.py` | `results/study4/s3_*`, `results/study2a/...` |
| S4 | Thiết kế 2 × 2 pivot | 2b (`s4`, logit-lens); Study 4 | `test_S4` | `results/study2b/table_S4.csv` |
| S5 | Quét len_scale | Study 5 (chưa có); đã bỏ khỏi 2a | — | — |
| S6 | Mức che ở diffusion | Study 1 (LLaDA, Dream) | `diffusion_s6_table` | `results/study1/diffusion_masklevels_*.csv` |

## Cổng

Kế hoạch khuyên chạy theo thứ tự này. Trong mã, **các cổng chỉ là báo cáo, không chặn lưới**: `run_grid` không đọc
kết quả cổng, run chưa hội tụ chỉ bị cảnh báo và vẫn được phân tích (báo cáo ghi rõ số run chưa hội tụ).
Chỉ G0 có `assert`, và G0 không nằm trên đường chạy lưới. `make study2a-all` chạy trọn main results của 2a.

| Cổng | Kiểm tra | Lệnh |
|---|---|---|
| G0 | Bộ sinh tổng hợp + số đo (D = 0/G = 1 khi trùng, G ≈ 0 khi độc lập, round-trip parser, BPE) — không cần GPU | `make g0` hoặc `uv run pytest -m slow` |
| G1 | Study 1 trên XLM-R base, Qwen3-1.7B, OLMoE: có G_k* > 0 so với null ở ít nhất một dense | `make study1-g1` |
| (pilot) | Study 2a: chọn BATCH / SEQ_MULT nhỏ nhất đã hội tụ (giao thức v2) | `make study2a-pilot` |
| G2 | Study 2a decoder M4/SEM, 3 seed: U₀/log L ≈ 1, có eo, hội tụ; đo GPU-giờ thật | `make study2a-gate` |
| G3 | Study 2b (`pilot_data` chọn `DATA_FRAC`) | `results/study2b/table_pilot.csv` |

## Ngân sách (kế hoạch)

≈170 A100-giờ (dự trù 200): Study 1 ≈ 20, Study 2a ≈ 50, Study 2b ≈ 48, Study 3 ≈ 20, Study 4 ≈ 8, Study 5 ≈ 20.
