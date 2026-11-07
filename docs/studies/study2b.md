# Study 2b — Ngôn ngữ tự nhiên: eo và căn chỉnh theo capacity và số ngôn ngữ (P3, S2, S4; P4, S5 phụ)

> Mã: `src/hourglass/study3_natural/` (không có notebook gốc). Config: `configs/experiment/study2b*.yaml`,
> `configs/data/natural.yaml`, `configs/model/natural_lm.yaml`, `configs/trainer/natural.yaml`.
> `python -m hourglass stages study2b` liệt kê các bước.

## Câu hỏi và giả thuyết

Study 1 cho thấy DCD tồn tại; Study 2a cho thấy eo đi theo yêu cầu của các lớp sau. Study 2b hỏi **eo từ đâu ra và
lớn lên thế nào khi đổi capacity và số ngôn ngữ, trên ngôn ngữ thật** — nơi đầu tiên tách được LE-D khỏi A2 (mọi mô
hình tốt đằng nào cũng giống nhau) và A3 (dịch qua ngôn ngữ áp đảo).

| Giả thuyết | Vai trò (Bảng 6) | Đo gì | Lưới |
|---|---|---|---|
| **P3** | ● (chung với 2c) | ΔA = A_joint − A_indep ở eo; ΔA **tăng theo \|L\|** ở d và dữ liệu mỗi ngôn ngữ cố định | `p3`, `p3_families` |
| **S2** | ● | căn chỉnh ở eo theo d là **bậc thang**; SD giữa seed đạt đỉnh ở bước nhảy | `s2` |
| **S4** | ● | pivot (logit-lens) là ngôn ngữ **ít token nhất** (LE-D) hay **nhiều dữ liệu nhất** (A3) | `s4` |
| **S3** | ● | chữ U ngược theo tần suất, r\* dịch theo capacity | dùng lại checkpoint qua **Study 4** (xem dưới) |
| P4 | ○ | hàm ΔA ~ log\|L\| + log d hiệu chỉnh trên `p3`, kiểm trên ô giữ ra | `holdout_p4` |
| S5 | ○ | ρ\* tương quan với chi phí token | `s4` (các tokenizer khác nhau) |

## Thiết kế

**Dữ liệu.** FineWeb-2 (tiếng Anh: FineWeb sample-10BT), 8 ngôn ngữ, hai thứ tự lồng 2 ⊂ 4 ⊂ 8:

| Thứ tự | 2 | 4 | 8 | Ý nghĩa |
|---|---|---|---|---|
| A | eng, hin | + rus, tur | + zho, jpn, fra, arb | cân bằng trật tự từ (SVO / SOV) ở mọi cỡ — **kết luận chính** |
| B | eng, fra | + rus, zho | + hin, tur, jpn, arb | dồn SOV / VSO về cuối: \|L\| lớn = khoảng cách typology lớn |

So A với B tách hiệu ứng |L| khỏi hiệu ứng khoảng cách typology (2a: trật tự từ khác nhau làm eo của decoder nông
đi rất mạnh). Lượng dữ liệu đo theo **byte đã hiệu chỉnh byte premium** (premium = byte của cùng nội dung FLORES dev so
với tiếng Anh): corpus mỗi ngôn ngữ = `CORPUS_BYTES_ENG` × premium (mặc định ≈ 200M token tiếng Anh); một run dùng
`DATA_FRAC` của corpus (mặc định 0.5 ≈ 100M token / ngôn ngữ). **Mỗi ngôn ngữ dùng đúng cùng dữ liệu ở mọi \|L\|**
(nhánh *matched-data* của paper). Văn bản chứa 13-gram của FLORES dev / devtest bị loại. Đuôi `VAL_FRAC` của mỗi
corpus là tập val (không huấn luyện).

**Tokenizer.** Một BPE byte-level chung (`VOCAB` = 64k) train trên cả 8 ngôn ngữ, dùng cho **mọi** run — nên twin của
một ngôn ngữ được huấn luyện **một lần** và dùng cho mọi \|L\| và mọi thứ tự lồng. Riêng S4 dùng hai biến thể làm X
(`S4_LANG`, mặc định `rus_Cyrl`) tốn token (trọng số 0.25 khi train BPE) hoặc rẻ (trọng số 4).

**Mô hình** (định dạng Hugging Face). Decoder: Llama 12 tầng, d ∈ {256, 384, 512} (P3), {128 … 512} (S2). MoE: Mixtral
8 expert nửa cỡ, top-2 (tham số kích hoạt khớp decoder). Diffusion: cùng khung Llama với mask 4D hai chiều, loss
MDLM (che t ~ U(0.05, 1), trọng số 1/t). MoE và diffusion chỉ ở d = 384 (P3 có đúng ngoài decoder không).

**Huấn luyện.** AdamW, batch 128 × 512 token, warmup 2% rồi cosine về 10%, không dừng sớm. Mô hình chung thấy mỗi
token một lần; twin lặp dữ liệu của mình 2 epoch để không kém mô hình chung vì thiếu bước (≤ ~4 epoch gần như dữ liệu
mới). Diffusion được ×2 epoch (2a: diffusion cần lâu hơn). Mỗi run ghi `conv_drop` / `converged` như 2a.

**Đo.** FLORES-200 devtest (1 012 câu), M1 (D, G) và M2 (CKA, A Procrustes-CV) theo tầng, **cả hai cách đọc**: token
cuối có chữ / số (bỏ dấu câu cuối câu) và trung bình token. Cách đọc chính chốt trước (`READOUT: family`): causal =
token cuối, hai chiều = trung bình, như 2a. A_joint và A_indep (giữa hai twin) dùng cùng một hàm và cùng cách chia nửa
nội dung. Bản khởi tạo ngẫu nhiên của mỗi mô hình chung được đo làm mốc (ΔS). Decoder / MoE có thêm logit-lens (S4).

**Thống kê.** Đơn vị là seed; báo cáo bằng ước lượng với CI95 t qua seed, như 2a.

| Kết luận | Điều kiện (`analysis.py`) |
|---|---|
| P3 ủng hộ | decoder, thứ tự A: CI của ΔA ở eo > 0 ở mọi (d, \|L\|) **và** CI của độ dốc ΔA theo log2\|L\| > 0 ở mọi d |
| P3 bác (A2) | mọi CI của ΔA nằm trong ±max(0.05, 1 SD seed) |
| S2 ủng hộ | ΔAICc (tuyến tính − một bậc) ≥ 2 và SD giữa seed lớn nhất trong ±1 điểm lưới quanh bước nhảy |
| S4 ủng hộ | pivot = ngôn ngữ ít token nhất ở ≥ 2/3 seed trong mọi ô; bác (A3) nếu pivot = X khi X nhiều dữ liệu ở > 1/2 seed |

Run có twin kém mô hình chung > `TWIN_EXCESS_MAX` (0.10 nats) được đếm và báo riêng (ΔA có thể bị thổi phồng).

## Các lưới

| Lưới | Nội dung | Run (khác nhau) |
|---|---|---|
| `pilot_data` | decoder d=384, \|L\|=2, `DATA_FRAC` ∈ {0.125, 0.25, 0.5, 1.0} × 2 seed + twin | 24 |
| `p3` | decoder d ∈ {256, 384, 512} × \|L\| ∈ {2, 4, 8} × thứ tự {A, B} × 3 seed + 72 twin | 126 (6 trùng pilot) |
| `p3_families` | MoE, diffusion d=384 × \|L\| ∈ {2, 4, 8} × 3 seed + twin cùng họ | 66 |
| `s4` | decoder d=384, \|L\|=4 (A) × dữ liệu X {đều, ×2} × tokenizer X {tốn, rẻ} × 3 seed | 12 |
| `s2` | decoder d ∈ {128, …, 512} (7 cỡ) × \|L\|=4 × 10 seed, nửa dữ liệu | 70 |
| `holdout_p4` | d=320 × \|L\| ∈ {2, 4, 8} và \|L\|=6 × d ∈ {256, 384, 512}, 2 seed + twin | 64 (28 mới) |
| `smoke` | toy, mọi họ, tí hon (CPU) | 19 |

Tổng 320 run khác nhau. `run_id` không chứa tên lưới: cùng cấu hình ở hai lưới là cùng một run.

## Chạy

```bash
# 0. chạy thử cả đường ống trên dữ liệu toy (CPU, < 1 phút)
make study2b-smoke

# 1. dữ liệu + tokenizer (CPU, vài giờ; cần mạng, HF token cho FLORES gated)
make study2b-data

# 2. pilot chọn DATA_FRAC (GPU) -> đọc results/study2b/table_pilot.csv, sửa DATA_FRAC trong configs/data/natural.yaml
make study2b-pilot

# 3. các lưới chính — mỗi dòng một GPU; FAMILY / SHARD chia việc, khoá theo run tránh chạy trùng
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py study2b_grids status -o FAMILY=decoder -o SHARD=0/2
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py study2b_grids status -o FAMILY=decoder -o SHARD=1/2
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py study2b_grids status -o FAMILY=moe
CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py study2b_grids status -o FAMILY=diffusion

# 4. phân tích + hình + báo cáo
make study2b-report
```

`study2b_grids` chạy theo thứ tự `p3 → p3_families → s4 → s2 → holdout_p4`. Ước tính thô (A100, ~1 giờ / tỉ token ở
d=384): decoder ≈ 48 A100-giờ, MoE ≈ 12, diffusion ≈ 18; bố trí trên ≈ 1 ngày. Tốc độ thật nên đo trong pilot.

Một run giữ khoá `runs/<run_id>/.lock` khi đang chạy; nếu một tiến trình bị giết, xoá các `.lock` còn sót trước khi
chạy lại. Checkpoint được giữ lại: đổi cách đo (tăng `METRIC_VERSION`) chỉ đo lại, không huấn luyện lại.

## Đầu ra

```
outputs/study2b/
├── corpus/<lang>.txt, manifest.json          corpus (mỗi dòng một văn bản), byte premium, số văn bản bị loại do trùng FLORES
├── tokenizers/<khoá>/                        tokenizer HF + tokens/<lang>.bin (uint16) + .counts.npy
├── runs/<run_id>/meta.json, init.npz, trained.npz   (twin: biểu diễn Z_last / Z_mean của ngôn ngữ của nó)
└── checkpoints/                              MoE, diffusion
study3_natural/checkpoints/<tên>/             decoder (joint và twin_*), định dạng HF -> Study 4 tự đăng ký
results/study2b/                              report_study2b.md, hypotheses_summary.csv, table_*.csv, features_joint_runs.csv,
                                              table_pilot.csv, p4_inputs_for_study5.csv
paper/figures/study2b/                        fig_P3_deltaA_by_languages.pdf, fig_S2_alignment_by_width.pdf
```

## S3 trên mô hình 2b (qua Study 4)

Study 4 tự đăng ký mọi checkpoint decoder trong `{ROOT}/study3_natural/checkpoints/` (bỏ `twin_*`) và đo truy hồi theo
từ (M3) để kiểm chữ U ngược và r\* theo capacity. **Cần làm thêm trước khi tin kết quả này:** Study 4 đo trên 12 ngôn
ngữ của nó, còn mỗi mô hình 2b chỉ biết \|L\| ngôn ngữ; phân tích S3 cho mô hình 2b phải giới hạn ở các cặp ngôn ngữ
mà mô hình đã học (chưa cài trong Study 4).

## Các quyết định còn mở

1. `DATA_FRAC` — chốt bằng `pilot_data` (mức nhỏ nhất mà S_edge, A, ΔA ổn định và twin kém < 0.10 nats).
2. Thứ tự lồng và 8 ngôn ngữ (`NESTINGS` trong `configs/data/natural.yaml`).
3. Ngân sách S2 (10 seed × 7 cỡ): có thể thu hẹp số cỡ nếu thiếu GPU.
4. Môi trường hiện tại: `datasets` không import được (pyarrow cần NumPy ≥ 2, `pyproject.toml` ghim 1.26.4); cần sửa
   trên máy chạy thật để tải FineWeb-2 / FLORES. Mọi bước khác không cần `datasets`.
