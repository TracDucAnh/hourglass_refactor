# Study 2a — Ngôn ngữ tổng hợp (P1, P2; S3 bổ trợ)

> Ghi chú gốc của `notebooks/archive/study-2a.ipynb` (chép nguyên văn các cell markdown, theo thứ tự).
> Mã tương ứng: `src/hourglass/study2_synthetic/`; chạy bằng `python -m hourglass stages study2a` để xem các bước.
> Lưu ý: đường dẫn `/mnt/cl/...` và tên "Cell A1, B2..." trong ghi chú là của notebook; trong repo, đường dẫn
> lấy từ `configs/` và mỗi cell là một file (xem header của file).

## Phạm vi (v3)

Study 2a chỉ kiểm **ba** giả thuyết trên ngôn ngữ tổng hợp, không so sánh trực tiếp với Study 1:

| Giả thuyết | Vai trò | Câu hỏi | Dữ liệu | Tiêu chí (`analysis/hypotheses.py`) |
|---|---|---|---|---|
| **P1** | chính | Cả 4 họ đều có waist? | `p1p2`, `moe_diff` (overlap 0 %, M4, LM+SEM; các mức / tác vụ khác là mô tả) | **ước lượng qua seed**, không bỏ phiếu theo run: ΔS = S_edge(trained) − S_edge(init) với CI95, độ sâu tương đối S/G\*, vị trí ρ\*; có eo ⇔ CI95 của ΔS > 0 và ρ\* nằm trong (0, 1). Nhãn mô tả như Bảng 9 của paper, theo cận dưới CI95 của S_edge: rõ ≥ 0,20 / vừa 0,10–0,20 / nông 0,05–0,10 / sát ngưỡng < 0,05 (chỉ để đọc). Tỉ lệ run qua tiêu chí từng run của Study 1 (ngưỡng 0,05) chỉ để đối chiếu |
| **P2** | chính | Waist đi theo tác vụ (LM / SEM / SURF)? | `p1p2`, `moe_diff` (overlap 0 %; kết luận ở M4 cho cả bốn họ, M0/M2 là mô tả) | **ước lượng qua seed** như P1: hiệu số ghép theo seed giữa các tác vụ (SEM−LM, SEM−SURF, SURF−LM) của độ sâu S_edge, đỉnh G\*, vị trí ρ\*, độ rộng w, với CI95. Độ sâu đổi ⇔ CI của hiệu S_edge không chứa 0 (ít nhất một cặp); vị trí dịch ⇔ CI của hiệu ρ\* không chứa 0 và \|hiệu\| > 0,10; eo cố định ⇔ mọi CI của hiệu ρ\* trong ±0,10 và độ sâu không đổi |
| **S3** | bổ trợ | Căn chỉnh tại waist theo tần suất khái niệm là chữ U ngược? | tập dò S3 của các run `p1p2`, `moe_diff` (overlap 0 %) | **nới**: độ cong trung bình theo seed < 0 (Wilcoxon p < 0,05) và ≥ 50 % run cong xuống với đỉnh nằm trong khoảng hạng; không kiểm đỉnh dịch theo capacity. G theo bin dùng **đường cơ sở xáo trong từng bin** (khoá `G_rank_w`) |
| D_anchor | chẩn đoán | Eo nông ở decoder / MoE có do thiếu **điểm neo** chung giữa các ngôn ngữ? | `mask_ablation`, `anchors`, so với `p1p2` (LM, M4) cùng seed | xem mục *Chẩn đoán điểm neo*; không phải giả thuyết của paper, P1/P2/S3 không dùng các lưới này |

Đã bỏ khỏi 2a: P3 (sinh đôi đơn ngữ), P4, S2, S4 (logit-lens), S5, S6 (mức che diffusion), cánh `balanced`, cùng các lưới
`balanced`, `p3s2`, `s4`, `s5`, `heldout_p4`. Các cell tương ứng vẫn còn trong notebook gốc nhưng đã được sửa / bỏ cho
khớp với mã (`logit_lens` là cell 24 trong `DROPPED_CELLS`).

**Biểu diễn câu theo họ.** Họ hai chiều (encoder, diffusion): trung bình sub-token theo từ rồi trung bình các từ. Họ causal
(decoder, MoE): trạng thái ở **token nội dung cuối** — vị trí duy nhất đã thấy cả câu. (Trung bình theo từ với họ causal
trộn các tiền tố khác nhau khi trật tự từ khác nhau — SVO / SOV / VSO ở M2, M4 — nên thiên lệch có lợi cho họ hai chiều.)
Tập dò S3 vẫn dùng vector của từ danh từ chủ ngữ cho mọi họ.

Họ causal lưu **thêm** số đo theo trung bình theo từ (`init_mean.npz`, `trained_mean.npz`) để đối chiếu cách đo. Hình P1 có hai bản:
`fig_P1_four_families_lasttoken.pdf` (decoder / MoE theo token cuối) và `fig_P1_four_families_meanpool.pdf` (decoder / MoE theo
trung bình theo từ); cột encoder / diffusion giống nhau ở hai bản. Bảng P1 / P2 vẫn dùng cách đo chính (token cuối cho họ causal).
Run decoder / MoE đã chạy trước thay đổi này không có file `_mean` (không có checkpoint nên phải huấn luyện lại):
`scripts/train.py study2a_decoder status -o 'RUN_GRIDS=[p1p2]' -o FORCE_RERUN=true` và
`scripts/train.py study2a_moe status -o FORCE_RERUN=true` (54 + 18 run).

## Chẩn đoán điểm neo (`mask_ablation`, `anchors`)

Ở v3, mọi run decoder / MoE đều có hình eo (đỉnh G nằm giữa mạng, vượt `init`) nhưng **nông** (S_edge ≈ 0,02 ở LM,
≈ 0,06 ở SEM) so với encoder / diffusion (0,12–0,20), nên trượt ngưỡng 0,05. Giả thuyết: eo cần **điểm neo** — token mà
mọi ngôn ngữ cùng đi qua. Ngôn ngữ tổng hợp overlap 0 % gần như không có (chỉ BOS / EOS); encoder / diffusion có `[MASK]`
chung ở đầu vào; nhãn SEM chung làm eo decoder sâu gấp ~3 lần; văn bản thật có rất nhiều (số, tên riêng, từ mượn, ...).

| Nhánh | Thay đổi (chỉ một yếu tố) | Dự đoán nếu giả thuyết đúng |
|---|---|---|
| `mask_ablation`, encoder, `mlm_corrupt=shuffle` | 80 % token bị chọn thay bằng token nội dung ngẫu nhiên **của chính câu đó** thay cho `[MASK]` | eo **nông** hơn |
| `mask_ablation`, decoder, `in_mask=0.15` | 15 % token nội dung **đầu vào** thay bằng `[MASK]`; mục tiêu (token kế tiếp) giữ nguyên | eo **sâu** hơn |
| `anchors`, decoder, `anchor=0.05 / 0.2` | 5 % / 20 % danh từ (trải đều theo hạng) có gốc từ **giống hệt ở mọi ngôn ngữ**, viết bằng bảng chữ chung 3 phụ âm × 3 nguyên âm — như tên riêng / số; phụ tố vẫn theo từng ngôn ngữ | eo **sâu** hơn, tăng theo tỉ lệ |

Tất cả: LM, M4, overlap 0 %, d = 192, 6 seed (24 run); đối chứng là run LM, M4 cùng họ, cùng seed trong `p1p2` (cùng thế giới
trừ các danh từ neo). `test_anchor` so S_edge từng nhánh với đối chứng (Wilcoxon một phía ghép theo seed); kết quả ở dòng
`D_anchor` của báo cáo và `table_D_anchor.csv`. Bộ sinh có điểm neo vẫn qua cổng G0 (round-trip đơn ánh, không ngôn ngữ
khác phân tích được câu: câu nào cũng có động từ riêng của ngôn ngữ đó). `anchor = 0` cho đúng thế giới cũ.

## Chạy theo từng họ mô hình

Mỗi họ có một experiment riêng; chạy mỗi họ trong một tiến trình / GPU. Cả bốn ghi chung
`outputs/study2a_pool1600000/` (worlds/, runs/), nên chỉ phân tích **một lần** sau khi cả bốn xong. Ngân sách theo họ đặt
ở `SEQ_MULT_FAMILY` trong `configs/experiment/study2a.yaml`, nên chạy gộp (`make study2a-all`, experiment `study2a_grids`)
cũng dùng đúng ngân sách đó.

| Họ | Experiment | Lưới | Ngân sách (`SEQ_MULT_FAMILY`, epoch trên pool 1,6M) | Lệnh |
|---|---|---|---|---|
| decoder | `study2a_decoder` | gate_g2 (3) + p1p2 (54) + mask_ablation (6) + anchors (12) = 75 run | ×2 = 6 000 bước ≈ 3,8 epoch | `make study2a-decoder` |
| encoder | `study2a_encoder` | p1p2 (54) + mask_ablation (6) = 60 run | ×4 = 12 000 bước ≈ 7,7 epoch | `make study2a-encoder` |
| MoE | `study2a_moe` | moe_diff (18 run) | ×3 = 9 000 bước ≈ 5,8 epoch | `make study2a-moe` |
| diffusion | `study2a_diffusion` | moe_diff (18 run) | ×5 = 15 000 bước ≈ 9,6 epoch | `make study2a-diffusion` |

```bash
# mỗi dòng một terminal / tmux pane; train.py chạy run_grids rồi in bảng status của họ đó
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py study2a_decoder   status
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py study2a_encoder   status
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py study2a_moe       status
CUDA_VISIBLE_DEVICES=3 uv run python scripts/train.py study2a_diffusion status

# sau khi cả bốn xong: đặc trưng + kiểm định + hình + báo cáo (results/study2a/, paper/figures/study2a/)
make study2a-report        # = uv run python scripts/make_figures.py study2a
```

**Cân tải.** Tính theo tổng số bước huấn luyện, encoder nặng nhất (60 × 12 000 ≈ 720 nghìn bước), decoder ≈ 450 nghìn
(75 × 6 000), MoE + diffusion ≈ 430 nghìn (18 × 9 000 + 18 × 15 000). Bố trí cân hơn: chia encoder làm 2 shard bằng `SHARD=k/n`
(mỗi shard nhận run thứ k, k+n, ... của mỗi lưới; các shard không trùng nhau) và chạy MoE rồi diffusion nối tiếp:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train.py study2a_decoder status
CUDA_VISIBLE_DEVICES=1 uv run python scripts/train.py study2a_encoder status -o SHARD=0/2
CUDA_VISIBLE_DEVICES=2 uv run python scripts/train.py study2a_encoder status -o SHARD=1/2
CUDA_VISIBLE_DEVICES=3 bash -c 'for f in moe diffusion; do uv run python scripts/train.py study2a_$f status; done'
```
`n` phải giữ nguyên trong suốt đợt chạy (đổi `n` giữa chừng thì các shard chia khác nhau, có thể chạy trùng run).
Theo dõi tiến độ của một họ: `uv run python scripts/train.py study2a_<họ> status` (run đã xong được bỏ qua).

**Thay đổi so với v2** (vì sao phải chạy lại từ đầu):
1. **Ngân sách theo họ.** v2 dùng chung ×2 cho mọi họ (chốt bằng `pilot_conv` chỉ trên decoder): encoder 0/54 và
   diffusion 1/18 run hội tụ. v3 đặt `SEQ_MULT` riêng cho từng họ (`SEQ_MULT_FAMILY` trong `configs/experiment/study2a.yaml`)
   (mọi họ vẫn chạy trọn lịch lr cosine, không early stopping). Các giá trị cho encoder / MoE / diffusion là ước lượng
   (chưa có pilot riêng): sau khi chạy, kiểm lại cột `converged` / `conv_drop`.
2. **Token cuối cho họ causal** (xem trên). `METRIC_VERSION = 3`.
3. **Thu hẹp phạm vi** còn P1, P2, S3 (xem trên); P1 tách thêm theo tác vụ (ở v2, decoder và MoE không có waist ở LM
   nhưng có ở SEM / SURF — kết luận gộp LM+SEM che mất điều này); S3 dùng tiêu chí nới.
4. **An toàn khi chạy song song.** Thế giới (`worlds/`) được ghi bằng tên tạm riêng từng tiến trình rồi đổi tên
   nguyên tử: nhiều họ dựng cùng một thế giới cùng lúc (vd. decoder và encoder của `p1p2`) không làm hỏng file.

**Chạy lại từ đầu / ghi đè.** `TRAIN_VERSION = 3` và `METRIC_VERSION = 3`: mọi run cũ (v2) tự động bị **huấn luyện
lại và ghi đè** trong `runs/`, và bị loại khỏi phân tích; không cần `FORCE_RERUN` (`FORCE_RERUN=true` sẽ chạy lại cả
run v3 đã xong khi khởi động lại tiến trình). Run v2 của các lưới đã bỏ (`p3s2`, `s4`, ...) nằm lại trong `runs/` nhưng
không được đọc (khác `train_version`). Repo không lưu checkpoint mô hình, chỉ lưu số đo. `worlds/` là dữ liệu tất định
(cùng cấu hình ⇒ cùng nội dung), được dùng lại; muốn dựng lại hẳn thì xoá `outputs/study2a_pool1600000/worlds/` trước
khi chạy. `results/study2a/` và `paper/figures/study2a/` bị ghi đè khi chạy `make study2a-report` (file cũ của các giả
thuyết đã bỏ, vd. `table_P3.csv`, không tự xoá).

**Thêm chẩn đoán điểm neo vào đợt v3 đã chạy.** Thêm lưới / khoá mới không đổi `run_id` hay phiên bản của run cũ: chạy lại
các lệnh theo họ ở trên thì run đã xong được bỏ qua, chỉ 24 run chẩn đoán được huấn luyện (decoder 18, encoder 6).
Riêng **S3** cần khoá `G_rank_w` (đường cơ sở trong bin) mà run cũ không có, nên S3 báo *chưa có dữ liệu* cho tới khi
`p1p2` / `moe_diff` được chạy lại (`-o FORCE_RERUN=true`, tức huấn luyện lại 144 run vì không có checkpoint).

---

<!-- notebook cell 1 -->
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)

Theo mục 5 / Study 2a của *"Principle of Least Effort in Depth"* và kế hoạch chạy LE-D (≈50 A100-giờ cho toàn lưới).

**Ý tưởng.** Giả định (G) của LE-D (nội dung độc lập với ngôn ngữ, mỗi ngôn ngữ dịch đơn ánh, văn bản lộ ngôn ngữ) chỉ đúng *theo cách xây dựng* trên ngôn ngữ tự tạo từ một "chương trình ngữ nghĩa" duy nhất. Notebook này (1) sinh các ngôn ngữ tổng hợp với luật rõ ràng và **kiểm tra (G) bằng parser nghịch**, (2) huấn luyện mô hình tí hon (K = 8 tầng) ở bốn họ (decoder, encoder-MLM, MoE, diffusion) với ba tác vụ (LM / SEM / SURF), (3) đo D_k, G_k, CKA, A_k (Procrustes-CV), U_k (usable information) theo tầng, và (4) kiểm định các giả thuyết P1, P2, P3, S2, S3, S4, S5, S6 bằng chính số đo đó.

**Cấu trúc notebook**

| Phần | Nội dung | Cần GPU |
|---|---|---|
| A | config, đường dẫn, kích thước lưới | không |
| S | **bộ sinh ngôn ngữ tổng hợp**: nội dung C (PCFG), ngôn ngữ (từ vựng, hình thái, trật tự từ), renderer, parser nghịch, tokenizer BPE, dựng "thế giới" (corpus + tập đo) | không |
| T | **testcase cho bộ sinh**: luật, thống kê, round-trip, mơ hồ, tokenizer (cổng G0) | không |
| B | mô hình 4 họ, tác vụ, huấn luyện, trích biểu diễn, số đo M1–M4, logit-lens | có |
| C | định nghĩa các lưới thí nghiệm, chạy (có resume), cổng G2 | có |
| D | đặc trưng waist, kiểm định giả thuyết, hình PDF, báo cáo kết quả (chỉ đọc `STUDY_DIR`) | không |

**Lưu trữ (`/mnt/cl/study2a`)**
```
worlds/{world_id}.npz + .json           # corpus tokenized, tập đo, tokenizer, fertility (dùng chung cho mọi family/task/d cùng thế giới)
runs/{run_id}/meta.json                 # cấu hình, loss, fertility, thời gian, trạng thái
runs/{run_id}/{init|trained|t0.25|...}.npz   # số đo theo tầng (G, D, CKA, A, U, g_content, G_rank ...)
runs/{run_id}/twins.npz, lens.npz       # A_indep / CKA_indep của cặp sinh đôi; khối lượng logit-lens (pivot)
analysis/*.csv|json|md                  # bảng đặc trưng, bảng kiểm định, report
figures/*.pdf                           # hình
```
`run_id` = `{grid}__{family}_{level}_{task}_d{d}_L{L}_ov{overlap}_ls{len_scale}_sh{share}_s{seed}`.

**Chạy lại / resume.** Mỗi run đã có `meta.json` với `status = ok` sẽ bị bỏ qua. Chọn lưới bằng `RUN_GRIDS`; chạy lại từ đầu: `FORCE_RERUN = True`. Ở kernel mới chỉ cần chạy Cell A1 rồi các cell Phần D (D1–D7) để vẽ lại hình và báo cáo; Phần D không phụ thuộc torch hay bộ sinh.

**Thứ tự cổng:** G0 (Phần T, không cần GPU) → `RUN_GRIDS = ["gate_g2"]` (một cấu hình decoder M4/SEM, 3 seed: phải có U₀/log L ≈ 1 và có eo; đo thời gian thực rồi nhân lại ngân sách) → các lưới còn lại. `SMOKE = True` để chạy thử cả đường ống với kích thước cực nhỏ trước.

**Các quyết định thiết kế mà kế hoạch để ngỏ (đã ghi trong comment ở đúng chỗ dùng)**
- d cố định = 192 cho lưới P1/P2, cánh cân bằng, MoE/diffusion, S4, S5; sinh đôi chạy ở d ∈ {64, 128, 256, 384}.
- Nhãn SEM = hạng tần suất của vị ngữ gốc, cắt ở 50 lớp; nhãn SURF = (độ dài từ thứ hai tính theo ký tự, ký tự đầu của nó).
- Tập huấn luyện là một *pool* câu: mỗi nội dung chỉ được viết bằng **một** ngôn ngữ (không có dữ liệu song song, giống π = 0).
- $G_k$ dùng kỳ vọng **chính xác** của khoảng cách khi ghép ngẫu nhiên (công thức đóng) thay cho 1 000 lần xáo; test T3 so khớp hai cách.
- Pivot (S4) đo bằng logit-lens trên decoder: khối lượng xác suất rơi vào token của từng ngôn ngữ ở tầng giữa.

**Lưu ý.** Mọi kết luận chỉ "ở quy mô nhỏ, trong ngôn ngữ tổng hợp"; kết quả âm chưa bác bỏ LE-D ở quy mô lớn.

<!-- notebook cell 2 -->
## Phần A — Config, đường dẫn, kích thước lưới

<!-- notebook cell 4 -->
### Giao thức huấn luyện v2 (sửa để hội tụ)

**Chẩn đoán (cổng G2 cũ).** Chỉ 750 bước cập nhật (batch rất lớn, 7,7 epoch): loss LM ≈ 5,3 (eval ≈ 5,05), trong khi sàn entropy của ngôn ngữ tổng hợp này chỉ ≈ 4,3 nats/token
(= (H(C) + log L) / số token đích mỗi câu; `entropy_floor` ở Cell B2). Train ≈ eval ⇒ mô hình **bị giới hạn bởi tối ưu hóa**, không phải bởi dữ liệu.
Cổng cũ "đạt" nhưng k\* = 1 (đỉnh G ngay sau khối đầu), nên chưa thể coi là waist.

**Sửa.**
1. Giữ cố định số chuỗi nhưng **giảm batch → tăng số lần cập nhật** (mặc định 1024 × 6000 bước; `SEQ_MULT = 2`). lr đỉnh nhân căn bậc hai theo batch, warmup tỉ lệ theo số bước.
2. **Giám sát hội tụ trong mọi run** cho cả 4 họ: `val` (câu giữ ra theo R28) và `trn` (một phần pool) được đo ~12 lần/run với cùng mặt nạ ngẫu nhiên cố định;
   ghi `history`, `conv_drop` (val còn giảm bao nhiêu trong ~1/6 bước cuối), `val_trn_gap`, và khoảng cách tới sàn entropy (decoder/MoE).
3. **Không dùng early stopping** làm quy trình chính: mọi điều kiện dùng cùng ngân sách để so sánh được; tiêu chí hội tụ được ghi lại và kiểm bằng lưới **`pilot_conv`**
   (cùng tổng chuỗi với nhiều batch khác nhau + 2×, 4× tổng chuỗi). Tăng **số mẫu mới** chỉ cần khi `val_trn_gap > OVERFIT_TOL`.
4. **Sinh đôi** (P3) không còn bị cắt còn `STEPS // L` bước: số bước twin = `min(STEPS, max(STEPS // L, sàn, TWIN_MAX_EPOCH · N_l / BATCH))`; `twins.npz` và `meta.json` lưu loss của twin so với mô hình chung
   (`twin_excess_loss` > 0 nghĩa là twin kém hơn ⇒ ΔA có thể bị thổi phồng).
5. Cổng G2 cần thêm **hội tụ** và **k\* ≥ `GATE_MIN_K`**. Run cũ (`train_version` ≠ 2) tự động bị chạy lại và bị loại khỏi Phần D.

Quy trình: `RUN_GRIDS = ["pilot_conv"]` → Cell C5b chọn ngân sách → sửa `BATCH`/`SEQ_MULT` ở A1 → `["gate_g2"]` → các lưới còn lại.

<!-- notebook cell 5 -->
## Phần S — Bộ sinh ngôn ngữ tổng hợp

Hai tầng tách bạch, đúng như X = g_Λ(C) trong lý thuyết:
1. **Nội dung C** (ngôn ngữ-độc-lập): cây ngữ nghĩa sinh bởi PCFG ≈ 30 luật (R01–R28). Khái niệm lấy mẫu theo Zipf α = 1.
2. **Renderer g_ℓ** (đặc thù ngôn ngữ): từ vựng (khái niệm → chuỗi âm tiết, độ dài ∝ −log tần suất × `len_scale`), trật tự từ (SVO/SOV/VSO), hướng đầu, phụ tố tiền/hậu tố, cách và hòa hợp (luật T01–T10). Mỗi từ ghi lại nút C gốc nên gióng hàng từ chính xác tuyệt đối.

Thang khác biệt giữa các ngôn ngữ: **M0** chỉ từ vựng; **M2** thêm trật tự từ + vị trí phụ tố; **M4** thêm hướng đầu, cách, hòa hợp. Overlap âm vị: **0 %** (mỗi ngôn ngữ một bảng chữ riêng, giao diện lộ ngôn ngữ) hoặc **100 %** (mọi ngôn ngữ dùng chung bảng âm tiết, "cánh cân bằng giao diện", không nhãn ngôn ngữ).

(G) được kiểm bằng `Parser` (parser nghịch theo span, đếm số cây hợp lệ): G2 (đơn ánh) ⇔ đúng một cây và đó là C; G3 (văn bản lộ ngôn ngữ) ⇔ các ngôn ngữ khác không phân tích được chuỗi.

<!-- notebook cell 11 -->
## Phần T — Testcase cho bộ sinh ngôn ngữ tổng hợp (cổng G0, không cần GPU)

| Cell | Kiểm tra |
|---|---|
| T1 | ví dụ trực quan: cùng một nội dung C viết ra ở M0 / M2 / M4, overlap 0 % và 100 % |
| T2 | thống kê luật PCFG (R01–R28): tần suất khớp xác suất khai báo, Zipf, độ sâu, độ dài tối thiểu, tập giữ ra rời tập huấn luyện |
| T3 | luật thực hiện (T01–T10): từ vựng duy nhất + độ dài theo Zipf, bảng chữ theo overlap, thang M0 < M2 < M4, đơn ánh bằng va chạm chuỗi |
| T4 | **giả định (G)**: round-trip bằng parser (G2) và tỉ lệ câu mơ hồ / phân tích được bởi ngôn ngữ khác (G3), mọi cấp × overlap × seed; tất định theo seed |
| T5 | số đo: D = 0 / G = 1 cho hai biểu diễn trùng nhau, G ≈ 0 cho độc lập, công thức đóng của $\bar D^\pi$ khớp 1 000 lần xáo; tokenizer BPE |
| T6 | tổng kết PASS/FAIL (dừng lại nếu có FAIL trước khi tốn GPU) |

<!-- notebook cell 19 -->
## Phần B — Mô hình, tác vụ, huấn luyện, trích biểu diễn, số đo (cần GPU)

- **Bốn họ** (K = 8 tầng, d ∈ {64 … 384}, đầu chú ý ~32 chiều, BPE chung): `decoder` (kiểu Llama: RMSNorm, RoPE, SwiGLU, causal), `encoder` (cùng khối, hai chiều, MLM 15 %), `moe` (decoder, 8 expert top-2, khớp tham số kích hoạt), `diffusion` (hai chiều, che liên tục t ~ U(0,1), trọng số 1/t).
- **Ba tác vụ**: `LM` (chỉ mục tiêu ngôn ngữ), `SEM` (+ 20 % chuỗi dự đoán nhãn ngôn ngữ-độc-lập), `SURF` (+ 20 % chuỗi dự đoán nhãn bề mặt của từ thứ hai).
- **Biểu diễn**: residual stream sau mỗi khối (tầng 0 = embedding; tầng cuối lấy *trước* final norm), gộp trung bình sub-token theo từ rồi trung bình các từ.
- **Số đo**: $D_k$, $\bar G_k$ (kỳ vọng chính xác của khoảng cách ghép ngẫu nhiên), CKA không chệch, Procrustes-CV $A_k$, usable information $U_k$ (probe tuyến tính + MLP, chia nhóm theo nội dung), $G$ theo bin hạng danh từ (S3), logit-lens (pivot, S4).

<!-- notebook cell 25 -->
## Phần C — Lưới thí nghiệm, chạy (có resume), cổng G2

| Lưới | Nội dung | Số run |
|---|---|---|
| `gate_g2` | một cấu hình decoder M4/SEM, 3 seed (cổng G2) | 3 |
| `p1p2` | {decoder, encoder} × M{0,2,4} × {LM, SEM, SURF} × 6 seed (P1, P2) | 108 |
| `balanced` | decoder, overlap 100 % × M{0,4} × {LM, SEM} × 6 seed (cánh cân bằng giao diện, A1) | 24 |
| `moe_diff` | {MoE, diffusion} × M4 × 3 tác vụ × 6 seed (P1, S6) | 36 |
| `p3s2` | decoder M4/SEM, 6 giá trị d × L ∈ {2,4,8} × 10 seed; **sinh đôi đơn ngữ** ở d ∈ {64,128,256,384} và seed < 6 (P3, S2, S3) | 180 |
| `s4` | thiết kế 2×2: ngôn ngữ X chiếm 40 % hay đều dữ liệu × fertility X thấp/cao (`len_scale_X`), L = 4, 6 seed + logit-lens (S4) | 24 |
| `s5` | `len_scale` ∈ {0.6, 0.8, 1.0, 1.4, 2.0} × 6 seed (S5) | 30 |
| `heldout_p4` | 24 cấu hình giữ ra (L ∈ {3,6}, d ∈ {80,160,224}, `len_scale` chưa dùng) × 2 seed, có sinh đôi (đầu vào cho Study 5) | 48 |
| `smoke` | bốn họ, d = 64, L = 3: kiểm tra đường ống (dùng với `SMOKE = True`) | 5 |

Một run = {dựng/đọc thế giới → mô hình khởi tạo (đo `init`, null "chưa huấn luyện") → huấn luyện → đo `trained` (+ `t0.25/0.5/0.75` cho diffusion) → (tùy lưới) sinh đôi đơn ngữ, logit-lens}.
Seed điều khiển cả thế giới ngôn ngữ (từ vựng, typology) lẫn khởi tạo mô hình; mọi điều kiện ghép cặp theo seed.

<!-- notebook cell 34 -->
## Phần D — Phân tích, kiểm định giả thuyết, hình, báo cáo (không cần GPU)

Chỉ đọc `STUDY_DIR`; ở kernel mới chỉ cần chạy **Cell A1** rồi các cell của Phần D. Quy ước giữ nguyên Study 1:
- **Waist** (mỗi run): đỉnh của $\bar G_k$ nằm trong thân; cận dưới CI95 (bootstrap theo nội dung) của *edge depth* $S_{edge}=\max_k \bar G_k-\max(\bar G_0,\bar G_K)$ vượt 0,05; đỉnh CKA rơi vào dải waist của G (nới ±1 tầng); và vượt mô hình chưa huấn luyện (`init`, cùng kiến trúc/seed).
- $\rho^*=k^*/K$, độ rộng $w$ = tỉ lệ tầng có $\bar G_k\ge \bar G_{k^*}-0{,}25\,S$.
- **Đơn vị thống kê là seed** (mỗi seed = một thế giới ngôn ngữ độc lập, ghép cặp qua điều kiện): Wilcoxon chính xác ghép cặp, TOST cho dự đoán "không đổi" (biên $\Delta\rho^*=0{,}10$; $\Delta A=\max(0{,}05,\ 1\,SD_{seed})$), Jonckheere–Terpstra cho thứ tự, hiệu chỉnh Holm trong từng giả thuyết.
- Dải mờ trong hình = CI95 (t) qua seed.
