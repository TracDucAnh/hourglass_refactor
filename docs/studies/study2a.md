# Study 2a — Ngôn ngữ tổng hợp (P1, P2, P3, S2–S6)

> Ghi chú gốc của `notebooks/archive/study-2a.ipynb` (chép nguyên văn các cell markdown, theo thứ tự).
> Mã tương ứng: `src/hourglass/study2_synthetic/`; chạy bằng `python -m hourglass stages study2a` để xem các bước.
> Lưu ý: đường dẫn `/mnt/cl/...` và tên "Cell A1, B2..." trong ghi chú là của notebook; trong repo, đường dẫn
> lấy từ `configs/` và mỗi cell là một file (xem header của file).

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
