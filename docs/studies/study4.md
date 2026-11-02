# Study 4 — Tần suất, loại hình (S3 trên văn bản tự nhiên)

> Ghi chú gốc của `notebooks/archive/study-4.ipynb` (chép nguyên văn các cell markdown, theo thứ tự).
> Mã tương ứng: `src/hourglass/study4_frequency/`; chạy bằng `python -m hourglass stages study4` để xem các bước.
> Lưu ý: đường dẫn `/mnt/cl/...` và tên "Cell A1, B2..." trong ghi chú là của notebook; trong repo, đường dẫn
> lấy từ `configs/` và mỗi cell là một file (xem header của file).

---

<!-- notebook cell 0 -->
# Study 4 — Từ nào, ngôn ngữ nào được chia sẻ ở eo của hourglass? (S3 trên văn bản tự nhiên)

**Câu hỏi.** Conjecture 2 (Zipf như khấu hao): căn chỉnh của một từ theo log-hạng tần suất là chữ **U ngược** (từ rất phổ biến mang mã riêng theo ngôn ngữ, từ hạng giữa được chia sẻ nhiều nhất, đuôi hiếm ước lượng kém), và đỉnh r* **dịch ra ngoài khi capacity tăng**. A2 (Platonic) dự đoán tăng đơn điệu. Ngoài ra: ngôn ngữ gần nhau về loại hình có căn chỉnh tốt hơn không (Mantel)?

**Quy trình (Algorithm 5).** tách từ + UPOS -> tần suất FineWeb-2 -> gióng hàng Eflomal + lọc MUSE -> biểu diễn mức từ (trung bình sub-token) -> R_k = MRR/CSLS cho từng từ ở từng tầng -> GAM(R_k ~ log-hạng + UPOS + độ dài + fertility + ngôn ngữ) -> đơn điệu vs unimodal -> Mantel.

| Cell | Nội dung | GPU |
|---|---|---|
| A1–A2 | config, registry model (core / sweep / 2b local) | |
| A3–A6 | dữ liệu, tách từ + UPOS, tần suất, gióng hàng + MUSE -> `ITEMS`, `NEED` | |
| B1–B2 | trích biểu diễn mức từ theo batch (ngân sách token), truy hồi CSLS theo tầng | x |
| C1–C2 | vòng chạy có resume (`runs/{model}/retrieval.npz`) | x |
| C3–C7 | bảng từ, GAM + bootstrap theo lô, shape CV, Mantel, capacity, kết luận | bootstrap nhanh hơn trên GPU |
| D1–D2 | hình (PDF, quy ước Study 1) | |

**Lưu ý trước khi chạy.** Các mục `[XM]` (tên dataset/model, mã FineWeb-2, URL MUSE, mã Stanza) viết theo trí nhớ, cần xác minh. Cell A5 (đếm tần suất) và A6 (Eflomal) chạy lâu nhưng chỉ một lần và có cache. Sau khi có kết quả, các cell C3–D2 chỉ đọc `study4/analysis/`.

---
### Bản sửa v2 (rà soát phương pháp)
| # | Vấn đề | Sửa |
|---|---|---|
| 1 | CSLS tính trên pool **instance**: từ phổ biến có hàng trăm bản sao nên bị coi là hub và bị phạt -> tự tạo "nhánh trái" | `HUB_MODE="type"`: mỗi loại từ chỉ được đếm một lần trong láng giềng (A1, B2). `"instance"` = hành vi cũ, `"cos"` = cosine thường |
| 2 | `amp` luôn ≥ 0 kể cả với đường đơn điệu | `amp` = đỉnh − max(hai đầu) (≈ 0 khi đơn điệu); công thức cũ giữ ở `amp_mean_ends` |
| 3 | `p_LM` không phải Lind–Mehlum, độ dốc đo bằng hai điểm biên + clip dồn điểm ở biên; `p_monotone_up` đo ngược hướng A2 | độ dốc = OLS trên 20% mỗi đầu; `p_LM` = max hai p một phía; thêm `p_left_up/p_right_down/...`, `shape`; bỏ clip (cắt bằng loại bỏ); `A2_refuted` = đầu trái tăng theo hạng |
| 4 | r\* được báo cáo cả khi không có đỉnh | cột `peak_ok`; slope capacity tính thêm trên các model có đỉnh, kèm Kendall p và leave-one-out |
| 5 | Hạng bằng nhau xếp theo bảng chữ cái | hạng trung bình cho nhóm bằng nhau (A5) |
| 6 | Từ unseen nhận hạng giả V+1; hiệu ứng thành phần cặp ngôn ngữ | `DROP_UNSEEN`, covariate `pmix` (độ dễ trung bình của các cặp mà từ tham gia), `MAIN_PAIRS="muse"` (chỉ cặp xác nhận bằng MUSE), `AXIS` |
| 7 | `n_inst` đếm hàng (instance × cặp); `n_pool`/waist bị cặp nhỏ (Dice) chi phối | `n_inst` đếm instance thật, thêm `n_pairs`; `n_pool` và waist chỉ dùng cặp ≥ `MIN_PAIR_ITEMS` |
| 8 | Chẩn đoán MUSE / hub | cell A7 `muse_coverage()`, cell E1 `diag_hub_modes()` |

**Chạy lại:** kết quả truy hồi cũ bị đánh dấu `stale` (theo `retr_version`/`hub_mode`), `run_all()` sẽ trích lại biểu diễn và tính lại. Chạy lại cell A5 (Cell 9) để cập nhật hạng.


---
### Bản v3 (tận dụng dữ liệu)
| # | Trước | Nay |
|---|---|---|
| 1 | Chỉ FLORES devtest (1012 câu) được ĐO; dev (997) + NTREX (1997) chỉ dùng để học Eflomal rồi **bỏ** liên kết | `MEASURE_SETS`: cả ba nguồn gộp thành một corpus (~4000 câu); thêm tập con `src_*` để kiểm tra lặp lại theo miền |
| 2 | Tần suất: 20M từ/ngôn ngữ (cap docs 100k không bao giờ chạm) | 200M từ (100M cho zh/ja/th/vi), cắt hapax ngoài corpus đo để tiết kiệm RAM; in tỉ lệ unseen mỗi ngôn ngữ |
| 3 | `POOL_N=500`, `POOL_REPS=10` | `1000`, `20` (cặp nhỏ nhất giờ có nhiều mục hơn) |
| 4 | Cache cũ | `RUN_TAG="_full"`, `FREQ_DIR` có hậu tố `FW_TAG`, `RETR_VERSION=3` -> không lẫn kết quả cũ |

Đặt `MEASURE_SETS = ["flores"]` để quay về hành vi cũ. Cache biểu diễn Qwen3-8B tăng ~4x (đặt `KEEP_WORD_CACHE=False`, cần ~200GB đĩa tạm).

---
### Bản v4 (làm sạch gióng hàng / từ điển / độ bền theo ngôn ngữ)
| # | Vấn đề | Sửa |
|---|---|---|
| 1 | MUSE `zh-en` là phồn thể (體, 球隊) còn corpus giản thể -> độ phủ zho 0.04–0.32 | `MUSE_ZH_T2S=True`: đổi MUSE zh sang giản thể khi nạp (OpenCC/zhconv) (A6) |
| 2 | fugashi cắt hình vị ngắn (使っ, 読み, まし) còn MUSE dùng dạng gốc -> độ phủ jpn top-100 chỉ 0.12 | `ALT_KEY_LANGS`: tra từ điển thêm bằng `orthBase`/`lemma` (A4, A6); đơn vị từ, tần suất, gióng hàng, biểu diễn KHÔNG đổi |
| 3 | MUSE không có Swahili (403) -> 11 cặp rơi xuống Dice; UPOS swh = UNK 100% (cộng tuyến với hiệu ứng ngôn ngữ) | Từ điển thay thế (`ALT_DICT`: tệp `raw/dict_alt/sw-en.txt` hoặc tự dựng từ Wiktionary/kaikki.org), nhãn nguồn `alt`; **tầng chính** (`MAIN_SRCS`) tách khỏi **tầng mở rộng**; chiếu UPOS qua căn chỉnh (Cell A6b) |
| 4 | Không có kiểm tra kết luận có phụ thuộc vào jpn / zho / swh hay không | Leave-one-language-out, bỏ cả 3 ngôn ngữ cùng lúc, tầng mở rộng riêng, Mantel chính vs mở rộng, bảng độ bền (Cell C5b) |

Cell mới: **A6b** (chiếu UPOS), **A7b** (chẩn đoán từ điển zho/jpn/swh), **C5b** (bảng độ bền). Kết quả truy hồi cũ tự bị coi là `stale` khi ITEMS đổi (`items_sig`), nên `run_all()` sẽ tính lại.
**Thứ tự chạy lại:** pip (cell 1) -> A1 -> ... -> A6 -> `build_items` (không chạy lại Eflomal, chỉ lọc lại) -> A6b -> A7/A7b -> C2 (`run_all`) -> C3...C7.

---
### Bản v5 (rà soát độ tin cậy: sửa lỗi đo, kiểm định, bootstrap, capacity, sanity, bảo mật)
| # | Vấn đề (đã xác nhận trong code) | Sửa |
|---|---|---|
| A | B2 chỉ che ứng viên CÙNG dạng đích với đáp án; các bản dịch hợp lệ khác (the→le thì la, les) vẫn là "đối thủ" -> từ chức năng/phổ biến nhiều dạng dịch có MRR thấp, tự tạo "đầu trái tăng" | `csls_mrr(which="both")`: thêm mặt nạ `acc` (mọi cặp nguồn→đích đã gióng + xác nhận); lưu cả `RR_*` (strict) và `RR_*_V` (valid); `RR_VARIANT="valid"` là chính, `rr_strict` là đối chiếu; covariate `trans_ent`; chẩn đoán `diag_mask_effect` (E2) |
| B | GAM vẽ đã hiệu chỉnh (UPOS, độ dài, nsub, pmix, ngôn ngữ) còn chấm xám là trung bình thô -> hai thứ khác estimand | GAM `covars="none"` (chỉ spline) + `covars="upos_pmix"` chạy ở tầng waist, vẽ chồng lên chấm thô (D2), báo trong tập con `gam_bare` / `gam_upos_pmix` |
| C | Độ dốc đo trên 20% ngoài cùng của LƯỚI x (không phải dữ liệu), nơi P-spline dao động và có khối hapax chung hạng; tiêu chí `p_LM` không dùng `amp` | Thêm tiêu chí song song, áp dụng đồng loạt cho mọi model: `*_q` (đầu/đuôi theo phân vị dữ liệu `EDGE_Q`) và `p_LM_amp` (đỉnh cao hơn cả hai đầu). Tập con `no_hapax`. `diag_tie_block` (E2). **Tiêu chí cũ giữ nguyên để so sánh** |
| D | MUSE phủ giảm mạnh theo hạng, DROP_SAME_FORM bỏ mục dễ ở đuôi: hai lọc kéo đuôi ngược chiều | Cell A7c `selection_diag()`: tỉ lệ dạng từ sống sót theo ngôn ngữ x khoảng hạng sau từng bộ lọc (chẩn đoán; đổi bộ lọc cần chạy lại build_items + C2) |
| E | Bootstrap theo từng dòng như độc lập (cùng khái niệm ở 12 ngôn ngữ), λ cố định, p = 0.000 là sàn của B lần | `CLUSTER_BOOT`: bootstrap theo cụm khái niệm (`_concepts`, pivot tiếng Anh); `LAM_SENS` (tập con `lam_x0.1`, `lam_x10`); p cộng một `(1+k)/(B+1)` |
| F | 4 decoder; argmax dính biên (0.6B ở cận dưới lưới, 8B/4B có CI chạm cận trên); Kendall n=4 có p tối thiểu 0.083; `peak_ok_only` chỉ 1 model | `at_boundary`, `centroid` (thống kê không dính biên), `kendall_p_floor`, slope gộp có hiệu ứng cố định theo họ + `concordant_pairs`, tập con `interior_peak_only` / `peak_ok_amp_only` |
| G | `sanity_batch_vs_single` trả giá trị nhưng bị bỏ qua; Qwen3-8B (2.50%, max 0.21) và **Qwen3-4B** (1.16%, max 0.84) đều bị cảnh báo lệch; chỉ thử tiếng Anh | Sanity trên `SANITY_LANGS`, kết quả ghi vào `meta.json`; `SANITY_ON_FAIL="fp32_check"` tải bản fp32 và thử lại; `EXTRACT_DTYPE`; cột `numerics_flag`, `n_clip` trong bảng tổng hợp; `diag_numerics()` (E2) |
| H | `.round(3)` làm `d_mono_vs_gam` ~1e-5 hiện thành 0.0; trục y mỗi panel một khác | `fmt_sci` (dạng khoa học có dấu); `plot_s3_curves(sharey=True)` + ghi range của GAM trên mỗi panel |
| S | Notebook chứa HF token ghi thẳng trong code (A1) | Gỡ token; chỉ đọc `HF_TOKEN` từ biến môi trường. **Thu hồi token cũ trên Hugging Face** |

**Phải chạy lại GPU:** `RETR_VERSION` 3 -> 4 (và kết quả lưu thêm `RR_*_V`, `meta.json` thêm `mask_valid`, `dtype`, `sanity`), nên mọi model bị đánh dấu `stale`; chạy lại A1 -> ... -> C2 (`run_all`) -> C3...C7 -> D1-D2 -> A7c, E2. Output còn lại trong notebook này của các cell kết quả đã bị xoá vì không còn khớp code.


---
### Bản v7 (mở rộng từ điển xác nhận để giữ lại từ hiếm ở đuôi)
**Vấn đề.** Bộ lọc xác nhận bằng MUSE đã chạy riêng từng cặp, nhưng MUSE thiên về từ phổ biến và chỉ có dạng gốc/dạng thường gặp. Đuôi tần suất của fin/tur/rus/arb/hin chủ yếu là dạng biến cách nên bị loại dù lemma có trong từ điển; cặp không có tiếng Anh (X–Y) còn phải qua pivot nên độ phủ ≈ tích hai độ phủ.

| Bước | Thay đổi | Nơi |
|---|---|---|
| 1 | **Lemma Stanza làm khoá tra từ điển** cho `LEMMA_LANGS` (cả tiếng Anh: `houses`→`house`). Chỉ đổi KHOÁ TRA; đơn vị từ, tần suất, gióng hàng, biểu diễn không đổi | A1, A6a (`ensure_lemmas`), A6 (`lookup_keys`) |
| 2 | **Hợp thêm Wiktionary (kaikki.org) cho mọi ngôn ngữ**, trước đây chỉ swh. Trích xuất mới: mục chỉ là dạng biến cách / chính tả khác (`form_of`, `alt_of`) được thừa hưởng chú giải của lemma thay vì bị bỏ; bỏ sense `misspelling`; zh đổi giản thể; vi giữ từ ghép `_`; tải theo luồng (tệp vài GB) | A1, A6 (`_kaikki_to_muse`, `wikt_for`, `prefetch_wikt`) |
| 3 | **Kiểm chứng** (cell A7d): độ phủ đuôi trước/sau, và độ ĐÚNG của các mục thêm bằng tín hiệu độc lập với từ điển (khớp lớp từ UPOS giữa hai từ được gióng, so với mức ngẫu nhiên) | A7d |

**Bậc xác nhận từng mục** (`ITEMS["tier"]`): 0 = từ điển gốc (= tập v5, khoá bề mặt [+ khoá phụ jpn]) · 1 = chỉ xác nhận khi thêm lemma · 2 = chỉ xác nhận khi thêm Wiktionary · 3 = Dice (không từ điển). `ITEM_MAX_TIER` (A1) giới hạn bậc khi dựng ITEMS; `word_table(..., max_tier=0)` và tập con `tier0_items` (C5) cho phép so với tập v5 ngay trên cùng kết quả truy hồi.

**Phải chạy lại.** `ensure_lemmas()` (Stanza, một lần), `prefetch_wikt()` (tải một lần), rồi `build_items` (KHÔNG chạy lại Eflomal) -> A6b -> A7d (kiểm chứng) -> C2 (`run_all`: ITEMS đổi nên `items_sig` đổi, mọi model bị coi là `stale`) -> C3...C7. Để dựng lại từ điển swh bằng cách trích xuất mới, đặt `WIKT_FORCE_REBUILD=True` một lần.

<!-- notebook cell 28 -->
---
### Bản v6 (kiểm định hình dạng không còn bị chi phối bởi điều kiện đuôi; đường cong tập con được lưu)
| # | Vấn đề thấy trong kết quả v5 | Sửa |
|---|---|---|
| 1 | `p_LM` đòi đồng thời: nhánh trái tăng, đỉnh trong thân, nhánh phải giảm, đo độ dốc ở 20% ngoài cùng của lưới x. Nhánh phải rơi vào vùng spline cong ngược / khối hapax: XLM-R large có `p_left_up`=0.000, đỉnh 733–1195 rất chắc, nhưng `p_right_down`=0.56 -> bị gán "không phải U ngược"; 5/6 model có nhánh trái tăng nhưng chỉ 1 model được gọi là U ngược | **C4b/C7c:** đối chiếu `rise = f(x̂) − f(xL)`, `fall = f(x̂) − f(xR)` với cửa sổ theo phân vị dữ liệu, quét lưới `LEFT_QS x RIGHT_QS`; báo độ lớn + CI + tỉ lệ cửa sổ có p<0.05; nhãn chỉ để đọc. Đỉnh dính biên = bị cắt |
| 2 | Kendall với n=4 có p_min = 0.083 (không bao giờ < 0.05); họ encoder chỉ 2 model bị bỏ; `outward` dùng CI độc lập giữa các model | **C7c `capacity_paired`:** bootstrap GHÉP CẶP (cùng mẫu từ/cụm), độ dịch đỉnh theo cặp cùng họ, slope gộp trong họ, độ nhạy theo cửa sổ |
| 3 | `robust_to_lang_choice` so nhãn chữ nên p=0.04 vs 0.09 đảo nhãn; tập con không lưu đường cong | **C5** lưu đường cong (quan sát + bootstrap + mốc phân vị) cho MỌI tập con; **C7c `robustness_curves`** báo tỉ lệ tập con còn tăng / giảm |
| 4 | Chưa tách ảnh hưởng của kích thước pool theo cặp (3.591–31.768 ứng viên; `pmix` đo độ dễ ở n=1000, không đo kích thước) | Thêm tập con `pool_small_pairs` / `pool_large_pairs` (tam phân vị theo số mục hợp lệ của cặp) ở C5. Đây là kiểm tra, CHƯA phải sửa; sửa đúng là pool phân tầng theo hạng, cần chạy lại B2 |
| 5 | Hình vẽ cả vùng đuôi nơi đường GAM cong ngược lên | **D3** vẽ đậm đến xR, phần sau xám/nét chấm |

**Thứ tự chạy lại (không cần GPU, không chạy lại C2):** C4b -> C5 (analyze_model, lưu thêm đường cong) -> C7c -> D3. Với npz v5 cũ, C7c vẫn chạy cho 'all' và 'gam_bare' (phần tập con cần C5 mới).
**Không dùng làm kết luận:** `S3_supported`, `robust_to_lang_choice`, `kendall_p` (cell cũ giữ để đối chiếu).
**Còn nợ:** Qwen3-4B / 8B có cảnh báo số học bf16 (sanity lệch 1.16% / 2.5%): kiểm lại bằng fp32 (`SANITY_ON_FAIL="fp32_check"`) trước khi diễn giải hai model này; pool phân tầng theo hạng (B2) chưa viết.

<!-- notebook cell 43 -->
## Chẩn đoán bổ sung
