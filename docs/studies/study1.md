# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)

> Ghi chú gốc của `notebooks/archive/study-1.ipynb` (chép nguyên văn các cell markdown, theo thứ tự).
> Mã tương ứng: `src/hourglass/study1_open_models/`; chạy bằng `python -m hourglass stages study1` để xem các bước.
> Lưu ý: đường dẫn `/mnt/cl/...` và tên "Cell A1, B2..." trong ghi chú là của notebook; trong repo, đường dẫn
> lấy từ `configs/` và mỗi cell là một file (xem header của file).

---

<!-- notebook cell 0 (comment) -->
# Study 1 — Hourglass ở 4 họ mô hình (M1–M4, $J_k$, random-init) — bản đầy đủ

Theo Algorithm 1 của *"Principle of Least Effort in Depth"* và kế hoạch chạy LE-D.

**Phạm vi.** 15 mô hình mở (3 encoder + ByT5 mức byte, 6 dense decoder, 3 MoE, 2 diffusion), dữ liệu **FLORES-200 devtest** (12 ngôn ngữ) và **NTREX-128** (lặp lại ở một tập con mô hình), cùng random-init control cho từng kiểu kiến trúc.

**Số đo tính cho mỗi (model, layer, cặp ngôn ngữ):** $D_k$, $G_k$ (M1), CKA không chệch và Procrustes-CV $A_k$ (M2), MRR truy hồi CSLS $R_k$ (M3), usable information $U_k$ (M4, probe tuyến tính), và $J_k$ (JS routing so với English) cho MoE.

**Cấu trúc notebook**

| Phần | Nội dung | Cần GPU |
|---|---|---|
| A | config, registry model, load/xử lý FLORES + NTREX | không |
| B | trích xuất biểu diễn + tính số đo | có |
| C | chạy thí nghiệm (có resume), bảng trạng thái | có |
| D | bootstrap CI, bảng đặc trưng waist, vẽ hình PDF (chỉ đọc `/mnt/cl`) | không |

**Lưu trữ (`/mnt/cl`)**
```
/mnt/cl/data/raw/ntrex128/*.txt                  # file thô NTREX
/mnt/cl/data/processed/parallel/{flores,ntrex}.jsonl   # {id, lang, text}, NFKC
/mnt/cl/study1/{dataset}/{run_id}/{variant}.npz  # số đo chi tiết (per pair, per layer, per content)
/mnt/cl/study1/{dataset}/{run_id}/meta.json      # cấu hình, thống kê, trạng thái
/mnt/cl/study1/analysis/*.csv|json               # bảng tổng hợp
/mnt/cl/study1/figures/*.pdf                     # hình
```
`run_id` = `key` (pretrained) hoặc `key__random` (random-init). `variant` = `base` (mọi họ trừ diffusion) hoặc `t{mức che}_{all|masked}` (diffusion).

**Chạy lại / resume.** Mỗi (dataset, model, variant) đã có file `.npz` sẽ được bỏ qua. Chỉ chạy một số model: đặt `RUN_MODELS`. Chạy lại từ đầu: `FORCE_RERUN = True`. Sau khi chạy xong, ở kernel mới chỉ cần chạy phần A (cell config + registry) và phần D để vẽ lại hình.

**Gợi ý thứ tự (cổng G1):** `RUN_MODELS = ["xlmr_base", "qwen3_1.7b", "olmoe"]` rồi `run_all(["flores"])` trước khi chạy cả 15 model.
