# Kiến trúc mã: workspace theo study

## Vì sao không phải module Python thông thường

Ba notebook gốc chạy trong một kernel: hàm ở cell sau đọc biến toàn cục do cell trước tạo ra. Ví dụ Study 4:
`build_items()` (cell A6) tạo `ITEMS`, rồi `run_model`, `word_table`, `analyze_model`… đọc `ITEMS`, `WT`, `TID` như
biến toàn cục; Study 1/2a: mọi hàm đọc `DEVICE`, `MODELS`, `STUDY_DIR`… Tách thành module có import tường minh buộc
phải đổi chữ ký hàm (truyền state qua tham số) — tức là sửa mã nghiên cứu, rủi ro đổi kết quả.

Giải pháp: mỗi study là một package gồm nhiều file nhỏ theo chức năng (data / models / training / metrics /
analysis / plotting / stages), **nạp vào một namespace chung** theo đúng thứ tự cell (`hourglass/workspace.py`):

```python
from hourglass.study4_frequency import load
ws = load("study4")                 # nạp mọi lib (định nghĩa + state dựng từ cache)
ws.run_stage("s3")                  # chạy một bước; ws.S3_ALL, ws.word_table(...) dùng được như trong notebook
```

- `lib`: định nghĩa, luôn nạp theo thứ tự. `gpu=False` bỏ qua các lib cần torch (đủ cho phân tích Study 1, 2a; đủ
  cho phần chuẩn bị dữ liệu Study 4).
- `stage`: hành động (chạy lưới, vẽ, chẩn đoán), chạy khi được yêu cầu, sau khi mọi lib đã nạp.
- `python -m hourglass stages <experiment>` liệt kê các bước; header mỗi file ghi cell gốc tương ứng.

Các file trong `src/hourglass/study*/` **không import trực tiếp được** (tên đến từ file khác trong cùng namespace),
nên chúng được loại khỏi ruff / pre-commit.

## Quy tắc giữ nguyên logic

`tests/test_notebook_parity.py` so sánh (theo AST, bỏ qua comment/khoảng trắng) mọi câu lệnh cấp module của
package với code cell của `notebooks/archive/*.ipynb`. Chỉ cho phép ba khác biệt:

1. `X = param("X", mặc_định)` trong `config.py` của study — giá trị đọc từ `configs/`, mặc định = giá trị notebook;
2. `from hourglass.common.m import f` thay cho `def f` **giống hệt** nhau giữa các notebook (`common/`);
3. cell được khai báo bỏ trong `DROPPED_CELLS` (cài package, cell rỗng, ví dụ đã comment).

Muốn sửa logic một study: sửa file trong `src/hourglass/study*/`, rồi cập nhật notebook trong `notebooks/archive/`
hoặc ghi lý do — test parity sẽ báo. Khi mã đã ổn định có thể bỏ ràng buộc này và chuyển dần sang module thường.

## Config

`configs/config.yaml` → `defaults:` (`data/`, `model/`, `trainer/`) → `experiment/<tên>.yaml` → override dòng lệnh
`-o KEY=VALUE` (giá trị đọc như YAML). Khoá VIẾT HOA = tên hằng trong mã. `tests/test_config.py` bắt khoá thừa
(sai tên). Kiểu tuple/set của mặc định được giữ khi đọc list từ YAML.

Một số cỡ của Study 2a phụ thuộc `SMOKE` (`_S(full, smoke)`) nên không đặt trong YAML; muốn đổi thì override
(`-o BATCH=512`). Registry model (tên HF, final norm, batch) nằm trong `registry.py` của Study 1 / Study 4.

## Lưu trữ

| Thư mục | Nội dung | Git |
|---|---|---|
| `data/` | FLORES, NTREX, MUSE, Wiktionary… tải về + đã xử lý (dùng chung Study 1, 4) | không |
| `outputs/` | kết quả thô theo run (`.npz`, `meta.json`), thế giới tổng hợp, cache biểu diễn, đếm tần suất | không |
| `results/` | bảng CSV/JSON/MD tổng hợp nhỏ (`ANALYSIS_DIR`) | có (trừ `.npz/.npy/.pkl`) |
| `paper/figures/` | hình PDF (`FIG_DIR`) | có |

Đặt `HOURGLASS_ROOT=/mnt/cl` để mọi thứ nằm trên ổ khác. Để dùng lại kết quả cũ của notebook (ở `/mnt/cl/study1`…)
override thẳng đường dẫn: `-o STUDY_DIR=/mnt/cl/study1 -o DATA_DIR=/mnt/cl/data`.
