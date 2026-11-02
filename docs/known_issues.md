# Vấn đề phát hiện khi tổ chức lại (chưa sửa — cần tác giả quyết định)

Mã được giữ nguyên văn từ notebook, nên các điểm dưới đây vẫn còn trong `src/`.

## 1. Study 1: run mới của mọi model không phải MoE báo lỗi `UnboundLocalError: JS`

`src/hourglass/study1_open_models/runner.py`, hàm `run_one` (Cell C1 của notebook):

```python
                if ext["router"] is not None and v == vs[0]:
                    JS, JS_raw, js_langs, rmean = router_js(ext["router"], list(texts))
                    nl = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "num_layers", 0)
                arrays.update(JS=JS, JS_raw=JS_raw, JS_langs=js_langs, JS_layers=np.array(ext["gate_layers"]),
                              router_mean=rmean, n_layers=np.array(int(nl)))
```

`arrays.update(...)` nằm ngoài khối `if`, nên với encoder / dense / diffusion (không có router) biến `JS` chưa được
gán. Đã xác nhận bằng cách chạy `run_one` với model giả: run kết thúc với trạng thái `error` và `error.txt` ghi
`UnboundLocalError`. Output lưu trong notebook không lộ lỗi này vì mọi run đều `skipped` (kết quả có sẵn từ phiên
bản mã trước). **Sửa đề xuất:** thụt `arrays.update(...)` (2 dòng) vào trong khối `if`.

## 2. Study 4 tải lại model Stanza mỗi lần nạp

Lib `stanza_models.py` (cell 9) gọi `stanza.download(...)` ở mức module, nên mỗi lần `load("study4")` cần mạng
(Stanza bỏ qua tệp đã có nhưng vẫn kiểm tra index). Giữ nguyên như notebook.

## 3. Phiên bản thư viện

Notebook Study 4 chạy `pip install -U transformers datasets huggingface_hub` (bản mới nhất) trong khi Study 1 pin
`transformers==4.57.1`, `datasets==4.0.0`, `huggingface_hub==0.36.0`. Repo dùng một môi trường với pin của Study 1
(`uv.lock`); đường GPU của Study 4 chưa được chạy lại với các bản pin này.

## 4. Khác biệt có chủ đích so với notebook (không đổi logic)

- Đường dẫn lấy từ `configs/` thay vì hard-code `/mnt/cl` (Study 4) hay `os.getcwd()` (Study 1). Biến môi trường
  `LED_ROOT` (Study 2a) được thay bằng `HOURGLASS_ROOT`.
- Mọi lib được nạp trước khi chạy stage. Trong notebook, `pilot_report()` (Cell C5b) dùng `plt` vốn chỉ được import
  ở Cell D4 phía sau; trong repo hàm này luôn chạy được.
- Cell 11 của Study 1 (ví dụ đã comment + `show_status()`) không chuyển thành stage riêng — dùng stage `status`.
