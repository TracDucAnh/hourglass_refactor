# data/ (không commit)

Dữ liệu tải về và đã xử lý, dùng chung Study 1 và Study 4 (`DATA_DIR` trong `configs/config.yaml`):

```
raw/ntrex128/            NTREX-128 (GitHub MicrosoftTranslator/NTREX)
raw/muse/, raw/dict_alt/ từ điển MUSE, Wiktionary (kaikki.org) — Study 4
processed/parallel/      {flores,flores_dev,ntrex}.jsonl  {id, lang, text}, NFKC
```

Tự tạo khi chạy (`make study1-g1`, `make study4-prepare`). FLORES-200 là dataset gated trên Hugging Face: đặt
`HF_TOKEN` trong biến môi trường trước khi chạy. Dùng DVC hoặc ổ ngoài (`HOURGLASS_ROOT`) nếu cần chia sẻ.
