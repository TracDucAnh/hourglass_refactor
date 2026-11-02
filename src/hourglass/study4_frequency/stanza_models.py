# Tải model Stanza (tokenize, pos)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 9 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

import stanza
from concurrent.futures import ThreadPoolExecutor
_codes = [c for c in STANZA_LANG.values() if c]
def _dl(c): stanza.download(c, processors="tokenize,pos", verbose=False)
with ThreadPoolExecutor(4) as ex:
    list(ex.map(_dl, _codes))
