"""Chuẩn hoá văn bản song song (NFKC, bỏ content rỗng ở bất kỳ ngôn ngữ nào)."""

import unicodedata


def norm_text(s):
    return unicodedata.normalize("NFKC", s).strip()


def clean_parallel(raw):
    """raw: {lang: [text,...]} cùng độ dài. NFKC + bỏ các content mà BẤT KỲ ngôn ngữ nào rỗng."""
    lens = {l: len(v) for l, v in raw.items()}
    if len(set(lens.values())) != 1:
        raise ValueError(f"Số câu không khớp giữa các ngôn ngữ: {lens}")
    n = next(iter(lens.values()))
    normed = {l: [norm_text(x) for x in v] for l, v in raw.items()}
    keep = [i for i in range(n) if all(normed[l][i] for l in normed)]
    if len(keep) < n:
        print(f"  [clean] bỏ {n - len(keep)} content có câu rỗng ở ít nhất một ngôn ngữ")
    return {l: [normed[l][i] for i in keep] for l in normed}, keep
