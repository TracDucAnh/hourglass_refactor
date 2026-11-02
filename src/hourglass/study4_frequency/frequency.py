# Tần suất từ FineWeb-2 -> hạng (Cell A5)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 8 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A5 — Tần suất từ (FineWeb-2) -> hạng r của từng dạng từ, theo từng ngôn ngữ
# Đếm một lần bằng CHÍNH bộ tách từ ở A4 (nên "từ" ở FLORES và ở FineWeb-2 là cùng một đơn vị).
# FineWeb-2 không có tiếng Anh -> lấy từ HuggingFaceFW/fineweb. [XM] tên cấu hình theo mã ngôn ngữ.
# ============================================================
import multiprocessing as mp

FW_SRC = {"eng_Latn": ("HuggingFaceFW/fineweb", "sample-10BT"),
          "fra_Latn": ("HuggingFaceFW/fineweb-2", "fra_Latn"), "rus_Cyrl": ("HuggingFaceFW/fineweb-2", "rus_Cyrl"),
          "hin_Deva": ("HuggingFaceFW/fineweb-2", "hin_Deva"), "zho_Hans": ("HuggingFaceFW/fineweb-2", "cmn_Hani"),
          "vie_Latn": ("HuggingFaceFW/fineweb-2", "vie_Latn"), "tur_Latn": ("HuggingFaceFW/fineweb-2", "tur_Latn"),
          "arb_Arab": ("HuggingFaceFW/fineweb-2", "arb_Arab"), "jpn_Jpan": ("HuggingFaceFW/fineweb-2", "jpn_Jpan"),
          "fin_Latn": ("HuggingFaceFW/fineweb-2", "fin_Latn"), "swh_Latn": ("HuggingFaceFW/fineweb-2", "swh_Latn"),
          "tha_Thai": ("HuggingFaceFW/fineweb-2", "tha_Thai")}

def count_words_chunk(args):
    """Worker: (lang, [văn bản]) -> Counter các dạng từ chuẩn hoá (cùng bộ tách từ như A4)."""
    lang, docs = args
    c = Counter()
    for d in docs:
        for w, _, _ in segment_text(lang, d):
            n = norm_word(w)
            if not _ALL_NUM.match(n):
                c[n] += 1
    return c

def count_words_stream(lang, text_iter, max_words, chunk_docs=1000, workers=FW_WORKERS, max_docs=None):
    """Đếm tần suất từ từ một iterator văn bản; chạy song song theo khối tài liệu; dừng khi đủ max_words / max_docs."""
    total, n_docs, tot_tokens = Counter(), 0, 0
    def chunks():
        nonlocal n_docs
        buf = []
        for t in text_iter:
            buf.append(t); n_docs += 1
            if len(buf) >= chunk_docs:
                yield (lang, buf); buf = []
            if max_docs and n_docs >= max_docs:
                break
        if buf:
            yield (lang, buf)
    if workers <= 1:
        it = (count_words_chunk(a) for a in chunks()); pool = None
    else:
        pool = mp.get_context("fork").Pool(workers)
        it = pool.imap(count_words_chunk, chunks())
    try:
        for c in it:
            total.update(c); tot_tokens += sum(c.values())
            if tot_tokens >= max_words:
                break
    finally:
        if pool is not None:
            pool.terminate()
    return dict(counts=total, n_tokens=int(tot_tokens), n_docs=int(n_docs))

def fw_text_iter(lang):
    from datasets import load_dataset
    path, name = FW_SRC[lang]
    ds = load_dataset(path, name, split="train", streaming=True, cache_dir=HF_CACHE_DIR)
    return (ex["text"] for ex in ds)

def freq_path(lang):
    return f"{FREQ_DIR}/{SHORT[lang]}.pkl"

FW_KEEP = {}          # {lang: tập dạng từ cần giữ cả khi count == 1} = mọi dạng từ trong corpus đo (được gán ở cell tần suất)

def build_freq(lang, force=False):
    p = freq_path(lang)
    if os.path.exists(p) and not force:
        return
    t0 = time.time()
    res = count_words_stream(lang, fw_text_iter(lang), FW_WORDS_OVERRIDE.get(lang, FW_MAX_WORDS_PER_LANG), max_docs=FW_MAX_DOCS_PER_LANG)
    keep, c = FW_KEEP.get(lang, set()), res["counts"]
    res["n_extra_c1"] = sum(1 for w, v in c.items() if v == 1 and w not in keep)          # hapax không thuộc corpus đo: chỉ cần số lượng
    res["counts"] = {w: v for w, v in c.items() if v >= 2 or w in keep}
    res["src"] = FW_SRC[lang]
    with open(p + ".tmp", "wb") as f:
        pickle.dump(res, f)
    os.replace(p + ".tmp", p)
    print(f"  [freq] {SHORT[lang]}: {res['n_docs']} tài liệu, {res['n_tokens'] / 1e6:.1f}M từ, {len(c)} dạng từ ({time.time() - t0:.0f}s)")

class RankTable:
    """Hạng tần suất (1 = phổ biến nhất; hạng bằng nhau lấy trung bình) của dạng từ trong một ngôn ngữ. Dạng từ chưa từng thấy -> hạng V+1 (seen=False; bị loại khỏi phân tích chính nếu DROP_UNSEEN).
    n_extra_c1: số dạng từ đếm đúng 1 lần đã bị cắt khỏi `counts` để tiết kiệm bộ nhớ (chúng chỉ nằm trong nhóm hạng bằng nhau cuối cùng nên không ảnh hưởng hạng của từ có count >= 2)."""
    def __init__(self, counts, n_tokens, n_extra_c1=0):
        order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        self.rank, n, i = {}, len(order), 0
        while i < n:                                   # hạng bằng nhau -> HẠNG TRUNG BÌNH của nhóm (không xếp theo bảng chữ cái)
            j, c = i, order[i][1]
            while j < n and order[j][1] == c:
                j += 1
            r = (i + 1 + j + (n_extra_c1 if c == 1 else 0)) / 2.0
            for t in range(i, j):
                self.rank[order[t][0]] = r
            i = j
        self.count = counts
        self.V, self.n_tokens = n + n_extra_c1, n_tokens
    def log_rank(self, w):
        r = self.rank.get(w)
        return (math.log10(r), True) if r is not None else (math.log10(self.V + 1), False)
    def zipf(self, w):
        """log10(tần suất trên triệu từ) — thang Zipf, dùng cho kiểm tra độ bền."""
        c = self.count.get(w, 0)
        return math.log10((c + 0.5) / self.n_tokens * 1e6)

RANKS = {}
def load_ranks(langs=None):
    for l in (langs or LANGS):
        if l in RANKS:
            continue
        with open(freq_path(l), "rb") as f:
            d = pickle.load(f)
        RANKS[l] = RankTable(d["counts"], d["n_tokens"], d.get("n_extra_c1", 0))
