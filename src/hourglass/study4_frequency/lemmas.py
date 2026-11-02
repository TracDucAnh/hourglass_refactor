# Lemma Stanza làm khoá tra từ điển (Cell A6a)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 12 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A6a — (v7, bước 1) LEMMA bằng Stanza -> khoá tra từ điển phụ (chạy sau A4/A4b, TRƯỚC build_items)
# Vì sao: đuôi tần suất của fin/tur/rus/arb/hin chủ yếu là dạng biến cách (taloissamme, evlerimizde, ...) còn từ điển (MUSE) chỉ có dạng gốc/dạng thường gặp
#   -> bị loại dù lemma có trong từ điển. Tiếng Anh cũng cần (to_en('eng') = chính từ đó: 'houses' khớp 'house' khi thêm lemma).
# Cách làm: chạy lại Stanza (tokenize_pretokenized, như UPOS ở A4) với processor lemma trên CÙNG token -> wt['lemma'][g] (chuỗi rỗng nếu = dạng bề mặt / không có).
#   Chỉ dùng làm KHOÁ TRA từ điển (lookup_keys). Đơn vị từ, tần suất, gióng hàng, biểu diễn mô hình KHÔNG đổi. Lemma theo NGỮ CẢNH từng token (Stanza dùng UPOS).
#   Lưu vào chính cache words/*.pkl nên chỉ chạy một lần mỗi ngôn ngữ (WT nạp lại từ cache sẽ có sẵn lemma).
# ============================================================
def _arb_lemma(x):
    return regex.sub(r"\p{Mn}", "", regex.sub(r"_\d+$", "", x))                  # Stanza ar: 'كِتاب_1' -> bỏ số phân biệt nghĩa và dấu (từ điển/corpus không dấu)
LEMMA_POST = {"arb_Arab": _arb_lemma}

def _lemmas_stanza(lang, wt):
    import stanza
    code = STANZA_LANG[lang]
    stanza.download(code, processors="tokenize,pos,lemma", verbose=False)
    nlp = stanza.Pipeline(code, processors="tokenize,pos,lemma", tokenize_pretokenized=True, use_gpu=True, pos_batch_size=5000, verbose=False)
    ptr = wt["ptr"]; N = len(ptr) - 1
    doc = nlp([wt["surf"][ptr[i]:ptr[i + 1]] or ["."] for i in range(N)])
    post = LEMMA_POST.get(lang)
    out, n_bad = [""] * len(wt["surf"]), 0
    for i, sent in enumerate(doc.sentences):
        n = int(ptr[i + 1] - ptr[i])
        if n == 0:
            continue
        if n != len(sent.words):
            n_bad += 1; continue
        for j, w in enumerate(sent.words):
            lm = w.lemma
            if not lm or lm == "_":
                continue
            lm = norm_word((post(lm) if post else lm).replace("#", ""))               # '#' = ranh giới từ ghép trong lemma tiếng Phần Lan (nối lại)
            if lm and lm != wt["norm"][ptr[i] + j]:
                out[ptr[i] + j] = lm
    return out, n_bad

def ensure_lemmas(langs=None, force=False):
    """Tính lemma cho các ngôn ngữ trong LEMMA_LANGS chưa có (hoặc force), ghi vào cache words/{MEAS}_*.pkl. Lỗi Stanza -> cảnh báo, ngôn ngữ đó chỉ dùng khoá bề mặt."""
    for l in (langs or [x for x in LANGS if x in LEMMA_LANGS]):
        wt = WT[MEAS][l]
        if wt.get("lemma") is not None and len(wt["lemma"]) == len(wt["norm"]) and not force:
            continue
        if not STANZA_LANG.get(l):
            print(f"  [lemma] {SHORT[l]}: không có Stanza -> chỉ khoá bề mặt"); continue
        t0 = time.time()
        try:
            lem, n_bad = _lemmas_stanza(l, wt)
        except Exception as e:
            print(f"  [warn] lemma {SHORT[l]} lỗi ({repr(e)[:140]}) -> chỉ khoá bề mặt"); wt["lemma"] = None; continue
        wt["lemma"] = lem
        keep_ = dict(wt)
        if l in UPOS_PROJECTED:                      # đừng ghi UPOS chiếu (A6b) vào cache: A6b cần thấy UNK để chiếu lại mỗi phiên
            keep_["upos"] = ["UNK"] * len(wt["norm"])
        p = f"{WORK_DIR}/words/{MEAS}_{SHORT[l]}.pkl"
        with open(p + ".tmp", "wb") as f:
            pickle.dump(keep_, f)
        os.replace(p + ".tmp", p)
        share = float(np.mean([bool(x) for x in lem])) if lem else 0.0
        print(f"  [lemma] {SHORT[l]}: {share:.1%} số token có lemma khác dạng bề mặt | {n_bad} câu lệch số từ (bỏ lemma ở câu đó) | {time.time() - t0:.0f}s")
