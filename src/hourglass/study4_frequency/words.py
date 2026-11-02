# Tách từ + UPOS (Cell A4)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 6 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A4 — Tách từ + UPOS
# Một bộ tách từ DUY NHẤT cho mọi thứ (đếm tần suất FineWeb-2, gióng hàng Eflomal, đo biểu diễn) để "từ" luôn là cùng một đơn vị:
#   eng/fra/rus/hin/tur/arb/fin/swh : khoảng trắng + dấu câu (regex Unicode, giữ dấu kết hợp \p{M} cho Hindi/Thai/Ả Rập)
#   zho_Hans: jieba | jpn_Jpan: fugashi (unidic-lite) | tha_Thai: PyThaiNLP newmm | vie_Latn: pyvi (từ ghép nhiều âm tiết, nối "_")
# UPOS: Stanza (huấn luyện trên UD) chạy trên các từ đã tách sẵn (tokenize_pretokenized). Ngôn ngữ không có tagger -> "UNK".
# ============================================================
import regex

_WORD_RE   = regex.compile(r"[\p{L}\p{M}\p{N}]+")
_HAS_CHAR  = regex.compile(r"[\p{L}\p{M}\p{N}]")
_ALL_NUM   = regex.compile(r"^\p{N}+$")
_WS_RE     = regex.compile(r"\s+")
_SEG_STATE = {}

def _spans_from_tokens(text, toks):
    """Khôi phục (surface, start, end) từ danh sách token liên tiếp (các tokenizer không trả offset)."""
    out, pos = [], 0
    for t in toks:
        if not t or not t.strip():
            continue
        j = text.find(t, pos)
        if j >= 0:
            out.append((t, j, j + len(t))); pos = j + len(t); continue
        parts = t.split()
        if len(parts) > 1:                                    # từ ghép nhiều âm tiết, khoảng trắng gốc có thể khác
            m = regex.compile(r"\s+".join(regex.escape(p) for p in parts)).search(text, pos)
            if m:
                out.append((m.group(), m.start(), m.end())); pos = m.end()
    return out

def _need(pkg, hint):
    raise ImportError(f"Thiếu thư viện '{pkg}' cho bộ tách từ ({hint}). Cài: %uv pip install {hint}")

def _seg_regex(text):
    return [(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]

def _seg_jieba(text):
    if "jieba" not in _SEG_STATE:
        try: import jieba
        except ImportError: _need("jieba", "jieba")
        jieba.setLogLevel(60); _SEG_STATE["jieba"] = jieba
    return [(w, s, e) for w, s, e in _SEG_STATE["jieba"].tokenize(text)]

def _get_fugashi():
    if "fugashi" not in _SEG_STATE:
        try: import fugashi
        except ImportError: _need("fugashi", "fugashi unidic-lite")
        _SEG_STATE["fugashi"] = fugashi.Tagger()
    return _SEG_STATE["fugashi"]

def _seg_fugashi(text):
    return _spans_from_tokens(text, [w.surface for w in _get_fugashi()(text)])

def _alt_fugashi(text):
    """Khoá tra từ điển PHỤ cho từng token (cùng thứ tự và cùng bộ lọc như segment_text): tuple (orthBase, lemma) đã chuẩn hoá.
    fugashi/unidic-lite cắt theo hình vị ngắn (使っ, 読み, まし, た) còn mục từ MUSE là từ/dạng gốc (使う, 読む); UniDic cho lemma dạng 'ソフトウェア-software' -> lấy phần trước '-'.
    Chỉ dùng để XÁC NHẬN bằng từ điển; không thay đổi từ là gì (tần suất/gióng hàng/biểu diễn vẫn theo dạng bề mặt)."""
    out = []
    for w in _get_fugashi()(text):
        sf = w.surface
        if not sf or not sf.strip() or not _HAS_CHAR.search(sf):
            continue
        ft, keys = w.feature, []
        for nm in ("orthBase", "lemma"):
            v = getattr(ft, nm, None)
            if v and v != "*":
                v = norm_word(v.split("-")[0])
                if v and v not in keys:
                    keys.append(v)
        out.append(tuple(keys))
    return out

ALT_KEYS = {"jpn_Jpan": _alt_fugashi}      # lang -> hàm(text) -> [tuple khoá phụ] cho mỗi token; bật bằng ALT_KEY_LANGS (Cell A1)

def _seg_thai(text):
    if "thai" not in _SEG_STATE:
        try: from pythainlp.tokenize import word_tokenize
        except ImportError: _need("pythainlp", "pythainlp")
        _SEG_STATE["thai"] = word_tokenize
    return _spans_from_tokens(text, _SEG_STATE["thai"](text, engine="newmm", keep_whitespace=False))

def _seg_vi(text):
    if "vi" not in _SEG_STATE:
        try: from pyvi import ViTokenizer
        except ImportError: _need("pyvi", "pyvi")
        _SEG_STATE["vi"] = ViTokenizer
    toks = [t.replace("_", " ") for t in _SEG_STATE["vi"].tokenize(text).split(" ")]
    return _spans_from_tokens(text, toks)

SEGMENTER = {l: _seg_regex for l in LANGS}
SEGMENTER.update({"zho_Hans": _seg_jieba, "jpn_Jpan": _seg_fugashi, "tha_Thai": _seg_thai, "vie_Latn": _seg_vi})

def norm_word(surf):
    """Dạng chuẩn của một từ: NFKC + chữ thường; khoảng trắng trong từ ghép -> '_' (Eflomal/MUSE dùng token không có khoảng trắng)."""
    return _WS_RE.sub("_", unicodedata.normalize("NFKC", surf).strip().lower())

def segment_text(lang, text):
    """[(surface, start, end)] — chỉ giữ token có ít nhất một chữ/dấu kết hợp/chữ số (bỏ dấu câu, khoảng trắng)."""
    return [(w, s, e) for w, s, e in SEGMENTER[lang](text) if _HAS_CHAR.search(w)]

def _build_alt(lang, sents, ptr):
    """Khoá phụ (xem _alt_fugashi) căn theo từng token của bảng từ; câu nào lệch số token thì bỏ khoá phụ ở câu đó."""
    f, out, n_bad = ALT_KEYS[lang], [], 0
    for i, s in enumerate(sents):
        n = int(ptr[i + 1] - ptr[i]); a = f(s)
        if len(a) != n:
            a = [()] * n; n_bad += 1
        out.extend(a)
    if n_bad:
        print(f"  [warn] khoá phụ {lang}: {n_bad} câu lệch số token -> bỏ khoá phụ ở các câu đó")
    return out

def build_wordtab(lang, sents):
    """Bảng từ phẳng: mỗi từ có chỉ số toàn cục g; câu i gồm các g trong [ptr[i], ptr[i+1])."""
    surf, start, end, ptr = [], [], [], [0]
    for s in sents:
        for w, a, b in segment_text(lang, s):
            surf.append(w); start.append(a); end.append(b)
        ptr.append(len(surf))
    norm = [norm_word(w) for w in surf]
    alt = _build_alt(lang, sents, ptr) if (lang in ALT_KEY_LANGS and lang in ALT_KEYS) else None
    return dict(lang=lang, surf=surf, norm=norm, start=np.asarray(start, np.int32), end=np.asarray(end, np.int32),
                ptr=np.asarray(ptr, np.int64), is_num=np.asarray([bool(_ALL_NUM.match(w)) for w in norm], bool),
                upos=None, seg=SEGMENTER[lang].__name__, alt=alt)

# ---- UPOS ----
STANZA_LANG = {"eng_Latn": "en", "fra_Latn": "fr", "rus_Cyrl": "ru", "hin_Deva": "hi", "zho_Hans": "zh-hans",
               "vie_Latn": "vi", "tur_Latn": "tr", "arb_Arab": "ar", "jpn_Jpan": "ja", "fin_Latn": "fi",
               "swh_Latn": None, "tha_Thai": None}      # [XM] Stanza không có Swahili; Thai dùng PyThaiNLP (bên dưới)
UPOS_TAGS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "SCONJ", "SYM", "VERB", "X", "UNK"]

def _tag_stanza(lang, wt):
    import stanza
    code = STANZA_LANG[lang]
    stanza.download(code, processors="tokenize,pos", verbose=False)
    nlp = stanza.Pipeline(code, processors="tokenize,pos", tokenize_pretokenized=True, use_gpu=True,
                          pos_batch_size=5000, verbose=False)
    ptr = wt["ptr"]; N = len(ptr) - 1
    inp = [wt["surf"][ptr[i]:ptr[i + 1]] or ["."] for i in range(N)]
    doc = nlp(inp)
    up = ["UNK"] * len(wt["surf"])
    for i, sent in enumerate(doc.sentences):
        ws = [w.upos for w in sent.words]
        if ptr[i + 1] - ptr[i] == len(ws):
            up[ptr[i]:ptr[i + 1]] = [u if u in UPOS_TAGS else "X" for u in ws]
    return up

def _tag_thai(wt):
    from pythainlp.tag import pos_tag
    ptr = wt["ptr"]; up = ["UNK"] * len(wt["surf"])
    for i in range(len(ptr) - 1):
        ws = wt["surf"][ptr[i]:ptr[i + 1]]
        if ws:
            tg = pos_tag(ws, corpus="orchid_ud")              # nhãn kiểu UD (NOUN, VERB, ...)
            up[ptr[i]:ptr[i + 1]] = [t if t in UPOS_TAGS else "X" for _, t in tg]
    return up

def tag_upos(lang, wt):
    try:
        if STANZA_LANG.get(lang):
            wt["upos"] = _tag_stanza(lang, wt)
        elif lang == "tha_Thai":
            wt["upos"] = _tag_thai(wt)
        else:
            wt["upos"] = ["UNK"] * len(wt["surf"])
    except Exception as e:
        print(f"  [warn] UPOS {lang} lỗi ({repr(e)[:140]}) -> 'UNK'")
        wt["upos"] = ["UNK"] * len(wt["surf"])
    return wt

def get_wordtab(split, lang, sents, with_upos=False):
    """Cache pickle ở WORK_DIR/words. split in {'test','dev','ntrex'}; UPOS chỉ cần cho devtest ('test')."""
    p = f"{WORK_DIR}/words/{split}_{SHORT[lang]}.pkl"
    if os.path.exists(p):
        with open(p, "rb") as f:
            wt = pickle.load(f)
        if len(wt["ptr"]) - 1 == len(sents) and (not with_upos or wt["upos"] is not None):
            if lang in ALT_KEY_LANGS and lang in ALT_KEYS and wt.get("alt") is None:      # cache cũ chưa có khoá phụ: bổ sung, KHÔNG tách lại từ / gắn lại UPOS
                wt["alt"] = _build_alt(lang, sents, wt["ptr"])
                with open(p, "wb") as f:
                    pickle.dump(wt, f)
            return wt
    wt = build_wordtab(lang, sents)
    if with_upos:
        tag_upos(lang, wt)
    with open(p, "wb") as f:
        pickle.dump(wt, f)
    return wt
