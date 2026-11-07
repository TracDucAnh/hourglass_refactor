# FLORES (đo), byte premium, corpus FineWeb-2 hoặc toy, lọc trùng 13-gram (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
from hourglass.common.io import read_json, read_jsonl, write_json, write_jsonl
from hourglass.common.text import clean_parallel, norm_text

# nguồn FineWeb (FineWeb-2 không có tiếng Anh -> FineWeb sample-10BT; zho lấy cmn_Hani như Study 4),
# trật tự từ cơ bản (WALS 81A; dùng để chia cặp cùng / khác trật tự trong phân tích P3), có dấu cách giữa từ hay không
LANG_INFO = {
    "eng_Latn": dict(fw=("HuggingFaceFW/fineweb", "sample-10BT"), order="SVO", spaced=True),
    "fra_Latn": dict(fw=("HuggingFaceFW/fineweb-2", "fra_Latn"), order="SVO", spaced=True),
    "rus_Cyrl": dict(fw=("HuggingFaceFW/fineweb-2", "rus_Cyrl"), order="SVO", spaced=True),
    "hin_Deva": dict(fw=("HuggingFaceFW/fineweb-2", "hin_Deva"), order="SOV", spaced=True),
    "zho_Hans": dict(fw=("HuggingFaceFW/fineweb-2", "cmn_Hani"), order="SVO", spaced=False),
    "tur_Latn": dict(fw=("HuggingFaceFW/fineweb-2", "tur_Latn"), order="SOV", spaced=True),
    "jpn_Jpan": dict(fw=("HuggingFaceFW/fineweb-2", "jpn_Jpan"), order="SOV", spaced=False),
    "arb_Arab": dict(fw=("HuggingFaceFW/fineweb-2", "arb_Arab"), order="VSO", spaced=True),
}
NGRAM = 13

# ---------------- FLORES ----------------
def _download_flores(langs, split):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"không import được `datasets` ({e!r}); cần nó để tải FLORES trên máy chạy thật") from e
    try:
        ds = load_dataset("facebook/flores", "all", split=split, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
        return {l: list(ds[f"sentence_{l}"]) for l in langs}
    except Exception as e:
        print("  [FLORES] facebook/flores lỗi:", repr(e)[:200], "-> thử openlanguagedata/flores_plus")
    out = {}
    for l in langs:                       # dataset gated: cần HF token
        ds = load_dataset("openlanguagedata/flores_plus", l, split=split, cache_dir=HF_CACHE_DIR)
        out[l] = [t for _, t in sorted(zip(ds["id"], ds["text"]))]
    return out

def get_flores(split="devtest", langs=None):
    """{lang: [câu]} căn hàng theo nội dung. devtest để đo, dev để tính byte premium / chi phí token.
    Dùng lại cache của Study 1 ({DATA_DIR}/processed/parallel/flores.jsonl) nếu đủ ngôn ngữ."""
    langs = list(langs or ALL_LANGS)
    if DATA_SOURCE == "toy":
        path = f"{CORPUS_DIR}/toy_flores_{split}.jsonl"
        if not os.path.exists(path):
            build_toy()
        texts = read_jsonl(path)
    else:
        cands = ([f"{DATA_DIR}/processed/parallel/flores.jsonl"] if split == "devtest" else []) + \
                [f"{DATA_DIR}/processed/parallel/flores_{split}_2b.jsonl"]
        texts = {}
        for p in cands:
            if os.path.exists(p):
                texts = read_jsonl(p)
                if all(l in texts for l in langs):
                    break
        if not all(l in texts for l in langs):
            print(f"[data] tải FLORES {split} -> {cands[-1]}")
            os.makedirs(os.path.dirname(cands[-1]), exist_ok=True)
            clean, keep = clean_parallel(_download_flores(ALL_LANGS, split))
            write_jsonl(cands[-1], clean, keep)
            texts = read_jsonl(cands[-1])
    texts = {l: texts[l] for l in langs}
    if split == "devtest" and N_EVAL:
        texts = {l: v[:N_EVAL] for l, v in texts.items()}
    assert len({len(v) for v in texts.values()}) == 1, "FLORES không căn hàng"
    return texts

def byte_premium(dev=None):
    """Byte premium của mỗi ngôn ngữ = số byte UTF-8 của cùng nội dung / số byte tiếng Anh (FLORES dev)."""
    dev = dev or get_flores("dev")
    eng = sum(len(s.encode("utf-8")) for s in dev["eng_Latn"])
    return {l: sum(len(s.encode("utf-8")) for s in v) / eng for l, v in dev.items()}

# ---------------- lọc trùng với tập đo (13-gram) ----------------
def _units(text, lang):
    return text.split() if LANG_INFO[lang]["spaced"] else [c for c in text if not c.isspace()]

def _ngrams(units):
    return {hash(tuple(units[i:i + NGRAM])) for i in range(len(units) - NGRAM + 1)}

def eval_ngrams(lang):
    s = set()
    for split in ("dev", "devtest"):
        for t in get_flores(split, [lang])[lang]:
            s |= _ngrams(_units(t, lang))
    return s

# ---------------- corpus ----------------
def corpus_path(lang):
    return f"{CORPUS_DIR}/{lang}.txt"

def _fineweb_docs(lang):
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(f"không import được `datasets` ({e!r}); cần nó để tải FineWeb-2") from e
    path, name = LANG_INFO[lang]["fw"]
    for r in load_dataset(path, name, split="train", streaming=True, cache_dir=HF_CACHE_DIR):
        yield r["text"]

def build_corpus(lang, budget_bytes):
    """Ghi corpus một ngôn ngữ: mỗi dòng một văn bản (NFKC, xuống dòng -> dấu cách), bỏ văn bản chứa 13-gram của
    FLORES dev / devtest, dừng khi đủ budget_bytes. Trả về thống kê."""
    path, bad = corpus_path(lang), eval_ngrams(lang)
    st = dict(lang=lang, bytes=0, docs=0, dropped_overlap=0, budget=int(budget_bytes))
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        bar = tqdm(total=int(budget_bytes), unit="B", unit_scale=True, desc=f"corpus {lang}", leave=False)
        for doc in _fineweb_docs(lang):
            doc = norm_text(" ".join(doc.split()))
            if not doc:
                continue
            if _ngrams(_units(doc, lang)) & bad:
                st["dropped_overlap"] += 1
                continue
            line = doc + "\n"
            f.write(line)
            n = len(line.encode("utf-8")); st["bytes"] += n; st["docs"] += 1; bar.update(n)
            if st["bytes"] >= budget_bytes:
                break
        bar.close()
    os.replace(path + ".tmp", path)
    return st

# ---------------- toy (SMOKE): ngôn ngữ giả lập có chữ viết và trật tự từ riêng ----------------
_TOY_SCRIPT = {"eng_Latn": (0x61, 26), "fra_Latn": (0x61, 26), "tur_Latn": (0x61, 26), "rus_Cyrl": (0x430, 32),
               "hin_Deva": (0x915, 30), "zho_Hans": (0x4E00, 2000), "jpn_Jpan": (0x3041, 80), "arb_Arab": (0x627, 26)}

def build_toy():
    """Dữ liệu toy: cùng một ngữ pháp nội dung (chủ ngữ [tính từ] - động từ - tân ngữ, khái niệm Zipf), mỗi ngôn ngữ có
    bảng chữ, từ vựng và trật tự từ (LANG_INFO) riêng. Chỉ để chạy thử đường ống; không dùng cho kết quả."""
    rng = random.Random(f"{SEED}-toy")
    n_noun, n_verb, n_adj = 500, 150, 100
    zipf = lambda n: [1.0 / r for r in range(1, n + 1)]
    pick = lambda n: rng.choices(range(n), weights=zipf(n))[0]
    lex = {}
    for l in ALL_LANGS:
        base, size = _TOY_SCRIPT[l]
        r = random.Random(f"{SEED}-toy-{l}")
        word = lambda: "".join(chr(base + r.randrange(size)) for _ in range(r.randint(1, 3) if size > 100 else r.randint(2, 6)))
        lex[l] = {k: [word() for _ in range(n)] for k, n in (("n", n_noun), ("v", n_verb), ("a", n_adj))}
    def content():
        return (pick(n_noun), pick(n_adj) if rng.random() < 0.4 else None, pick(n_verb), pick(n_noun))
    def render(l, c):
        s, a, v, o = c; L = lex[l]
        subj = [L["n"][s]] + ([L["a"][a]] if a is not None else [])
        parts = {"S": subj, "V": [L["v"][v]], "O": [L["n"][o]]}
        words = [w for k in LANG_INFO[l]["order"] for w in parts[k]]
        sep = " " if LANG_INFO[l]["spaced"] else ""
        return sep.join(words) + ("." if LANG_INFO[l]["spaced"] else "。")
    os.makedirs(CORPUS_DIR, exist_ok=True)
    for split, n in (("dev", TOY_N_EVAL), ("devtest", TOY_N_EVAL)):
        cs = [content() for _ in range(n)]
        write_jsonl(f"{CORPUS_DIR}/toy_flores_{split}.jsonl", {l: [render(l, c) for c in cs] for l in ALL_LANGS}, range(n))
    for l in ALL_LANGS:                   # mỗi văn bản là nội dung mới, viết bằng đúng một ngôn ngữ (không song song)
        with open(corpus_path(l), "w", encoding="utf-8") as f:
            for _ in range(TOY_N_DOCS):
                sep = " " if LANG_INFO[l]["spaced"] else ""
                f.write(sep.join(render(l, content()) for _ in range(rng.randint(5, 12))) + "\n")

def prepare_data():
    """FLORES dev + devtest, byte premium, corpus mỗi ngôn ngữ (= CORPUS_BYTES_ENG x premium); ghi manifest."""
    man_p = f"{CORPUS_DIR}/manifest.json"
    man = read_json(man_p, {}) or {}
    if DATA_SOURCE == "toy":
        build_toy()
    prem = byte_premium()
    get_flores("devtest")
    for l in ALL_LANGS:
        if DATA_SOURCE == "toy":
            st = dict(lang=l, bytes=os.path.getsize(corpus_path(l)), docs=sum(1 for _ in open(corpus_path(l), encoding="utf-8")),
                      dropped_overlap=0, budget=None)
        elif l in man.get("langs", {}) and os.path.exists(corpus_path(l)) and man["langs"][l]["budget"] == int(CORPUS_BYTES_ENG * prem[l]):
            st = man["langs"][l]
        else:
            st = build_corpus(l, CORPUS_BYTES_ENG * prem[l])
        st["premium"] = prem[l]
        man.setdefault("langs", {})[l] = st
        write_json(man_p, dict(man, source=DATA_SOURCE, corpus_bytes_eng=CORPUS_BYTES_ENG))
        print(f"[data] {l}: {st['bytes'] / 1e6:.1f} MB, {st['docs']} văn bản, premium {prem[l]:.2f}, bỏ trùng FLORES {st['dropped_overlap']}")
    return man
