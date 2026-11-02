# Dữ liệu song song FLORES-200 devtest + NTREX-128 (Cell A3)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 3 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell A3 — Dữ liệu song song: FLORES-200 devtest + NTREX-128
# Cả hai được chuẩn hoá NFKC, căn hàng theo content, lưu jsonl {id, lang, text} vào /mnt/cl/data
# ============================================================
from hourglass.common.text import norm_text

from hourglass.common.text import clean_parallel

from hourglass.common.io import write_jsonl

from hourglass.common.io import read_jsonl

# ---------------- FLORES-200 devtest ----------------
def load_raw_flores(langs, split="devtest"):
    from datasets import load_dataset
    try:
        ds = load_dataset("facebook/flores", "all", split=split, trust_remote_code=True, cache_dir=HF_CACHE_DIR)
        cols = {l: f"sentence_{l}" for l in langs}
        missing = [l for l, c in cols.items() if c not in ds.column_names]
        if missing:
            raise ValueError(f"Thiếu cột: {missing}")
        return {l: list(ds[c]) for l, c in cols.items()}
    except Exception as e:
        print("  [FLORES] facebook/flores lỗi:", repr(e)[:200], "-> thử openlanguagedata/flores_plus")
    out = {}
    for l in langs:                       # [XM] cấu hình theo mã ngôn ngữ; cần HF token (dataset gated)
        ds = load_dataset("openlanguagedata/flores_plus", l, split=split, cache_dir=HF_CACHE_DIR)
        out[l] = [t for _, t in sorted(zip(ds["id"], ds["text"]))]
    return out

# ---------------- NTREX-128 ----------------
NTREX_RAW = "https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/main/NTREX-128"
NTREX_API = "https://api.github.com/repos/MicrosoftTranslator/NTREX/contents/NTREX-128"
NTREX_DIR = f"{DATA_DIR}/raw/ntrex128"
# mã ngôn ngữ kiểu FLORES -> mã trong tên file NTREX-128 [XM] (file: newstest2019-ref.<mã>.txt;
# tiếng Anh là bản nguồn newstest2019-src.eng.txt)
NTREX_CODES = {"eng_Latn": ["eng"], "fra_Latn": ["fra"], "rus_Cyrl": ["rus"], "hin_Deva": ["hin"],
               "zho_Hans": ["zho-CN", "zho"], "vie_Latn": ["vie"], "tur_Latn": ["tur"],
               "arb_Arab": ["arb"], "jpn_Jpan": ["jpn"], "fin_Latn": ["fin"],
               "swh_Latn": ["swa"], "tha_Thai": ["tha"]}

from hourglass.common.net import _http_get

def _ntrex_listing():
    p = os.path.join(NTREX_DIR, "_listing.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    try:
        names = [x["name"] for x in json.loads(_http_get(NTREX_API))]
        with open(p, "w") as f:
            json.dump(names, f)
        return names
    except Exception as e:
        print("  [NTREX] không liệt kê được repo qua GitHub API:", repr(e)[:120], "-> dùng tên file đoán")
        return []

def _ntrex_candidates(lang, listing):
    guess = ["newstest2019-src.eng.txt"] if lang == "eng_Latn" else \
            [f"newstest2019-ref.{c}.txt" for c in NTREX_CODES[lang]]
    hit = [g for g in guess if g in listing]
    if not hit and listing:
        iso3 = NTREX_CODES[lang][0].split("-")[0]
        hit = sorted(n for n in listing if re.match(rf"newstest2019-(ref|src)\.{iso3}(-[A-Za-z]+)?\.txt$", n))
    return hit or guess

def _ntrex_fetch_file(lang, listing):
    for fname in _ntrex_candidates(lang, listing):
        path = os.path.join(NTREX_DIR, fname)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
        try:
            data = _http_get(f"{NTREX_RAW}/{fname}")
            if len(data) == 0:
                continue
            with open(path, "wb") as f:
                f.write(data)
            return path
        except (urllib.error.HTTPError, urllib.error.URLError):
            continue
    raise FileNotFoundError(f"Không tải được NTREX-128 cho {lang}. Thử: {_ntrex_candidates(lang, listing)}. "
                            f"Có thể tự đặt file vào {NTREX_DIR}.")

def _read_lines(path):
    # chỉ tách theo '\n' (KHÔNG dùng splitlines: nó tách cả U+2028/\x0b làm lệch hàng song song)
    with open(path, encoding="utf-8-sig") as f:
        txt = f.read().replace("\r\n", "\n")
    lines = txt.split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines

def load_raw_ntrex(langs):
    listing = _ntrex_listing()
    raw = {}
    for l in langs:
        raw[l] = _read_lines(_ntrex_fetch_file(l, listing))
    return raw

# ---------------- Giao diện chung ----------------
def get_texts(ds, n_sent=None):
    """Trả về {lang: [câu]} đã căn hàng. ds in {'flores', 'ntrex'}; xử lý + cache một lần vào /mnt/cl/data."""
    path = f"{DATA_DIR}/processed/parallel/{ds}.jsonl"
    texts = read_jsonl(path) if os.path.exists(path) else {}
    if not all(l in texts for l in LANGS):
        print(f"[data] xử lý {ds} -> {path}")
        raw = load_raw_flores(LANGS) if ds == "flores" else load_raw_ntrex(LANGS)
        clean, keep = clean_parallel(raw)
        write_jsonl(path, clean, keep)
        texts = read_jsonl(path)
    texts = {l: texts[l] for l in LANGS}
    if n_sent:
        texts = {l: v[:n_sent] for l, v in texts.items()}
    n = {len(v) for v in texts.values()}
    assert len(n) == 1, f"độ dài không khớp: {n}"
    print(f"[data] {ds}: {n.pop()} câu x {len(LANGS)} ngôn ngữ | EN[0]: {texts['eng_Latn'][0][:90]}")
    print("       độ dài trung bình (ký tự):", {l.split('_')[0]: round(float(np.mean([len(s) for s in v])), 1)
                                                for l, v in texts.items()})
    return texts
