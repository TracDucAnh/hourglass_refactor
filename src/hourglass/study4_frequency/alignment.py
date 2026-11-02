# Gióng hàng Eflomal + xác nhận MUSE / Wiktionary (Cell A6)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 11 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell A6 — Gióng hàng từ (Eflomal) + xác nhận bằng từ điển MUSE -> các "cặp từ" dùng để đo truy hồi
# Quy trình cho mỗi cặp ngôn ngữ (a, b):
#   1. Eflomal (model 3) chạy hai chiều trên devtest + dev + NTREX (nhiều dữ liệu hơn -> gióng hàng tốt hơn; chỉ devtest được đo).
#      Giữ GIAO của hai chiều => mỗi từ có tối đa một liên kết => cặp một-một. Gióng hàng KHÔNG dùng biểu diễn của model đang đo.
#   2. Giữ cặp được MUSE xác nhận: hai từ có chung ít nhất một từ tiếng Anh tương ứng (xx<->en, pivot qua tiếng Anh cho cặp không có en).
#      v4: MUSE zh đổi giản thể; jpn tra thêm bằng dạng gốc (lemma); ngôn ngữ MUSE không có (swh) dùng từ điển thay thế -> nhãn 'alt' (tầng mở rộng).
#      Cặp ngôn ngữ ít hơn MIN_PAIR_ITEMS cặp được xác nhận -> xác nhận bằng Dice thống kê (cột confirm_src ghi rõ, để lọc khi phân tích).
#   3. Bỏ số, bỏ cặp cùng dạng chữ (tên riêng/vay mượn).
# Kết quả: ITEMS (mọi cặp, mọi câu devtest) + NEED[lang] (từ nào cần trích biểu diễn).
# v7: từ điển xác nhận = MUSE ∪ Wiktionary, khoá tra = dạng bề mặt (+ khoá phụ jpn) + LEMMA (Cell A6a). Mỗi mục có BẬC xác nhận ITEMS['tier']:
#     0 = từ điển gốc (= v5) | 1 = chỉ khi thêm lemma | 2 = chỉ khi thêm Wiktionary | 3 = Dice. Cặp 'tầng chính' vẫn định nghĩa theo MAIN_SRCS (cả hai ngôn ngữ có MUSE).
# ============================================================
from concurrent.futures import ThreadPoolExecutor

# ---------------- MUSE ----------------
MUSE_BASE = "https://dl.fbaipublicfiles.com/arrival/dictionaries"
MUSE_CODE = {"eng_Latn": "en", "fra_Latn": "fr", "rus_Cyrl": "ru", "hin_Deva": "hi", "zho_Hans": "zh", "vie_Latn": "vi",
             "tur_Latn": "tr", "arb_Arab": "ar", "jpn_Jpan": "ja", "fin_Latn": "fi", "swh_Latn": "sw", "tha_Thai": "th"}   # [XM]
_MUSE, _EN_INV, _DICTS = {}, {}, {}

_T2S = {}
def _t2s(s):
    """Phồn thể -> giản thể (OpenCC, rồi zhconv). Không có thư viện nào -> giữ nguyên và cảnh báo MỘT lần."""
    f = _T2S.get("f")
    if f is None:
        f = False
        try:
            from opencc import OpenCC
            try: cc = OpenCC("t2s")
            except Exception: cc = OpenCC("t2s.json")
            f = cc.convert
        except Exception:
            try:
                from zhconv import convert as _zc
                f = lambda x: _zc(x, "zh-hans")
            except Exception:
                print("  [warn] thiếu opencc-python-reimplemented / zhconv -> MUSE zh giữ nguyên (phồn thể). Cài: %uv pip install opencc-python-reimplemented zhconv")
        _T2S["f"] = f
    return f(s) if f else s

def load_muse(src, tgt):
    """Từ điển MUSE đầy đủ src-tgt -> dict {từ: {từ dịch}}; None nếu MUSE không có cặp này.
    v4: cột tiếng Trung được đổi sang GIẢN THỂ (MUSE_ZH_T2S) cho khớp corpus."""
    k = (src, tgt)
    if k in _MUSE:
        return _MUSE[k]
    path = f"{DATA_DIR}/raw/muse/{src}-{tgt}.txt"
    if not os.path.exists(path):
        try:
            data = _http_get(f"{MUSE_BASE}/{src}-{tgt}.txt", timeout=120)
            with open(path, "wb") as f:
                f.write(data)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            print(f"  [MUSE] không có {src}-{tgt} ({repr(e)[:60]})")
            _MUSE[k] = None
            return None
    conv = [(_t2s if (MUSE_ZH_T2S and c == "zh") else None) for c in (src, tgt)]
    d = defaultdict(set); n_all = n_bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split(); n_all += 1
            if len(p) == 2:
                x, y = p
                if conv[0]: x = conv[0](x)
                if conv[1]: y = conv[1](y)
                d[norm_word(x)].add(norm_word(y))
            else:
                n_bad += 1
    if n_all and n_bad / n_all > 0.01:
        print(f"  [MUSE] {src}-{tgt}: bỏ {n_bad}/{n_all} dòng không đúng dạng 'từ từ' (mục nhiều từ?) -> kiểm tra độ phủ bằng muse_coverage()")
    _MUSE[k] = dict(d)
    return _MUSE[k]

# ---- Từ điển THAY THẾ cho ngôn ngữ MUSE không có (swh) + (v7) Wiktionary BỔ SUNG cho mọi ngôn ngữ ----
ALT_DICT_DIR  = f"{DATA_DIR}/raw/dict_alt"
ALT_DICT_AUTO = {"sw": dict(name="Swahili")}                                               # [XM] mã MUSE -> tên ngôn ngữ trên kaikki.org (ngôn ngữ MUSE không có: Wiktionary là từ điển DUY NHẤT, nhãn 'alt')
WIKT_NAME     = {"fr": "French", "ru": "Russian", "hi": "Hindi", "zh": "Chinese", "vi": "Vietnamese", "tr": "Turkish",
                 "ar": "Arabic", "ja": "Japanese", "fi": "Finnish", "sw": "Swahili", "th": "Thai"}   # [XM] tên ngôn ngữ (đúng chính tả, hoa chữ đầu) trên kaikki.org
KAIKKI_URL    = "https://kaikki.org/dictionary/{n}/kaikki.org-dictionary-{n}.jsonl"        # [XM] Wiktionary tiếng Anh, trích xuất theo ngôn ngữ (JSONL)
_WIKT_BUILT = set()           # mã đã dựng trong phiên này (WIKT_FORCE_REBUILD chỉ dựng lại MỘT lần mỗi phiên, kể cả khi diag_zh_t2s xoá cache từ điển)
WIKT_EXTRACT_VERSION = 2          # tăng khi đổi cách trích: tệp dựng bởi bản cũ (có .meta.json cũ hơn) được dựng lại từ JSONL
_GLOSS_PAREN = regex.compile(r"\([^)]*\)|\[[^\]]*\]")
_GLOSS_LEAD  = regex.compile(r"^(?:to|a|an|the)\s+")
_GLOSS_WORD  = regex.compile(r"^[a-z][a-z'\-]*$")
_GLOSS_STOP  = {"plural", "singular", "genitive", "nominative", "accusative", "dative", "ablative", "locative", "instrumental", "vocative",
                "partitive", "inessive", "elative", "illative", "adessive", "allative", "essive", "translative", "abessive", "comitative",
                "alternative", "obsolete", "archaic", "inflection", "participle", "gerund", "infinitive", "imperative", "misspelling",
                "romanization", "transliteration", "abbreviation", "initialism", "acronym", "nonstandard", "spelling"}      # từ ngữ pháp/siêu dữ liệu lọt vào chú giải
_GLOSS_BAD_TAGS = {"misspelling", "obsolete", "archaic", "nonstandard"}
_FORM_OF_RE  = regex.compile(r"\bof\s+([^\s:,;()]+)\s*$")                        # dự phòng: sense gắn tag form-of/alt-of nhưng thiếu trường form_of -> lấy lemma ở cuối chú giải ("illative plural of talo")                     # sense mang nhãn này không phải bản dịch dùng trong văn bản hiện đại

def _download_stream(url, path, timeout=600):
    """Tải về đĩa theo khối (tệp kaikki.org có thể vài GB: KHÔNG đọc cả tệp vào RAM như _http_get)."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path + ".part", "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    os.replace(path + ".part", path)

def _gloss_parts(gl):
    """Một chú giải kaikki -> các từ Anh đơn lẻ (bỏ ngoặc, 'to/a/an/the', từ ngữ pháp)."""
    out = []
    for part in regex.split(r"[,;/]", _GLOSS_PAREN.sub(" ", gl.lower())):
        part = _GLOSS_LEAD.sub("", part.strip())
        if _GLOSS_WORD.match(part) and part not in _GLOSS_STOP:
            out.append(part)
    return out

def _kaikki_to_muse(raw_path, out_path, conv=None, keep_us=False, max_hops=3):
    """JSONL kaikki.org -> tệp định dạng MUSE ('từ từ_anh' mỗi dòng). Chỉ giữ chú giải MỘT từ tiếng Anh.
    v7: mục chỉ là DẠNG BIẾN CÁCH / chính tả khác (form_of / alt_of, vd 'inessive plural of talo') KHÔNG còn bị bỏ: ghi ánh xạ dạng -> lemma, rồi dạng đó THỪA HƯỞNG
    chú giải của lemma (tối đa max_hops bước). Đây là chỗ Wiktionary phủ đuôi tần suất: phần lớn từ hiếm trong corpus là dạng biến cách, trước đây chúng bị loại.
    conv: hàm đổi chuỗi áp lên từ đầu mục và lemma (zh: phồn -> giản); keep_us: giữ mục chứa '_' (tiếng Việt: từ ghép nối '_').
    Trả về (số mục, số cặp ghi ra, số dạng biến cách nhận được chú giải từ lemma)."""
    cv = conv or (lambda x: x)
    ok = lambda w: bool(w) and (keep_us or "_" not in w)
    gl, fm = defaultdict(set), defaultdict(set)
    n_ent = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            try: e = json.loads(line)
            except ValueError: continue
            w = norm_word(cv(e.get("word", "")))
            if not ok(w):
                continue
            n_ent += 1
            for sn in e.get("senses", []):
                if (set(sn.get("tags") or ()) | set(sn.get("raw_tags") or ())) & _GLOSS_BAD_TAGS:
                    continue
                base = sn.get("form_of") or sn.get("alt_of")
                if not base and ({"form-of", "alt-of"} & set(sn.get("tags") or ())):
                    m_ = _FORM_OF_RE.search((sn.get("glosses") or [""])[0])
                    base = [m_.group(1)] if m_ else None
                if base:
                    for x in base:
                        t = x.get("word") if isinstance(x, dict) else x
                        t = norm_word(cv(t)) if t else ""
                        if ok(t) and t != w:
                            fm[w].add(t)
                    continue
                for g_ in sn.get("glosses", []):
                    gl[w].update(_gloss_parts(g_))
    add = {}
    for w, lems in fm.items():                       # dạng biến cách -> lemma (có thể qua vài bước) -> chú giải của lemma
        seen, frontier, got = {w}, set(lems), set()
        for _ in range(max_hops):
            nxt = set()
            for lm in frontier:
                if lm in seen:
                    continue
                seen.add(lm); got |= gl.get(lm, set()); nxt |= fm.get(lm, set())
            frontier = nxt
            if not frontier:
                break
        if got:
            add[w] = got
    n_form = sum(1 for w in add if not gl.get(w))
    for w, gs in add.items():
        gl[w] |= gs
    n_pair = 0
    with open(out_path + ".tmp", "w", encoding="utf-8") as g:
        for w, gs in gl.items():
            for t in sorted(gs):
                g.write(f"{w} {t}\n"); n_pair += 1
    os.replace(out_path + ".tmp", out_path)
    return n_ent, n_pair, n_form

def _read_pairs(path):
    dd = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                dd[norm_word(p[0])].add(norm_word(p[1]))
    return dict(dd) or None

def _wikt_path(code):
    """swh: tệp từ điển alt cũ ({code}-en.txt, để tương thích); ngôn ngữ khác: {code}-en.wikt.txt (bổ sung cho MUSE)."""
    return f"{ALT_DICT_DIR}/{code}-en.txt" if code in ALT_DICT_AUTO else f"{ALT_DICT_DIR}/{code}-en.wikt.txt"

def _build_wikt_file(code):
    """Đảm bảo tệp Wiktionary của `code` tồn tại. Có sẵn mà không có .meta.json (người dùng tự đặt) hoặc đủ phiên bản -> dùng nguyên;
    thiếu / cũ phiên bản / WIKT_FORCE_REBUILD -> tải kaikki.org (theo luồng) rồi trích. Trả về đường dẫn hoặc None."""
    path = _wikt_path(code); meta_p = path + ".meta.json"
    os.makedirs(ALT_DICT_DIR, exist_ok=True)
    have, stale = os.path.exists(path), False
    if have and os.path.exists(meta_p):
        try:
            with open(meta_p) as f:
                stale = json.load(f).get("version", 0) < WIKT_EXTRACT_VERSION
        except Exception:
            stale = True
    if have and not stale and not (WIKT_FORCE_REBUILD and code not in _WIKT_BUILT):
        return path
    name = WIKT_NAME.get(code)
    if not name:
        return path if have else None
    raw = f"{ALT_DICT_DIR}/{code}.kaikki.jsonl"
    try:
        if not os.path.exists(raw):
            print(f"  [WIKT] tải {name} từ kaikki.org (vài trăm MB – vài GB, chỉ một lần)...")
            _download_stream(KAIKKI_URL.format(n=name), raw)
        conv = _t2s if (code == "zh" and MUSE_ZH_T2S) else None
        n_ent, n_pair, n_form = _kaikki_to_muse(raw, path, conv=conv, keep_us=(code == "vi"))
        with open(meta_p, "w") as f:
            json.dump(dict(version=WIKT_EXTRACT_VERSION, n_entries=n_ent, n_pairs=n_pair, n_form_entries=n_form,
                           built=datetime.now().isoformat(timespec="seconds")), f)
        print(f"  [WIKT] {code}-en dựng từ Wiktionary: {n_ent:,} mục -> {n_pair:,} cặp (từ, từ Anh); {n_form:,} dạng biến cách/chính tả khác nhận chú giải từ lemma")
        _WIKT_BUILT.add(code)
        if not WIKT_KEEP_RAW and os.path.exists(raw):
            os.remove(raw)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        print(f"  [WIKT] không dựng được từ điển '{code}' ({repr(e)[:80]}); đặt tệp MUSE-format vào {path} để dùng nguồn khác (hoặc cập nhật network)")
        return path if have else None
    return path

def load_alt_dict(code):
    """Từ điển xx->en thay thế (dict {từ: {từ Anh}}) cho ngôn ngữ MUSE không có; None nếu không có nguồn nào.
    Thứ tự: (1) tệp người dùng đặt sẵn {ALT_DICT_DIR}/{code}-en.txt; (2) ALT_DICT_AUTO: tải Wiktionary (kaikki.org) và trích (v7: có dạng biến cách)."""
    if not ALT_DICT:
        return None
    k = ("alt", code)
    if k in _MUSE:
        return _MUSE[k]
    d = None
    p = _build_wikt_file(code) if code in ALT_DICT_AUTO else f"{ALT_DICT_DIR}/{code}-en.txt"
    if p and os.path.exists(p):
        d = _read_pairs(p)
    _MUSE[k] = d
    return d

def wikt_for(code):
    """(v7) Từ điển xx->en từ Wiktionary BỔ SUNG cho ngôn ngữ đã có MUSE. None nếu tắt / không dựng được / ngôn ngữ dùng từ điển 'alt' (đã là Wiktionary)."""
    if not WIKT_DICT or code == "en" or (WIKT_CODES is not None and code not in WIKT_CODES):
        return None
    k = ("wikt", code)
    if k in _MUSE:
        return _MUSE[k]
    d = None
    if dicts_for(code)[2] == "muse":
        p = _build_wikt_file(code)
        if p and os.path.exists(p):
            d = _read_pairs(p)
    _MUSE[k] = d
    return d

def dicts_for(code):
    """(xx->en, en->xx, nguồn) của một mã ngôn ngữ; nguồn: 'muse' | 'alt' | None."""
    if code not in _DICTS:
        d1, d2 = load_muse(code, "en"), load_muse("en", code)
        origin = "muse" if (d1 is not None or d2 is not None) else None
        if origin is None:
            d1 = load_alt_dict(code); origin = "alt" if d1 else None
        _DICTS[code] = (d1, d2, origin)
    return _DICTS[code]

def dict_origin(lang):
    return "muse" if lang == "eng_Latn" else dicts_for(MUSE_CODE[lang])[2]

def prefetch_wikt():
    """Tải/dựng/nạp trước từ điển Wiktionary cho mọi ngôn ngữ (chạy trước build_items để lỗi tải hiện ra sớm, không phải giữa vòng lọc)."""
    for l in LANGS:
        if l == "eng_Latn":
            continue
        c = MUSE_CODE[l]; d1, d2, o = dicts_for(c)
        dw = wikt_for(c) if o == "muse" else None
        print(f"  [dict] {SHORT[l]}: nguồn chính={o or 'không có'} (xx->en {len(d1) if d1 else 0:,} | en->xx {len(d2) if d2 else 0:,}) | Wiktionary bổ sung: "
              + (f"{len(dw):,} từ" if dw else ("— (dùng làm từ điển chính)" if o == "alt" else "KHÔNG CÓ")))

def to_en(lang, w, level=0):
    """Tập từ tiếng Anh ứng với w. level 0 = từ điển gốc (MUSE; swh: từ điển alt) — đúng hành vi v4/v5; level >= 2 = hợp thêm Wiktionary (wikt_for).
    None nếu ngôn ngữ không có từ điển nào. Tiếng Anh: chính nó."""
    if lang == "eng_Latn":
        return {w}
    code = MUSE_CODE[lang]
    d1, d2, _ = dicts_for(code)
    dw = wikt_for(code) if level >= 2 else None
    if d1 is None and d2 is None and dw is None:
        return None
    s = set(d1.get(w, ())) if d1 else set()
    if d2:
        if code not in _EN_INV:
            inv = defaultdict(set)
            for en_w, xs in d2.items():
                for x in xs:
                    inv[x].add(en_w)
            _EN_INV[code] = inv
        s |= _EN_INV[code].get(w, set())
    if dw:
        s |= dw.get(w, set())
    return s

def to_en_keys(lang, keys, level=0):
    """Hợp các bản dịch của nhiều khoá tra (dạng bề mặt + khoá phụ + lemma). None nếu ngôn ngữ không có từ điển."""
    if isinstance(keys, str):
        keys = (keys,)
    out, none = set(), True
    for k in keys:
        t = to_en(lang, k, level)
        if t is not None:
            none = False; out |= t
    return None if none else out

def lookup_keys(wt, g):
    """(keys0, keys1) của token g. keys0 = (dạng chuẩn + khoá phụ ALT_KEY_LANGS như orthBase/lemma fugashi) — đúng khoá v5; keys1 = keys0 + LEMMA Stanza (v7, Cell A6a).
    Khoá phụ chỉ dùng để tra từ điển; đơn vị từ, tần suất, gióng hàng, biểu diễn vẫn theo dạng bề mặt."""
    n = wt["norm"][g]; k0 = [n]
    alt = wt.get("alt")
    if alt is not None and wt.get("lang") in ALT_KEY_LANGS and alt[g]:
        k0 += [x for x in alt[g] if x and x not in k0]
    k1 = list(k0)
    lem = wt.get("lemma")
    if lem is not None and wt.get("lang") in LEMMA_LANGS:
        x = lem[g]
        if x and x not in k1:
            k1.append(x)
    return tuple(k0), tuple(k1)

def muse_confirm(la, lb, ka, kb):
    """(v4/v5, giữ để tương thích) 1 = xác nhận, 0 = từ điển có nhưng không xác nhận, -1 = không kiểm được. Chỉ dùng từ điển gốc (level 0). Hàm mới: dict_tier."""
    ea, eb = to_en_keys(la, ka), to_en_keys(lb, kb)
    if ea is None or eb is None:
        return -1
    return int(len(ea & eb) > 0)

def dict_tier(la, lb, ka, kb):
    """(tier, known); ka/kb = (keys0, keys1) từ lookup_keys.
    tier: -2 = không kiểm được (thiếu từ điển) | -1 = kiểm được nhưng không có bản dịch chung | 0 = xác nhận bởi từ điển gốc (= v5) |
          1 = chỉ xác nhận khi thêm LEMMA | 2 = chỉ xác nhận khi thêm WIKTIONARY.
    known: cả hai từ đều có ít nhất một bản dịch trong từ điển đầy đủ (-> từ điển đủ khả năng PHÁN XÉT cặp này; known & tier == -1 = bị loại có cơ sở).\n    Lưu ý: tiếng Anh luôn 'có' (tập bản dịch của nó là chính nó) nên với cặp eng-X, known phản ánh phía X."""
    ea2, eb2 = to_en_keys(la, ka[1], 2), to_en_keys(lb, kb[1], 2)
    if ea2 is None or eb2 is None:
        return -2, False
    known = bool(ea2) and bool(eb2)
    e0a, e0b = to_en_keys(la, ka[0], 0), to_en_keys(lb, kb[0], 0)
    if e0a is not None and e0b is not None and (e0a & e0b):
        return 0, known
    if ka[1] != ka[0] or kb[1] != kb[0]:
        e1a, e1b = to_en_keys(la, ka[1], 1), to_en_keys(lb, kb[1], 1)
        if e1a is not None and e1b is not None and (e1a & e1b):
            return 1, known
    if ea2 & eb2:
        return 2, known
    return -1, known

# ---------------- Eflomal ----------------
def pair_sources(a, b):
    """[(tên nguồn, bảng từ a, bảng từ b)]: devtest đứng đầu (chỉ nguồn này được đo)."""
    return [("test", WT[MEAS][a], WT[MEAS][b])] + [(SPLIT_OF[ds], WT[SPLIT_OF[ds]][a], WT[SPLIT_OF[ds]][b]) for ds in ALIGN_EXTRA]

def align_pair(pi, force=False):
    """Chạy Eflomal cho cặp pi, lưu mọi liên kết giao (một-một) ở devtest + thống kê Dice. Trả về đường dẫn links.npz."""
    a, b = LANGS[PAIRS[pi][0]], LANGS[PAIRS[pi][1]]
    d = f"{WORK_DIR}/align/{PAIR_NAME[pi]}"; os.makedirs(d, exist_ok=True)
    out = f"{d}/links.npz"
    if os.path.exists(out) and not force:
        return out
    srcs = pair_sources(a, b)
    ls, lt, meta = [], [], []
    for si, (_, wa, wb) in enumerate(srcs):
        for i in range(len(wa["ptr"]) - 1):
            na, nb = wa["norm"][wa["ptr"][i]:wa["ptr"][i + 1]], wb["norm"][wb["ptr"][i]:wb["ptr"][i + 1]]
            if 0 < len(na) <= MAX_WORDS_SENT and 0 < len(nb) <= MAX_WORDS_SENT:
                ls.append(" ".join(na)); lt.append(" ".join(nb)); meta.append((si, i))
    fs, ft, ff, fr = (f"{d}/{n}" for n in ("src.txt", "trg.txt", "fwd.txt", "rev.txt"))
    for p, L in ((fs, ls), (ft, lt)):
        with open(p, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
    cmd = ["eflomal-align", "-s", fs, "-t", ft, "-f", ff, "-r", fr, "-m", str(EFLOMAL_MODEL), "--n-samplers", "2", "--overwrite"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"eflomal-align lỗi ({PAIR_NAME[pi]}): {r.stderr[-400:]}")
    with open(ff) as f:
        fwd = f.read().split("\n")
    with open(fr) as f:
        rev = f.read().split("\n")
    cnt_pair, cnt_a, cnt_b = Counter(), Counter(), Counter()
    t_sent, t_wa, t_wb = [], [], []
    for k, (si, i) in enumerate(meta):
        inter = set(fwd[k].split()) & set(rev[k].split())          # giao hai chiều => một-một
        if not inter:
            continue
        _, wa, wb = srcs[si]
        ga, gb = wa["ptr"][i], wb["ptr"][i]
        for lk in inter:
            x, y = lk.split("-")
            x, y = int(x), int(y)
            na, nb = wa["norm"][ga + x], wb["norm"][gb + y]
            cnt_pair[(na, nb)] += 1; cnt_a[na] += 1; cnt_b[nb] += 1
            if si == 0:
                t_sent.append(i); t_wa.append(ga + x); t_wb.append(gb + y)
    wa0, wb0 = srcs[0][1], srcs[0][2]
    n_pair = np.zeros(len(t_sent), np.int32); dice = np.zeros(len(t_sent), np.float32)
    for q, (i, ga, gb) in enumerate(zip(t_sent, t_wa, t_wb)):
        na, nb = wa0["norm"][ga], wb0["norm"][gb]
        c = cnt_pair[(na, nb)]
        n_pair[q] = c; dice[q] = 2.0 * c / (cnt_a[na] + cnt_b[nb])
    np.savez_compressed(out + ".tmp.npz", sent=np.asarray(t_sent, np.int32), wa=np.asarray(t_wa, np.int64),
                        wb=np.asarray(t_wb, np.int64), n_pair=n_pair, dice=dice, n_lines=np.int64(len(meta)))
    os.replace(out + ".tmp.npz", out)
    for p in (fs, ft, ff, fr):                                       # bỏ tệp trung gian (lớn)
        os.remove(p)
    return out

_LINKS = {}      # cache trong bộ nhớ: pi -> liên kết Eflomal + bộ lọc + bậc xác nhận (build_items xoá cache; cell A7d đọc lại)

def link_tiers(pi, wt_a, wt_b, force=False):
    """(v7) Mọi liên kết Eflomal giao hai chiều của cặp pi + bộ lọc DROP_* + BẬC xác nhận từ điển của từng liên kết.
    keep: qua DROP_NUMERIC / DROP_SAME_FORM | tier: -2 không kiểm được (hoặc không thuộc keep), -1 bị từ điển loại, 0/1/2 xem dict_tier | known: từ điển đủ khả năng phán xét."""
    if pi in _LINKS and not force:
        return _LINKS[pi]
    a, b = LANGS[PAIRS[pi][0]], LANGS[PAIRS[pi][1]]
    z = np.load(f"{WORK_DIR}/align/{PAIR_NAME[pi]}/links.npz")
    sent, wa, wb, n_pair, dice = z["sent"], z["wa"], z["wb"], z["n_pair"], z["dice"]
    na = np.asarray(wt_a["norm"], dtype=object)[wa] if len(wa) else np.asarray([], dtype=object)
    nb = np.asarray(wt_b["norm"], dtype=object)[wb] if len(wb) else np.asarray([], dtype=object)
    keep = np.ones(len(wa), bool)
    if DROP_NUMERIC and len(wa):
        keep &= ~(wt_a["is_num"][wa] | wt_b["is_num"][wb])
    if DROP_SAME_FORM and len(wa):
        keep &= (na != nb)
    tier = np.full(len(wa), -2, np.int8); known = np.zeros(len(wa), bool)
    if CONFIRM_MODE == "muse" and len(wa):
        memo = {}
        for q in np.where(keep)[0]:
            ka, kb = lookup_keys(wt_a, wa[q]), lookup_keys(wt_b, wb[q])      # (khoá v5, khoá v5 + lemma)
            if (ka, kb) not in memo:
                memo[(ka, kb)] = dict_tier(a, b, ka, kb)
            tier[q], known[q] = memo[(ka, kb)]
    _LINKS[pi] = dict(sent=sent, wa=wa, wb=wb, n_pair=n_pair, dice=dice, keep=keep, tier=tier, known=known)
    return _LINKS[pi]

def select_items(pi, wt_a, wt_b):
    """Áp bộ lọc lên các liên kết Eflomal của cặp pi. Trả về (dict mảng, dòng thống kê).
    confirm_src: 'muse' (cả hai ngôn ngữ có MUSE) | 'alt' (ít nhất một ngôn ngữ dùng từ điển thay thế) | 'dice' (thống kê) | 'none'.
    Chỉ 'muse' (MAIN_SRCS) vào phân tích chính; phần còn lại là TẦNG MỞ RỘNG.
    v7: mục được giữ nếu bậc xác nhận nằm trong [0, ITEM_MAX_TIER]; items['tier'] ghi bậc (3 = Dice / không từ điển)."""
    a, b = LANGS[PAIRS[pi][0]], LANGS[PAIRS[pi][1]]
    L = link_tiers(pi, wt_a, wt_b)
    sent, wa, wb, n_pair, dice, keep, tier = (L[k] for k in ("sent", "wa", "wb", "n_pair", "dice", "keep", "tier"))
    n_links, n_filtered = len(wa), int(keep.sum())
    oa, ob = dict_origin(a), dict_origin(b)
    src, sel = "none", keep
    if CONFIRM_MODE == "muse":
        src = "muse" if (oa == "muse" and ob == "muse") else ("alt" if (oa and ob) else "none")
        sel = keep & (tier >= 0) & (tier <= ITEM_MAX_TIER)
        if sel.sum() < MIN_PAIR_ITEMS and FALLBACK_DICE:
            sel, src = keep & (dice >= DICE_MIN) & (n_pair >= DICE_MIN_COUNT), "dice"
    t_sel = tier[sel].astype(np.int8) if (CONFIRM_MODE == "muse" and src != "dice") else np.full(int(sel.sum()), 3, np.int8)
    items = dict(sent=sent[sel], wa=wa[sel], wb=wb[sel], tier=t_sel)
    st = dict(pair=PAIR_NAME[pi], n_links=n_links, n_after_filters=n_filtered, n_muse_checked=int((tier >= -1).sum()),
              n_muse_ok=int((tier >= 0).sum()), n_tier0=int((tier == 0).sum()), n_tier1=int((tier == 1).sum()), n_tier2=int((tier == 2).sum()),
              n_items=int(sel.sum()), confirm_src=src, low_items=bool(sel.sum() < MIN_PAIR_ITEMS), dict_a=str(oa), dict_b=str(ob))
    return items, st

def build_items(force=False):
    """Gióng hàng 66 cặp (song song) -> ITEMS, NEED, bảng thống kê. Lưu WORK_DIR/align/items.npz."""
    path = f"{WORK_DIR}/align/items.npz"
    _LINKS.clear()                                                   # v7: bậc xác nhận phụ thuộc từ điển/lemma hiện tại
    todo = [pi for pi in range(len(PAIRS)) if force or not os.path.exists(f"{WORK_DIR}/align/{PAIR_NAME[pi]}/links.npz")]
    if todo:
        print(f"[align] chạy Eflomal cho {len(todo)}/{len(PAIRS)} cặp ({EFLOMAL_WORKERS} luồng)...")
        t0 = time.time()
        with ThreadPoolExecutor(EFLOMAL_WORKERS) as ex:
            list(ex.map(lambda pi: align_pair(pi, force), todo))
        print(f"[align] xong sau {time.time() - t0:.0f}s")
    rows, parts = [], []
    for pi in range(len(PAIRS)):
        a, b = LANGS[PAIRS[pi][0]], LANGS[PAIRS[pi][1]]
        it, st = select_items(pi, WT[MEAS][a], WT[MEAS][b])
        parts.append(it); rows.append(st)
    ptr = np.cumsum([0] + [len(p["sent"]) for p in parts])
    cat = lambda k: np.concatenate([p[k] for p in parts]) if ptr[-1] else np.zeros(0, np.int64)
    items = dict(ptr=ptr, sent=cat("sent"), wa=cat("wa"), wb=cat("wb"), tier=cat("tier").astype(np.int8))
    import pandas as pd
    stats = pd.DataFrame(rows)
    np.savez_compressed(path, **items)
    stats.to_csv(f"{WORK_DIR}/align/pair_stats.csv", index=False)
    return items, stats

def need_masks(items):
    need = {l: np.zeros(len(WT[MEAS][l]["norm"]), bool) for l in LANGS}
    for pi, (ia, ib) in enumerate(PAIRS):
        s, e = items["ptr"][pi], items["ptr"][pi + 1]
        need[LANGS[ia]][items["wa"][s:e]] = True
        need[LANGS[ib]][items["wb"][s:e]] = True
    return need

def items_signature(items):
    """Chữ ký của ITEMS: kết quả truy hồi chỉ hợp lệ với đúng bộ cặp từ đã dùng để tính (đổi từ điển/khoá tra -> ITEMS đổi -> kết quả cũ stale)."""
    import hashlib
    h = hashlib.md5()
    for k in ("ptr", "sent", "wa", "wb"):
        h.update(np.ascontiguousarray(items[k]).tobytes())
    return h.hexdigest()[:12]
