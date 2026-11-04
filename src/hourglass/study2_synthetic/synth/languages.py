# Ngôn ngữ + renderer g_l, luật T01–T10 (Cell S2)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 7 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell S2 — Ngôn ngữ: bảng âm, từ vựng, hình thái, trật tự từ; renderer g_l
# ============================================================
# ====================== QUY TẮC THỰC HIỆN (renderer) ======================
# Âm vị và từ vựng:
#   T01  Mỗi âm tiết là CV (2 ký tự). Bảng chữ: 8 phụ âm + 5 nguyên âm => 40 âm tiết/ngôn ngữ.
#   T02  40 âm tiết chia hai lớp rời nhau: 9 âm tiết PHỤ TỐ (mang hình vị chức năng) + 31 âm tiết GỐC (tạo gốc từ).
#        => một từ luôn tách được duy nhất thành [phụ tố]* gốc+ [phụ tố]* (parser dựa vào đây).
#   T03  Gốc từ của khái niệm c = chuỗi n âm tiết gốc, n = max(1, round(len_scale*(0.6 + 0.42*ln(hạng_c)))):
#        khái niệm thường gặp -> từ ngắn (Zipf); len_scale điều khiển "token trên từ" (S4, S5).
#        Gốc từ duy nhất trong từng ngôn ngữ; hết chỗ thì n tăng thêm 1 (từ dài hơn danh nghĩa).
#   T04  Overlap 0%: mỗi ngôn ngữ một bảng chữ RIÊNG, rời nhau (giao diện lộ ngôn ngữ ở mọi token).
#        Overlap 100%: mọi ngôn ngữ dùng CÙNG bảng chữ và CÙNG phân lớp âm tiết (gốc/phụ tố); chỉ cách gán
#        gốc từ và phụ tố cho khái niệm là riêng -> cánh cân bằng giao diện (không nhãn ngôn ngữ).
# Hình thái (mỗi đặc trưng đánh dấu bằng đúng 1 âm tiết phụ tố; giá trị không đánh dấu = mặc định):
#   T05  Danh từ: số (PL đánh dấu, SG không) | xác định (DEF đánh dấu) | cách (ACC đánh dấu, chỉ khi has_case).
#   T06  Động từ: thì (PAST, FUT đánh dấu, PRES không) | phủ định (NEG) | hòa hợp (PL đánh dấu khi chủ ngữ số nhiều, chỉ khi has_agr).
#   T07  Vị trí phụ tố: mỗi đặc trưng là tiền tố (pre) hoặc hậu tố (suf); nhiều phụ tố cùng phía xếp theo thứ tự chuẩn
#        num<def<case (danh từ), tense<neg<agr (động từ). Tính từ/trạng từ không biến hình.
#   T08  Hình vị chức năng POSS, REL là một từ riêng gồm đúng 1 âm tiết phụ tố, nằm giữa phụ thuộc và trung tâm.
# Cú pháp:
#   T09  Trật tự từ trong mệnh đề: SVO | SOV | VSO (nội động: SV, SV, VS). Trạng từ dính với động từ thành "đơn vị V".
#        Hướng đầu cho từng cấu trúc (adj, poss, rel, adv): 'I' = phụ thuộc đứng SAU trung tâm, 'F' = đứng TRƯỚC.
#        NP = [tính từ] danh từ [tính từ] rồi mới đến POSS/REL (mở rộng đệ quy bên phải/trái theo hướng đầu).
#   T09b Hòa hợp hướng đầu: poss và rel luôn CÙNG hướng. (Nếu POSS đứng sau trung tâm mà REL đứng trước, mệnh đề quan hệ
#        có thể bám vào hai danh từ khác nhau -> mơ hồ cấu trúc thật; parser phát hiện ~0,4% câu khi để hai hướng độc lập.)
#   T10  Cách: nếu has_case, arg1 của mệnh đề chính mang ACC; mọi NP lồng và arg0 không đánh dấu.
# Thang khác biệt giữa các ngôn ngữ (sample_typology):
#   M0: SVO, mọi phụ tố hậu tố, hướng đầu 'I', không cách, không hòa hợp (chỉ từ vựng khác)
#   M2: + trật tự từ lấy ngẫu nhiên theo ngôn ngữ + vị trí phụ tố (pre/suf) từng đặc trưng
#   M4: + hướng đầu từng cấu trúc + has_case (p=0.5) + has_agr (p=0.5)
# ==========================================================================
LEVELS = ("M0", "M2", "M4")
ALL_FEATS = ["num", "def", "case", "tense", "neg", "agr"]
NOUN_FEATS = ["num", "def", "case"]
VERB_FEATS = ["tense", "neg", "agr"]
AFFIX_ITEMS = [("num", "PL"), ("def", "DEF"), ("case", "ACC"), ("tense", "PAST"), ("tense", "FUT"),
               ("neg", "NEG"), ("agr", "PL"), ("mark", "POSS"), ("mark", "REL")]
N_AFF = len(AFFIX_ITEMS)
N_CONS, N_VOW = 8, 5
N_ANCHOR_CONS, N_ANCHOR_VOW = 3, 3      # bảng chữ CHUNG của điểm neo (9 âm tiết), không thuộc riêng ngôn ngữ nào

def _letter_pool():
    latin = "abcdefghijklmnopqrstuvwxyz"
    greek = "αβγδεζηθικλμνξοπρστυφχψω"
    cyr = "абвгдежзийклмнопрстуфхцчшщыэюя"
    hira = "あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわ"
    pool = list(latin + greek + cyr + hira)
    assert len(set(pool)) == len(pool)
    return pool

def sample_typology(rng, level):
    t = dict(order="SVO", head=dict(adj="I", poss="I", rel="I", adv="I"),
             pos={f: "suf" for f in ALL_FEATS}, has_case=False, has_agr=False)
    if level in ("M2", "M4"):
        t["order"] = rng.choice(["SVO", "SOV", "VSO"])
        t["pos"] = {f: rng.choice(["pre", "suf"]) for f in ALL_FEATS}
    if level == "M4":
        t["head"] = {c: rng.choice(["I", "F"]) for c in ("adj", "poss", "adv")}
        t["head"]["rel"] = t["head"]["poss"]      # hòa hợp hướng đầu của hai mở rộng NP (xem T09b), tránh mơ hồ bám
        t["has_case"] = rng.random() < 0.5
        t["has_agr"] = rng.random() < 0.5
    return t

def build_lexicon(rng, stem_sylls, len_scale):
    order = sorted(range(N_CONCEPT), key=lambda c: (CONCEPT_RANK[c], c))     # khái niệm thường gặp được gán trước
    lex, used = [None] * N_CONCEPT, set()
    for cid in order:
        n = max(1, int(len_scale * (0.6 + 0.42 * math.log(CONCEPT_RANK[cid])) + 0.5))   # T03
        while True:
            for _ in range(24):
                s = "".join(rng.choice(stem_sylls) for _ in range(n))
                if s not in used:
                    break
            else:
                n += 1
                continue
            break
        used.add(s); lex[cid] = s
    return lex, {s: c for c, s in enumerate(lex)}

def make_anchors(rng, sylls, frac):
    """Điểm neo: một phần `frac` danh từ (chọn ngẫu nhiên, trải đều theo hạng) có gốc từ GIỐNG HỆT ở mọi ngôn ngữ, viết bằng
    bảng chữ chung - như tên riêng / số trong văn bản thật. Phụ tố (số, xác định, cách) vẫn theo từng ngôn ngữ. Độ dài theo T03
    (len_scale = 1). Trả về {cid: gốc từ}."""
    picked = sorted(rng.sample(POOLS["N"], int(round(frac * len(POOLS["N"])))), key=lambda c: (CONCEPT_RANK[c], c))
    out, used = {}, set()
    for cid in picked:
        n = max(1, int(0.6 + 0.42 * math.log(CONCEPT_RANK[cid]) + 0.5))
        while True:
            for _ in range(24):
                s = "".join(rng.choice(sylls) for _ in range(n))
                if s not in used:
                    break
            else:
                n += 1
                continue
            break
        used.add(s); out[cid] = s
    return out

class LangSpec:
    def __init__(self, idx, C, V, stem_sylls, aff, typ, lex, len_scale, anchor_sylls=()):
        self.idx, self.C, self.V, self.len_scale = idx, list(C), list(V), len_scale
        self.stem_sylls = list(stem_sylls)
        self.stem_set = set(stem_sylls) | set(anchor_sylls)      # âm tiết điểm neo cũng là âm tiết gốc (parser nhận ra danh từ neo)
        self.anchor_letters = sorted({ch for s in anchor_sylls for ch in s})
        self.aff = aff                                           # (feat, val) -> âm tiết
        self.aff_inv = {s: k for k, s in aff.items()}
        self.order, self.head, self.pos = typ["order"], typ["head"], typ["pos"]
        self.has_case, self.has_agr = typ["has_case"], typ["has_agr"]
        self.lex = lex
        self.stem2cid = {s: c for c, s in enumerate(lex)}
        self.letters = sorted(set(self.C + self.V))
        self._parser = None
    def typology_key(self):
        return (self.order, tuple(sorted(self.head.items())), tuple(sorted(self.pos.items())), self.has_case, self.has_agr)
    def describe(self):
        return (f"L{self.idx}: {self.order}, head={''.join(self.head[c] for c in ('adj','poss','rel','adv'))}, "
                f"pos={''.join(self.pos[f][0] for f in ALL_FEATS)}, case={int(self.has_case)}, agr={int(self.has_agr)}, "
                f"len_scale={self.len_scale:g}")

def make_languages(L, level, overlap, len_scales, seed, anchor=0.0):
    """anchor > 0: thêm điểm neo (make_anchors) dùng chung cho mọi ngôn ngữ; anchor = 0 cho đúng ngôn ngữ như trước."""
    assert overlap in (0, 100) and level in LEVELS and len(len_scales) == L
    pool = _letter_pool()
    random.Random(f"{seed}-letters").shuffle(pool)
    n_let = N_CONS + N_VOW
    assert overlap == 100 or L * n_let <= len(pool), "không đủ chữ cái cho overlap 0% với L này"
    def syll_split(C, V, rng):
        syl = [c + v for c in C for v in V]
        rng.shuffle(syl)
        return syl[:N_AFF], syl[N_AFF:]                          # T02: 9 phụ tố + 31 gốc
    anchors, a_syl = {}, []
    if anchor > 0:                                               # chữ cái của điểm neo: ngay sau phần chữ các ngôn ngữ đã dùng
        k0 = n_let if overlap == 100 else L * n_let
        assert k0 + N_ANCHOR_CONS + N_ANCHOR_VOW <= len(pool), "không đủ chữ cái cho điểm neo"
        a_syl = [c + v for c in pool[k0: k0 + N_ANCHOR_CONS] for v in pool[k0 + N_ANCHOR_CONS: k0 + N_ANCHOR_CONS + N_ANCHOR_VOW]]
        anchors = make_anchors(random.Random(f"{seed}-anchor"), a_syl, anchor)
    shared = None
    if overlap == 100:                                           # T04: cùng bảng chữ, cùng phân lớp cho mọi ngôn ngữ
        C, V = pool[:N_CONS], pool[N_CONS:n_let]
        shared = (C, V) + syll_split(C, V, random.Random(f"{seed}-part"))
    specs = []
    for i in range(L):
        if overlap == 100:
            C, V, aff_pool, stem_sylls = shared
        else:
            C = pool[i * n_let: i * n_let + N_CONS]
            V = pool[i * n_let + N_CONS: (i + 1) * n_let]
            aff_pool, stem_sylls = syll_split(C, V, random.Random(f"{seed}-part{i}"))
        r_aff = random.Random(f"{seed}-aff{i}")
        sh = list(aff_pool); r_aff.shuffle(sh)
        aff = {item: sh[j] for j, item in enumerate(AFFIX_ITEMS)}
        typ = sample_typology(random.Random(f"{seed}-typ{i}"), level)
        lex, _ = build_lexicon(random.Random(f"{seed}-lex{i}"), stem_sylls, len_scales[i])
        for cid, st in anchors.items():                            # ghi đè SAU khi dựng: mọi từ không phải neo giữ nguyên
            lex[cid] = st
        specs.append(LangSpec(i, C, V, stem_sylls, aff, typ, lex, len_scales[i], a_syl))
    return specs

# ---------------- Renderer g_l ----------------
_TENSE_ITEM = {0: ("tense", "PAST"), 2: ("tense", "FUT")}

def _noun_word(spec, cid, num, df, case_marked):
    marks = {"num": ("num", "PL") if num == 1 else None,
             "def": ("def", "DEF") if df == 1 else None,
             "case": ("case", "ACC") if (case_marked and spec.has_case) else None}
    pre = [spec.aff[marks[f]] for f in NOUN_FEATS if marks[f] and spec.pos[f] == "pre"]
    suf = [spec.aff[marks[f]] for f in NOUN_FEATS if marks[f] and spec.pos[f] == "suf"]
    return "".join(pre) + spec.lex[cid] + "".join(suf)

def _verb_word(spec, cid, tense, neg, plural_subj):
    marks = {"tense": _TENSE_ITEM.get(tense),
             "neg": ("neg", "NEG") if neg == 1 else None,
             "agr": ("agr", "PL") if (spec.has_agr and plural_subj) else None}
    pre = [spec.aff[marks[f]] for f in VERB_FEATS if marks[f] and spec.pos[f] == "pre"]
    suf = [spec.aff[marks[f]] for f in VERB_FEATS if marks[f] and spec.pos[f] == "suf"]
    return "".join(pre) + spec.lex[cid] + "".join(suf)

def _render_np(spec, np_, case_marked, tag):
    _, noun, num, df, adjs, ext = np_
    head = [(_noun_word(spec, noun, num, df, case_marked), noun, tag)]
    mods = [(spec.lex[a], a, "ADJ") for a in adjs]
    core = head + mods if spec.head["adj"] == "I" else mods + head
    if ext is None:
        return core
    kind, sub = ext
    if kind == "POSS":
        subw, mk, side = _render_np(spec, sub, False, "N_other"), [(spec.aff[("mark", "POSS")], -1, "FUNC")], spec.head["poss"]
    else:
        subw, mk, side = _render_clause(spec, sub, False), [(spec.aff[("mark", "REL")], -1, "FUNC")], spec.head["rel"]
    return core + mk + subw if side == "I" else subw + mk + core

def _render_clause(spec, S, top):
    _, pred, tense, neg, adv, args = S
    vw = [(_verb_word(spec, pred, tense, neg, args[0][2] == 1), pred, "V")]
    if adv >= 0:
        aw = [(spec.lex[adv], adv, "ADV")]
        vw = vw + aw if spec.head["adv"] == "I" else aw + vw
    subj = _render_np(spec, args[0], False, "N_subj" if top else "N_other")
    if len(args) == 1:
        return vw + subj if spec.order == "VSO" else subj + vw
    obj = _render_np(spec, args[1], True, "N_obj" if top else "N_other")      # T10
    if spec.order == "SVO": return subj + vw + obj
    if spec.order == "SOV": return subj + obj + vw
    return vw + subj + obj

def render_words(spec, C):
    """Danh sách (từ, id khái niệm gốc hoặc -1, vai trò) theo thứ tự bề mặt."""
    return _render_clause(spec, C, True)

def render(spec, C):
    return " ".join(w for w, _, _ in render_words(spec, C))
