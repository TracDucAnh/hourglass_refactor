# Parser nghịch kiểm giả định (G) (Cell S3)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 8 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell S3 — Parser nghịch (kiểm tra giả định G) + verify
# ============================================================
# Parser nhận một chuỗi từ và trả về TẬP các cây nội dung C có thể sinh ra nó trong ngôn ngữ spec.
#   G2 (đơn ánh):          Parser(spec).parse(render(spec, C)) == {C}   (đúng một cây, chính là C)
#   G3 (lộ ngôn ngữ):      với mọi ngôn ngữ khác, parse(...) rỗng
# Câu "mơ hồ" = có >1 cây trong chính ngôn ngữ đó, hoặc phân tích được bởi ngôn ngữ khác.
# Cài đặt: phân tích hình thái từng từ (tách theo lớp âm tiết) rồi parse theo span với bộ nhớ đệm, dùng đúng các
# mẫu trật tự/hướng đầu của renderer (T05-T10).
_TENSE_CODE = {"PAST": 0, "FUT": 2}

class Parser:
    def __init__(self, spec):
        self.s = spec

    def analyze(self, w):
        """Từ -> ('MARK', tên) | (loại, cid, (đặc trưng...)) | None."""
        s = self.s
        if not w or len(w) % 2:
            return None
        sy = [w[i:i + 2] for i in range(0, len(w), 2)]
        kinds = []
        for x in sy:
            if x in s.stem_set: kinds.append("S")
            elif x in s.aff_inv: kinds.append("A")
            else: return None
        if "S" not in kinds:
            if len(sy) == 1 and s.aff_inv[sy[0]][0] == "mark":
                return ("MARK", s.aff_inv[sy[0]][1])
            return None
        first = kinds.index("S")
        last = len(kinds) - 1 - kinds[::-1].index("S")
        if any(k != "S" for k in kinds[first:last + 1]):
            return None
        cid = s.stem2cid.get("".join(sy[first:last + 1]))
        if cid is None:
            return None
        cat = category(cid)
        if cat in ("ADJ", "ADV"):
            return (cat, cid, ()) if (first == 0 and last == len(sy) - 1) else None
        allowed = NOUN_FEATS if cat == "N" else VERB_FEATS
        feats = {}
        for group, side in ((sy[:first], "pre"), (sy[last + 1:], "suf")):
            prev = -1
            for x in group:
                f, v = s.aff_inv[x]
                if f == "mark" or f not in allowed or s.pos[f] != side:
                    return None
                if f == "case" and not s.has_case: return None
                if f == "agr" and not s.has_agr: return None
                ix = allowed.index(f)
                if ix <= prev or f in feats:
                    return None
                prev = ix; feats[f] = v
        return (cat, cid, tuple(sorted(feats.items())))

    def parse(self, words):
        s = self.s
        n = len(words)
        an = [self.analyze(w) for w in words]
        if n == 0 or any(a is None for a in an):
            return set()
        memo_core, memo_np, memo_cl = {}, {}, {}

        def core(i, j, acc):
            key = (i, j, acc)
            if key in memo_core: return memo_core[key]
            res = []
            if j > i:
                hi = i if s.head["adj"] == "I" else j - 1
                a = an[hi]
                adj_idx = range(i + 1, j) if s.head["adj"] == "I" else range(i, j - 1)
                if a[0] == "N" and all(an[t][0] == "ADJ" for t in adj_idx):
                    f = dict(a[2])
                    if ("case" in f) == (acc and s.has_case):
                        res.append(("N", a[1], 1 if f.get("num") == "PL" else 0, 1 if "def" in f else 0,
                                    tuple(an[t][1] for t in adj_idx), None))
            memo_core[key] = res
            return res

        def np_(i, j, acc):
            key = (i, j, acc)
            if key in memo_np: return memo_np[key]
            res = set(core(i, j, acc))
            for m in range(i, j):
                a = an[m]
                if a[0] != "MARK":
                    continue
                if a[1] == "POSS":
                    cs, ss = ((i, m), (m + 1, j)) if s.head["poss"] == "I" else ((m + 1, j), (i, m))
                    for c in core(cs[0], cs[1], acc):
                        for sub in np_(ss[0], ss[1], False):
                            res.add(c[:5] + (("POSS", sub),))
                else:
                    cs, ss = ((i, m), (m + 1, j)) if s.head["rel"] == "I" else ((m + 1, j), (i, m))
                    for c in core(cs[0], cs[1], acc):
                        for cl in clause_(ss[0], ss[1], 1):
                            res.add(c[:5] + (("REL", cl),))
            memo_np[key] = res
            return res

        def vinfo(a, adv):
            f = dict(a[2])
            return (a[1], _TENSE_CODE.get(f.get("tense"), 1), 1 if "neg" in f else 0, adv, "agr" in f)

        def vunit(a, b):
            if b - a == 1 and an[a][0] == "V":
                return [vinfo(an[a], -1)]
            if b - a == 2:
                vi, ai = (a, a + 1) if s.head["adv"] == "I" else (a + 1, a)
                if an[vi][0] == "V" and an[ai][0] == "ADV":
                    return [vinfo(an[vi], an[ai][1])]
            return []

        def clause_(i, j, val):
            key = (i, j, val)
            if key in memo_cl: return memo_cl[key]
            res = set()
            def mk(vu, subj, obj=None):
                pred, tense, neg, adv, agr = vu
                if verb_valency(pred) != val or agr != (s.has_agr and subj[2] == 1):
                    return
                res.add(("S", pred, tense, neg, adv, (subj,) if obj is None else (subj, obj)))
            if val == 1:
                for m in range(i + 1, j):
                    if s.order == "VSO":
                        for vu in vunit(i, m):
                            for sb in np_(m, j, False): mk(vu, sb)
                    else:
                        for sb in np_(i, m, False):
                            for vu in vunit(m, j): mk(vu, sb)
            else:
                for a in range(i + 1, j):
                    for b in range(a + 1, j):
                        if s.order == "SVO":   vus, sbs, obs = vunit(a, b), np_(i, a, False), np_(b, j, True)
                        elif s.order == "SOV": vus, sbs, obs = vunit(b, j), np_(i, a, False), np_(a, b, True)
                        else:                  vus, sbs, obs = vunit(i, a), np_(a, b, False), np_(b, j, True)
                        for vu in vus:
                            for sb in sbs:
                                for ob in obs: mk(vu, sb, ob)
            memo_cl[key] = res
            return res

        return clause_(0, n, 1) | clause_(0, n, 2)

def get_parser(spec):
    if spec._parser is None:
        spec._parser = Parser(spec)
    return spec._parser

def verify_languages(specs, sampler, n=500, heldout=False):
    """Kiểm (G) trên n nội dung: round-trip đơn ánh (G2), mơ hồ trong ngôn ngữ, và phân tích được bởi ngôn ngữ khác (G3)."""
    L = len(specs)
    st = dict(n=0, roundtrip_fail=0, ambiguous_within=0, parsed_by_other=0, per_lang_fail=[0] * L)
    for _ in tqdm(range(n), desc="kiểm (G) bằng parser", leave=False, disable=(n < 100), dynamic_ncols=True):
        C = sampler.sample(heldout)
        for l, sp in enumerate(specs):
            words = [w for w, _, _ in render_words(sp, C)]
            res = get_parser(sp).parse(words)
            st["n"] += 1
            if res != {C}:
                st["roundtrip_fail"] += 1; st["per_lang_fail"][l] += 1
                if len(res) > 1: st["ambiguous_within"] += 1
            if any(get_parser(o).parse(words) for k, o in enumerate(specs) if k != l):
                st["parsed_by_other"] += 1
    st["rate_roundtrip_fail"] = st["roundtrip_fail"] / max(st["n"], 1)
    st["rate_parsed_by_other"] = st["parsed_by_other"] / max(st["n"], 1)
    return st
print("[S3] parser nghịch sẵn sàng")
