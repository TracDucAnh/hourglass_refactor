# Ngữ nghĩa C: kho khái niệm + PCFG R00–R28 (Cell S1)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 6 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell S1 — Ngữ nghĩa C: kho khái niệm + văn phạm PCFG
# ============================================================
# ====================== QUY TẮC NỘI DUNG ======================
# Kho khái niệm (5000):
#   R00  id 0..2499 DANH TỪ (N) | 2500..3999 ĐỘNG TỪ (V; 60% bắc cầu val=2, 40% nội động val=1)
#        | 4000..4699 TÍNH TỪ (ADJ) | 4700..4999 TRẠNG TỪ (ADV).
#        Mọi lựa chọn khái niệm lấy mẫu theo Zipf alpha=1 TRONG từng nhóm: P(hạng r) ∝ 1/r.
# Mệnh đề S (cây có kiểu):  ('S', pred, tense, neg, adv, args)   args = (NP,) hoặc (NP, NP)
#   R01  S -> PRED1 NP NP          p = 0.50   (động từ bắc cầu: arg0 = chủ thể, arg1 = đối tượng)
#   R02  S -> PRED0 NP             p = 0.50   (động từ nội động: arg0)
#   R03-R05  thì:      PAST 0.30 | PRES 0.50 | FUT 0.20      (mã 0, 1, 2)
#   R06-R07  phủ định: POS 0.85  | NEG 0.15
#   R08-R09  trạng từ: không 0.70 | một ADV 0.30            (ADV chỉ bổ nghĩa cho động từ)
# Cụm danh từ NP:  ('N', noun, num, def, adjs, ext)
#   R10-R11  số:        SG 0.70 | PL 0.30
#   R12-R13  xác định:  INDEF 0.50 | DEF 0.50
#   R14-R16  số tính từ: 0 (0.55) | 1 (0.30) | 2 (0.15)       (ADJ chỉ bổ nghĩa cho danh từ)
#   R17-R19  mở rộng:   không 0.80 | POSS NP (sở hữu, đệ quy) 0.10 | REL S0 (mệnh đề quan hệ nội động) 0.10
# Từ vựng (Zipf):  R20 N->noun_i | R21 PRED0->verb val1 | R22 PRED1->verb val2 | R23 ADJ->adj_i | R24 ADV->adv_i
# Ràng buộc:
#   R25  độ sâu lồng <= MAX_DEPTH (=4): chạm giới hạn thì R17 bắt buộc "không mở rộng"
#   R26  mệnh đề nằm trong REL luôn là R02 (nội động)
#   R27  câu có < MIN_WORDS từ bị loại và lấy mẫu lại
#   R28  tổ hợp (vị ngữ, danh từ arg0, danh từ arg1) có băm % 100 < HELDOUT_PCT bị GIỮ RA:
#        không bao giờ vào huấn luyện, chỉ dùng để đo (chia theo tổ hợp vị ngữ–đối số)
# =============================================================
N_NOUN, N_VERB, N_ADJ, N_ADV = 2500, 1500, 700, 300
NOUN0, VERB0, ADJ0, ADV0 = 0, 2500, 4000, 4700
N_CONCEPT = 5000
ZIPF_ALPHA = 1.0
P_INTR = 0.50
P_NEG, P_ADV, P_PL, P_DEF = 0.15, 0.30, 0.30, 0.50
P_TENSE = (0.30, 0.50, 0.20)
P_NADJ = (0.55, 0.30, 0.15)
P_POSS, P_REL = 0.10, 0.10

def category(cid):
    if cid < VERB0: return "N"
    if cid < ADJ0: return "V"
    if cid < ADV0: return "ADJ"
    return "ADV"

def verb_valency(cid):
    return 2 if ((cid - VERB0) % 5) < 3 else 1

def _zipf_probs(n):
    w = [1.0 / (r ** ZIPF_ALPHA) for r in range(1, n + 1)]
    tot = sum(w)
    return [x / tot for x in w]

def _cdf(p):
    acc, out = 0.0, []
    for x in p:
        acc += x; out.append(acc)
    out[-1] = 1.0
    return out

POOLS = {"N": list(range(NOUN0, NOUN0 + N_NOUN)),
         "V1": [c for c in range(VERB0, ADJ0) if verb_valency(c) == 1],
         "V2": [c for c in range(VERB0, ADJ0) if verb_valency(c) == 2],
         "ADJ": list(range(ADJ0, ADV0)), "ADV": list(range(ADV0, N_CONCEPT))}
POOL_PROBS = {k: _zipf_probs(len(v)) for k, v in POOLS.items()}
POOL_CDF = {k: _cdf(p) for k, p in POOL_PROBS.items()}

# hạng tần suất trong nhóm lấy mẫu của từng khái niệm (1 = thường gặp nhất); dùng cho độ dài từ và cho S3
CONCEPT_RANK = {}
for _k, _v in POOLS.items():
    for _r, _c in enumerate(_v, 1):
        CONCEPT_RANK[_c] = _r

# hạng tổng thể của động từ theo xác suất xuất hiện làm vị ngữ chính -> nhãn SEM (50 lớp)
_vp = [(P_INTR * POOL_PROBS["V1"][r] if verb_valency(c) == 1 else (1 - P_INTR) * POOL_PROBS["V2"][r], c)
       for k in ("V1", "V2") for r, c in enumerate(POOLS[k])]
VERB_OVERALL_RANK = {c: i + 1 for i, (_, c) in enumerate(sorted(_vp, key=lambda x: (-x[0], x[1])))}

def _pick(rng, pool):
    return POOLS[pool][bisect.bisect_left(POOL_CDF[pool], rng.random())]

def sample_np(rng, depth):
    noun = _pick(rng, "N")
    num = 1 if rng.random() < P_PL else 0                       # R10-R11
    df = 1 if rng.random() < P_DEF else 0                       # R12-R13
    u = rng.random()
    na = 0 if u < P_NADJ[0] else (1 if u < P_NADJ[0] + P_NADJ[1] else 2)   # R14-R16
    adjs = tuple(_pick(rng, "ADJ") for _ in range(na))
    ext = None
    if depth < MAX_DEPTH:                                       # R25
        u = rng.random()
        if u < P_POSS:                                          # R18
            ext = ("POSS", sample_np(rng, depth + 1))
        elif u < P_POSS + P_REL:                                # R19 (+R26)
            ext = ("REL", sample_clause(rng, depth + 1, intransitive=True))
    return ("N", noun, num, df, adjs, ext)

def sample_clause(rng, depth, intransitive=False):
    val = 1 if (intransitive or rng.random() < P_INTR) else 2   # R01 / R02
    pred = _pick(rng, "V1" if val == 1 else "V2")               # R21 / R22
    u = rng.random()
    tense = 0 if u < P_TENSE[0] else (1 if u < P_TENSE[0] + P_TENSE[1] else 2)   # R03-R05
    neg = 1 if rng.random() < P_NEG else 0                      # R06-R07
    adv = _pick(rng, "ADV") if rng.random() < P_ADV else -1     # R08-R09
    args = tuple(sample_np(rng, depth) for _ in range(val))
    return ("S", pred, tense, neg, adv, args)

def np_words(np_):
    n = 1 + len(np_[4])
    if np_[5] is not None:
        n += 1 + (np_words(np_[5][1]) if np_[5][0] == "POSS" else clause_words(np_[5][1]))
    return n

def clause_words(S):
    return 1 + (1 if S[4] >= 0 else 0) + sum(np_words(a) for a in S[5])

def clause_depth(S):
    def nd(np_):
        if np_[5] is None: return 1
        return 1 + (nd(np_[5][1]) if np_[5][0] == "POSS" else clause_depth(np_[5][1]))
    return max(nd(a) for a in S[5])

def is_heldout(C):                                              # R28
    n1 = C[5][1][1] if len(C[5]) == 2 else -1
    h = ((C[1] * 1000003 + C[5][0][1] * 10007 + (n1 + 1) * 101) * 2654435761) % (2 ** 32)
    return (h >> 8) % 100 < HELDOUT_PCT

class ContentSampler:
    """Dòng nội dung tất định theo seed. heldout=False: nội dung cho huấn luyện; heldout=True: nội dung giữ ra để đo."""
    def __init__(self, seed):
        self.rng = random.Random(f"content-{seed}")
    def sample(self, heldout=False):
        while True:
            C = sample_clause(self.rng, 1)
            if clause_words(C) >= MIN_WORDS and is_heldout(C) == heldout:     # R27, R28
                return C
    def sample_probe(self):
        """Tập dò S3: khung 'V0(N ADJ)' với danh từ chủ ngữ lấy LOG-ĐỀU theo hạng (mỗi bát phân log2 có số mẫu gần bằng nhau,
        kể cả đầu bảng tần suất), tổ hợp (vị ngữ, danh từ) thuộc tập giữ ra."""
        while True:
            noun = NOUN0 + min(int(math.exp(self.rng.random() * math.log(N_NOUN))), N_NOUN) - 1
            pred = _pick(self.rng, "V1")
            C = ("S", pred, 1, 0, -1, (("N", noun, 0, 0, (_pick(self.rng, "ADJ"),), None),))
            if is_heldout(C):
                return C
print("[S1] khái niệm:", {k: len(v) for k, v in POOLS.items()}, "| đã định nghĩa PCFG R00-R28")
