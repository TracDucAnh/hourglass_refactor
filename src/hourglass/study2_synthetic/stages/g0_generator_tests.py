# Cổng G0: testcase bộ sinh + số đo (Cell T1–T6)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 13, 14, 15, 16, 17, 18 (mã giữ nguyên văn).
# STAGE (hành động): chạy bằng load(stages=["g0_generator_tests"]) hoặc `python -m hourglass run <experiment> g0_generator_tests`.
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell T1 — Ví dụ trực quan: cùng một nội dung C, ba mức khác biệt, hai kiểu giao diện
# ============================================================
def show_tree(C):
    def np_s(n):
        s = f"N{n[1]}" + ("[PL]" if n[2] else "") + ("[DEF]" if n[3] else "") + "".join(f" +ADJ{a}" for a in n[4])
        if n[5]:
            s += f" +{n[5][0]}(" + (np_s(n[5][1]) if n[5][0] == "POSS" else cl_s(n[5][1])) + ")"
        return s
    def cl_s(S):
        t = ["PAST", "PRES", "FUT"][S[2]]
        return f"V{S[1]}[{t}{',NEG' if S[3] else ''}]" + (f" +ADV{S[4]}" if S[4] >= 0 else "") + " <- " + ", ".join(np_s(a) for a in S[5])
    return cl_s(C)

_demo = ContentSampler("demo")
_C_demo = next(c for c in (_demo.sample(False) for _ in range(500)) if clause_words(c) >= 5 and len(c[5]) == 2)
print("Nội dung C:", show_tree(_C_demo), "\n")
for _ov in (0, 100):
    for _lev in LEVELS:
        _sp = make_languages(3, _lev, _ov, [1.0] * 3, seed=7)
        print(f"--- overlap {_ov}%, mức {_lev} ---")
        for s in _sp:
            print(f"  {s.describe():78s} | {render(s, _C_demo)}")

# ============================================================
# Cell T2 — Thống kê luật PCFG (R01–R28)
# ============================================================
_rng = random.Random("t2-rules")
_N = 30000
_raw = [sample_clause(_rng, 1) for _ in range(_N)]         # mẫu thô, TRƯỚC khi áp R27/R28
_cl = [c for S in _raw for c, _d in walk_clauses(S)]
_top = _raw

check("R01/R02 xác suất bắc cầu = 0.50", _z(sum(len(S[5]) == 2 for S in _top) / _N, 0.5, _N) < 4)
for code, p in zip((0, 1, 2), (0.30, 0.50, 0.20)):
    check(f"R03-R05 thì {['PAST','PRES','FUT'][code]} = {p}", _z(sum(S[2] == code for S in _cl) / len(_cl), p, len(_cl)) < 4)
check("R06-R07 phủ định = 0.15", _z(sum(S[3] for S in _cl) / len(_cl), P_NEG, len(_cl)) < 4)
check("R08-R09 trạng từ = 0.30", _z(sum(S[4] >= 0 for S in _cl) / len(_cl), P_ADV, len(_cl)) < 4)

_nps = [(n, d) for S in _top for n, d in walk_nps(S)]
_n = len(_nps)
check("R10-R11 số nhiều = 0.30", _z(sum(n[2] for n, _ in _nps) / _n, P_PL, _n) < 4)
check("R12-R13 xác định = 0.50", _z(sum(n[3] for n, _ in _nps) / _n, P_DEF, _n) < 4)
for k, p in enumerate(P_NADJ):
    check(f"R14-R16 số tính từ = {k} ({p})", _z(sum(len(n[4]) == k for n, _ in _nps) / _n, p, _n) < 4)
_low = [n for n, d in _nps if d < MAX_DEPTH]
check("R18 POSS = 0.10 (khi độ sâu < giới hạn)", _z(sum(n[5] is not None and n[5][0] == "POSS" for n in _low) / len(_low), P_POSS, len(_low)) < 4)
check("R19 REL = 0.10 (khi độ sâu < giới hạn)", _z(sum(n[5] is not None and n[5][0] == "REL" for n in _low) / len(_low), P_REL, len(_low)) < 4)
check("R25 độ sâu <= MAX_DEPTH và không mở rộng ở độ sâu giới hạn",
      all(clause_depth(S) <= MAX_DEPTH for S in _top) and all(n[5] is None for n, d in _nps if d >= MAX_DEPTH),
      f"độ sâu lớn nhất = {max(clause_depth(S) for S in _top)}")
check("R26 mệnh đề trong REL luôn nội động",
      all(len(n[5][1][5]) == 1 for n, _ in _nps if n[5] is not None and n[5][0] == "REL"))

# Zipf trên danh từ trung tâm (R20): độ dốc log-log ≈ -alpha
_cnt = Counter(n[1] for n, _ in _nps)
_r = np.arange(1, 101)
_f = np.array([_cnt.get(NOUN0 + int(i) - 1, 0) for i in _r], float)
_slope = np.polyfit(np.log(_r), np.log(np.maximum(_f, 1)), 1)[0]
check("R20 Zipf: độ dốc log-log của danh từ ≈ -1 (hạng 1-100)", abs(_slope + ZIPF_ALPHA) < 0.15, f"độ dốc = {_slope:.3f}")

# R27 + R28
_s_tr, _s_ev = ContentSampler("t2-train"), ContentSampler("t2-eval")
_tr = [_s_tr.sample(False) for _ in range(20000)]
_ev = [_s_ev.sample(True) for _ in range(2000)]
check("R27 mọi câu >= MIN_WORDS từ", all(clause_words(c) >= MIN_WORDS for c in _tr + _ev), f"MIN_WORDS={MIN_WORDS}")
_combo = lambda c: (c[1], c[5][0][1], c[5][1][1] if len(c[5]) == 2 else -1)
check("R28 không tổ hợp giữ ra nào lọt vào huấn luyện", not ({_combo(c) for c in _tr} & {_combo(c) for c in _ev}))
check("R28 phần trăm tổ hợp bị giữ ra ≈ HELDOUT_PCT", abs(sum(is_heldout(c) for c in _raw) / _N - HELDOUT_PCT / 100) < 0.01,
      f"{sum(is_heldout(c) for c in _raw) / _N:.3%}")
_probe = ContentSampler("t2-probe")
_pc = [_probe.sample_probe() for _ in range(2000)]
_nr = np.array([c[5][0][1] for c in _pc])
check("tập dò S3: danh từ lấy log-đều theo hạng (hạng <= 50 chiếm ≈ 50%) và đều thuộc tập giữ ra",
      abs((_nr < 50).mean() - 0.5) < 0.05 and all(is_heldout(c) for c in _pc) and all(clause_words(c) >= MIN_WORDS for c in _pc))

# ============================================================
# Cell T3 — Luật thực hiện (T01–T10)
# ============================================================
from scipy.stats import spearmanr

# T01-T04: bảng chữ, âm tiết, từ vựng
sp0 = make_languages(8, "M4", 0, [1.0] * 8, seed=3)
sp1 = make_languages(8, "M4", 100, [1.0] * 8, seed=3)
check("T04 overlap 0%: bảng chữ rời nhau từng đôi", all(not (set(a.letters) & set(b.letters)) for a, b in itertools.combinations(sp0, 2)))
check("T04 overlap 100%: mọi ngôn ngữ cùng bảng chữ và cùng phân lớp âm tiết",
      all(a.letters == sp1[0].letters and a.stem_set == sp1[0].stem_set for a in sp1))
check("T02 lớp gốc và lớp phụ tố rời nhau; 9 phụ tố + 31 gốc",
      all(not (s.stem_set & set(s.aff_inv)) and len(s.aff_inv) == N_AFF and len(s.stem_set) == N_CONS * N_VOW - N_AFF for s in sp0 + sp1))
check("T03 gốc từ duy nhất trong từng ngôn ngữ (đơn ánh từ vựng)", all(len(set(s.lex)) == N_CONCEPT for s in sp0 + sp1))
check("T04 overlap 100%: cùng khái niệm vẫn khác dạng từ giữa các ngôn ngữ (từ vựng riêng)",
      np.mean([sp1[0].lex[c] != sp1[1].lex[c] for c in range(N_CONCEPT)]) > 0.99)
_rk = np.array([CONCEPT_RANK[c] for c in range(NOUN0, NOUN0 + N_NOUN)])
_ln = np.array([len(sp0[0].lex[c]) // 2 for c in range(NOUN0, NOUN0 + N_NOUN)])
_rho = spearmanr(_rk, _ln).correlation
check("T03 từ dài dần theo hạng (Zipf): Spearman(hạng, độ dài) > 0.6", _rho > 0.6, f"rho = {_rho:.2f}")
_m = {ls: np.mean([len(s.lex[c]) // 2 for c in range(N_CONCEPT)])
      for ls, s in ((ls, make_languages(1, "M0", 0, [ls], 1)[0]) for ls in (0.6, 1.0, 1.6))}
check("T03 len_scale tăng thì độ dài trung bình tăng", _m[0.6] < _m[1.0] < _m[1.6], {k: round(float(v), 2) for k, v in _m.items()})

# thang khác biệt M0 < M2 < M4: khoảng cách typology trung bình giữa các cặp ngôn ngữ
def _typ_vec(s):
    return [s.order] + [s.pos[f] for f in ALL_FEATS] + [s.head[c] for c in ("adj", "poss", "adv")] + [s.has_case, s.has_agr]
def _mean_typ_dist(level, seeds=range(30), L=6):
    d = []
    for sd in seeds:
        sp = make_languages(L, level, 0, [1.0] * L, seed=sd)
        d += [sum(x != y for x, y in zip(_typ_vec(a), _typ_vec(b))) for a, b in itertools.combinations(sp, 2)]
    return float(np.mean(d))
_td = {lv: _mean_typ_dist(lv) for lv in LEVELS}
check("thang khác biệt: khoảng cách typology M0 = 0 < M2 < M4", _td["M0"] == 0 and _td["M0"] < _td["M2"] < _td["M4"], {k: round(v, 2) for k, v in _td.items()})
check("M0: mọi ngôn ngữ SVO, hậu tố, đầu 'I', không cách/hòa hợp",
      all(s.order == "SVO" and set(s.pos.values()) == {"suf"} and set(s.head.values()) == {"I"} and not s.has_case and not s.has_agr
          for s in make_languages(6, "M0", 0, [1.0] * 6, 5)))
check("T09b hòa hợp hướng đầu: poss và rel cùng hướng ở M4", all(s.head["poss"] == s.head["rel"] for s in make_languages(40, "M4", 100, [1.0] * 40, 5)))

# đơn ánh bằng va chạm chuỗi: số chuỗi khác nhau == số nội dung khác nhau (mỗi ngôn ngữ)
_sm = ContentSampler("t3-collide")
_cs = list({_sm.sample(False) for _ in range(30000)})
_ok = True
for s in make_languages(4, "M4", 100, [1.0] * 4, seed=11):
    _ok &= len({render(s, c) for c in _cs}) == len(_cs)
check("G2 (va chạm chuỗi): 30 000 nội dung khác nhau -> 30 000 chuỗi khác nhau ở mỗi ngôn ngữ", _ok)
# bảo toàn thông tin: số từ không phụ thuộc ngôn ngữ và bằng clause_words
check("số từ bề mặt = số nút nội dung (renderer không thêm/bớt từ)",
      all(len(render_words(s, c)) == clause_words(c) for s in sp1[:3] for c in _cs[:2000]))

# ============================================================
# Cell T4 — Giả định (G): round-trip bằng parser, mơ hồ, lộ ngôn ngữ, tất định theo seed
# ============================================================
rows = []
for ov in (0, 100):
    for lev in LEVELS:
        agg = dict(n=0, rt=0, amb=0, other=0)
        for sd in range(3):
            sp = make_languages(4, lev, ov, [1.0] * 4, seed=100 + sd)
            st = verify_languages(sp, ContentSampler(f"t4-{ov}-{lev}-{sd}"), n=600)
            agg["n"] += st["n"]; agg["rt"] += st["roundtrip_fail"]; agg["amb"] += st["ambiguous_within"]; agg["other"] += st["parsed_by_other"]
        rows.append(dict(overlap=ov, level=lev, n=agg["n"], roundtrip_fail=agg["rt"], ambiguous_within=agg["amb"],
                         parsed_by_other=agg["other"], rate_other=agg["other"] / agg["n"]))
import pandas as pd
_t4 = pd.DataFrame(rows)
print(_t4.to_string(index=False))
check("G2: round-trip parser == {C} cho 100% câu, mọi cấp x overlap (0 va chạm cấu trúc)", int(_t4.roundtrip_fail.sum()) == 0,
      f"{int(_t4.roundtrip_fail.sum())}/{int(_t4.n.sum())} lỗi")
check("G3 overlap 0%: không câu nào được ngôn ngữ khác phân tích", int(_t4[_t4.overlap == 0].parsed_by_other.sum()) == 0)
check("G3 overlap 100%: tỉ lệ câu mơ hồ chéo ngôn ngữ < 0.1% (ngưỡng của kế hoạch)", float(_t4[_t4.overlap == 100].rate_other.max()) < 0.001,
      f"lớn nhất = {float(_t4[_t4.overlap == 100].rate_other.max()):.4%}")

# tất định theo seed: cùng seed -> cùng ngôn ngữ và cùng câu; khác seed -> khác
def _fp(seed):
    sp = make_languages(3, "M4", 0, [1.0] * 3, seed=seed)
    cs = ContentSampler(f"fp-{seed}")
    return [render(s, cs.sample(False)) for s in sp for _ in range(200)] + [s.describe() + s.lex[5] for s in sp]
check("tất định: cùng seed cho đúng cùng ngôn ngữ và câu", _fp(1) == _fp(1))
check("khác seed cho thế giới khác", _fp(1) != _fp(2))

# ============================================================
# Cell T5 — Kiểm thử số đo (G0) và tokenizer
# ============================================================
# ---- bản numpy của M1 để kiểm tính đúng đắn (Phần B dùng cùng công thức bằng torch) ----
def d_std_np(ZA, ZB):
    Z = np.concatenate([ZA, ZB]); mu, sd = Z.mean(0), np.maximum(Z.std(0), 1e-8)
    A, B = (ZA - mu) / sd, (ZB - mu) / sd
    A, B = A - A.mean(0), B - B.mean(0)                      # trừ trung bình từng ngôn ngữ (σ² như paper)
    s2 = (np.square(A).sum(1).mean() + np.square(B).sum(1).mean()) / 2
    return float(np.square(A - B).sum(1).mean() / s2), A, B, s2

def g_exact_np(ZA, ZB):
    D, A, B, s2 = d_std_np(ZA, ZB)
    d_shuf = (np.square(A).sum(1).mean() + np.square(B).sum(1).mean() - 2 * A.mean(0) @ B.mean(0)) / s2   # kỳ vọng chính xác trên hoán vị đều
    return D, 1 - D / d_shuf

_g = np.random.default_rng(0)
_Z = _g.normal(size=(600, 32))
D, G = g_exact_np(_Z, _Z.copy())
check("G0: hai biểu diễn giống hệt -> D = 0, G = 1", abs(D) < 1e-9 and abs(G - 1) < 1e-9)
D, G = g_exact_np(_Z, _g.normal(size=(600, 32)))
check("G0: hai biểu diễn độc lập -> D ≈ 2, G ≈ 0", abs(D - 2) < 0.1 and abs(G) < 0.1, f"D={D:.3f}, G={G:.3f}")
_Zb = _Z @ np.linalg.qr(_g.normal(size=(32, 32)))[0] + 0.3 * _g.normal(size=(600, 32))
D, G = g_exact_np(_Z, _Zb)
_, A_, B_, s2_ = d_std_np(_Z, _Zb)
_mc = np.mean([np.square(A_ - B_[_g.permutation(600)]).sum(1).mean() / s2_ for _ in range(1000)])
check("G0: kỳ vọng chính xác của D_shuffle khớp trung bình 1000 lần xáo (sai số tương đối < 1%)",
      abs((D / (1 - G)) - _mc) / _mc < 0.01, f"đóng={D / (1 - G):.4f}, MC={_mc:.4f}")

# ---- tokenizer BPE ----
_spt = make_languages(4, "M4", 0, [1.0] * 4, seed=21)
_smt = ContentSampler("t5-bpe")
_ws, _lg = [], []
for _i in range(6000):
    _l = _i % 4
    _ws.append([w for w, _, _ in render_words(_spt[_l], _smt.sample(False))]); _lg.append(_l)
_wc = Counter(w for s in _ws for w in s)
_alpha = sorted({ch for s in _spt for ch in s.C + s.V})
_bpe = SharedBPE.train(_wc, _alpha, 700)
check("BPE: số ký hiệu đạt đích (hoặc hết cặp để gộp)", len(_bpe.symbols) <= 700 and len(_bpe.symbols) > len(_alpha) + 100, f"{len(_bpe.symbols)} ký hiệu")
check("BPE: mã hóa rồi giải mã trả lại đúng từ", all(_bpe.decode_ids(_bpe.encode_word(w)) == w for w in list(_wc)[:3000]))
_owner = {ch: s.idx for s in _spt for ch in s.C + s.V}
_tl = _bpe.token_language(_owner)
_enc_lang_ok = all(len({int(_tl[t]) for t in _bpe.encode_word(w) if _tl[t] >= 0}) <= 1 for w in list(_wc)[:2000])
check("BPE + overlap 0%: mọi token của một từ cùng thuộc một ngôn ngữ (token lộ ngôn ngữ)", _enc_lang_ok)
_ids = [t for t in range(SYM0, _bpe.vocab_total) if _tl[t] >= 0]
check("BPE: gần như mọi ký hiệu (trừ '▁') có chủ ngôn ngữ", len(_ids) >= len(_bpe.symbols) - 2, f"{len(_ids)}/{len(_bpe.symbols)}")
def _fert(ls):
    sp = make_languages(2, "M0", 0, [ls] * 2, seed=31)
    sm = ContentSampler(f"t5-f{ls}")
    ws = [[w for w, _, _ in render_words(sp[i % 2], sm.sample(False))] for i in range(3000)]
    b = SharedBPE.train(Counter(w for s in ws for w in s), sorted({c for s in sp for c in s.C + s.V}), 500)
    return float(np.mean([len(b.encode_word(w)) for s in ws for w in s]))
_fr = {ls: _fert(ls) for ls in (0.6, 1.0, 1.6)}
check("fertility (token/từ) tăng theo len_scale (S4, S5)", _fr[0.6] < _fr[1.0] < _fr[1.6], {k: round(float(v), 2) for k, v in _fr.items()})

# ============================================================
# Cell T6 — Tổng kết cổng G0 (bộ sinh)
# ============================================================
import pandas as pd
_df = pd.DataFrame(TEST_LOG, columns=["test", "pass", "detail"])
print(_df.assign(status=_df["pass"].map({True: "PASS", False: "FAIL"}))[["status", "test", "detail"]].to_string(index=False))
n_fail = int((~_df["pass"]).sum())
print(f"\nG0 bộ sinh: {len(_df) - n_fail}/{len(_df)} PASS, {n_fail} FAIL")
assert n_fail == 0, "Có test FAIL: sửa bộ sinh trước khi chạy GPU (đừng chạy lưới)."
