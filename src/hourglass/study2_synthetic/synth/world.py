# Dựng 'thế giới': corpus + tập đo, có cache (Cell S5)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 10 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell S5 — Dựng "thế giới": ngôn ngữ + tokenizer + pool huấn luyện + tập đo (có cache)
# ============================================================
# Một thế giới = (L, level, overlap, len_scales, shares, seed). Mọi (family, task, d) cùng thế giới dùng chung file này.
#  - pool: N_POOL câu; MỖI nội dung được viết bằng đúng MỘT ngôn ngữ (chọn theo `shares`) -> không có dữ liệu song song.
#    Nhãn kèm theo: sem (hạng vị ngữ, cắt 50 lớp), surf (độ dài ký tự x ký tự đầu của từ THỨ HAI).
#  - eval: N_EVAL nội dung giữ ra (R28) x L ngôn ngữ, song song theo cấu trúc; wid = chỉ số từ của từng token (-1: đặc biệt/pad).
#  - probe: N_PROBE khung 'V0(N ADJ)' với danh từ chủ ngữ lấy log-đều theo hạng (S3); subj = chỉ số từ của danh từ chủ ngữ.
def world_cfg(L, level, overlap, len_scale, seed, len_scale_X=None, share_X=None):
    ls = [len_scale] * L
    if len_scale_X is not None:
        ls[0] = len_scale_X                              # ngôn ngữ X = chỉ số 0
    sh = [1.0 / L] * L
    if share_X is not None:
        sh = [share_X] + [(1 - share_X) / (L - 1)] * (L - 1)
    return dict(L=L, level=level, overlap=overlap, len_scales=tuple(ls), shares=tuple(sh), seed=seed)

def world_id(w):
    ls, sh = w["len_scales"], w["shares"]
    lst = f"{ls[0]:g}" if len(set(ls)) == 1 else "-".join(f"{x:g}" for x in ls)
    sht = "eq" if max(sh) - min(sh) < 1e-9 else "-".join(f"{x:.2f}" for x in sh)
    return f"L{w['L']}_{w['level']}_ov{w['overlap']}_ls{lst}_sh{sht}_s{w['seed']}" + ("_smoke" if SMOKE else "")

def _encode_sentence(bpe, words):
    ids, wid = [BOS], [-1]
    for j, w in enumerate(words):
        t = bpe.encode_word(w)
        ids += t; wid += [j] * len(t)
    return ids + [EOS], wid + [-1]

def _render_eval(specs, bpe, contents, probe=False):
    L = len(specs)
    ids_l = [[] for _ in range(L)]; wid_l = [[] for _ in range(L)]; subj = []; keep = []
    for ci, C in enumerate(contents):
        enc = []
        for sp in specs:
            wr = render_words(sp, C)
            ids, wid = _encode_sentence(bpe, [w for w, _, _ in wr])
            enc.append((ids, wid, next((j for j, (_, _, r) in enumerate(wr) if r == "N_subj"), -1)))
        if any(len(e[0]) > MAX_SENT_TOK + 2 for e in enc):
            continue
        for l, (ids, wid, sj) in enumerate(enc):
            ids_l[l].append(ids); wid_l[l].append(wid)
        subj.append(enc[0][2]); keep.append(ci)
    N = len(keep)
    ids_a = np.zeros((L, N, MAX_LEN), np.int16); wid_a = np.full((L, N, MAX_LEN), -1, np.int16)
    for l in range(L):
        for n in range(N):
            k = len(ids_l[l][n])
            ids_a[l, n, :k] = ids_l[l][n]; wid_a[l, n, :k] = wid_l[l][n]
    return ids_a, wid_a, np.array(subj, np.int16), keep

def build_world(wcfg, verbose=True):
    t0 = time.time()
    L, seed = wcfg["L"], wcfg["seed"]
    specs = make_languages(L, wcfg["level"], wcfg["overlap"], list(wcfg["len_scales"]), seed)
    alphabet = sorted({ch for s in specs for ch in s.C + s.V})
    letter_idx = {c: i for i, c in enumerate(alphabet)}
    n_cls_surf = 8 * len(alphabet)
    assert n_cls_surf <= N_LABEL, "tăng N_LABEL"
    owner = {}
    if wcfg["overlap"] == 0:
        for s in specs:
            for ch in s.C + s.V:
                owner[ch] = s.idx
    # ---- pool huấn luyện ----
    sampler = ContentSampler(f"{seed}-train")
    rng_l = random.Random(f"{seed}-langpick")
    cum = np.cumsum(wcfg["shares"])
    word2i, sent_w, langs, sems, surfs = {}, [], [], [], []
    for _ in tqdm(range(N_POOL), desc="sinh pool (render)", leave=False, mininterval=2.0, dynamic_ncols=True):
        C = sampler.sample(False)
        l = min(bisect.bisect_left(cum, rng_l.random() * cum[-1]), L - 1)
        ws = [w for w, _, _ in render_words(specs[l], C)]
        sent_w.append([word2i.setdefault(w, len(word2i)) for w in ws])
        langs.append(l)
        sems.append(min(VERB_OVERALL_RANK[C[1]] - 1, N_SEM_CLS - 1))
        w2 = ws[1]
        surfs.append((min(len(w2) // 2, 8) - 1) * len(alphabet) + letter_idx[w2[0]])
    words_by_i = [None] * len(word2i)
    for w, i in word2i.items():
        words_by_i[i] = w
    cnt = np.bincount(np.concatenate([np.array(s) for s in sent_w]), minlength=len(words_by_i))
    top = np.argsort(-cnt)[:BPE_MAX_TYPES]
    bpe = SharedBPE.train({words_by_i[i]: int(cnt[i]) for i in top}, alphabet, BPE_VOCAB)
    word_tok = [bpe.encode_word(w) for w in tqdm(words_by_i, desc="BPE encode từ", leave=False, mininterval=2.0, dynamic_ncols=True)]
    ids_a = np.zeros((N_POOL, MAX_LEN), np.int16); lens = np.zeros(N_POOL, np.int16)
    keep = np.zeros(N_POOL, bool)
    for n, sw in enumerate(tqdm(sent_w, desc="ghép token cho pool", leave=False, mininterval=2.0, dynamic_ncols=True)):
        toks = [BOS] + [t for wi in sw for t in word_tok[wi]] + [EOS]
        if len(toks) <= MAX_SENT_TOK + 2:
            ids_a[n, :len(toks)] = toks; lens[n] = len(toks); keep[n] = True
    pool = dict(ids=ids_a[keep], len=lens[keep], lang=np.array(langs, np.int8)[keep],
                sem=np.array(sems, np.int16)[keep], surf=np.array(surfs, np.int16)[keep])
    # ---- tập đo ----
    se = ContentSampler(f"{seed}-eval")
    ev_ids, ev_wid, _, _ = _render_eval(specs, bpe, [se.sample(True) for _ in range(N_EVAL)])
    pr_c = ContentSampler(f"{seed}-probe")
    pcont = [pr_c.sample_probe() for _ in range(N_PROBE)]
    pr_ids, pr_wid, pr_subj, pkeep = _render_eval(specs, bpe, pcont)
    pr_rank = np.array([pcont[i][5][0][1] + 1 for i in pkeep], np.int16)
    # fertility (token / từ) theo ngôn ngữ trên tập đo
    fert = []
    for l in range(L):
        nt = ((ev_wid[l] >= 0).sum(1)).astype(float); nw = ev_wid[l].max(1) + 1.0
        fert.append(float(nt.sum() / nw.sum()))
    # kiểm (G) nhanh khi dựng
    ver = verify_languages(specs, ContentSampler(f"{seed}-verify"), n=100 if SMOKE else 200)
    tok_lang = bpe.token_language(owner)
    meta = dict(id=world_id(wcfg), cfg=wcfg, vocab_total=bpe.vocab_total, n_symbols=len(bpe.symbols),
                alphabet=alphabet, n_cls_sem=N_SEM_CLS, n_cls_surf=n_cls_surf, fertility=fert,
                n_pool=int(pool["ids"].shape[0]), verify={k: v for k, v in ver.items() if k != "per_lang_fail"},
                typology=[s.describe() for s in specs], bpe=bpe.to_json(), seconds=round(time.time() - t0, 1))
    arrays = {f"pool_{k}": v for k, v in pool.items()}
    arrays.update(eval_ids=ev_ids, eval_wid=ev_wid, probe_ids=pr_ids, probe_wid=pr_wid, probe_subj=pr_subj,
                  probe_rank=pr_rank, tok_lang=tok_lang)
    base = f"{WORLD_DIR}/{world_id(wcfg)}"
    tmp = f"{base}.{os.getpid()}.tmp"     # tên tạm riêng từng tiến trình: các họ chạy song song có thể dựng cùng một thế giới
    np.savez_compressed(tmp + ".npz", **arrays); os.replace(tmp + ".npz", base + ".npz")
    with open(tmp + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp + ".json", base + ".json")
    if verbose:
        print(f"[world] {meta['id']}: pool={meta['n_pool']}  vocab={meta['vocab_total']}  fertility={[round(x, 2) for x in fert]}  "
              f"roundtrip_fail={ver['rate_roundtrip_fail']:.4f}  parsed_by_other={ver['rate_parsed_by_other']:.4f}  ({meta['seconds']}s)")
    return load_world(wcfg)

def load_world(wcfg):
    base = f"{WORLD_DIR}/{world_id(wcfg)}"
    with np.load(base + ".npz") as z:
        W = {k: z[k] for k in z.files}
    with open(base + ".json", encoding="utf-8") as f:
        W["meta"] = json.load(f)
    return W

def get_world(wcfg, verbose=True):
    base = f"{WORLD_DIR}/{world_id(wcfg)}"
    if os.path.exists(base + ".npz") and os.path.exists(base + ".json"):
        return load_world(wcfg)
    return build_world(wcfg, verbose)
