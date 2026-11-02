# Tokenizer BPE chung (Cell S4)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 9 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell S4 — Tokenizer BPE chung (viết tay, không phụ thuộc thư viện ngoài)
# ============================================================
# Bố cục id: [0..5] token đặc biệt | [6 .. 6+N_LABEL) token nhãn phụ (SEM/SURF) | [SYM0 ..) ký hiệu BPE (ký tự + merge).
# Mỗi từ được mã hóa ĐỘC LẬP với tiền tố '▁' (merge không bắt qua ranh giới từ); câu = <bos> từ... <eos>.
BOW = "▁"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<mask>", "<sem>", "<surf>"]
PAD, BOS, EOS, MASK, SEM_TOK, SURF_TOK = range(6)
N_LABEL = 1100
LABEL0 = len(SPECIALS)
SYM0 = LABEL0 + N_LABEL

class SharedBPE:
    def __init__(self, base_chars, merges):
        self.base_chars = sorted(base_chars)
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        syms = [BOW] + self.base_chars
        seen = set(syms)
        for a, b in self.merges:
            if a + b not in seen:
                seen.add(a + b); syms.append(a + b)
        self.symbols = syms
        self.sym2id = {s: SYM0 + i for i, s in enumerate(syms)}
        self.cache = {}

    @property
    def vocab_total(self):
        return SYM0 + len(self.symbols)

    @classmethod
    def train(cls, word_counts, base_chars, n_symbols):
        words = [[BOW] + list(w) for w in word_counts]
        freqs = list(word_counts.values())
        pair_cnt, where = defaultdict(int), defaultdict(set)
        for wi, seq in enumerate(words):
            for p in zip(seq, seq[1:]):
                pair_cnt[p] += freqs[wi]; where[p].add(wi)
        heap = [(-c, p) for p, c in pair_cnt.items()]
        heapq.heapify(heap)
        symbols = {BOW} | set(base_chars)
        merges = []
        pbar = tqdm(total=max(0, n_symbols - len(symbols)), desc="BPE merges", leave=False, mininterval=2.0, dynamic_ncols=True)
        while len(symbols) < n_symbols:
            best = None
            while heap:
                negc, p = heapq.heappop(heap)
                c = pair_cnt.get(p, 0)
                if c > 0 and c == -negc:
                    best = p; break
                if c > 0:
                    heapq.heappush(heap, (-c, p))
            if best is None or pair_cnt[best] < 2:
                break
            a, b = best
            merges.append(best); symbols.add(a + b); pbar.update(1)
            for wi in list(where[best]):
                seq = words[wi]
                for p in zip(seq, seq[1:]):
                    pair_cnt[p] -= freqs[wi]
                new, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                        new.append(a + b); i += 2
                    else:
                        new.append(seq[i]); i += 1
                words[wi] = new
                for p in zip(new, new[1:]):
                    pair_cnt[p] += freqs[wi]; where[p].add(wi)
                    heapq.heappush(heap, (-pair_cnt[p], p))
            for p in [p for p, c in pair_cnt.items() if c <= 0]:
                del pair_cnt[p]
            pair_cnt.pop(best, None); where.pop(best, None)
        pbar.close()
        return cls(base_chars, merges)

    def encode_word(self, w):
        r = self.cache.get(w)
        if r is not None:
            return r
        seq = [BOW] + list(w)
        while len(seq) > 1:
            best, bi = None, -1
            for i in range(len(seq) - 1):
                rk = self.rank.get((seq[i], seq[i + 1]))
                if rk is not None and (best is None or rk < best):
                    best, bi = rk, i
            if best is None:
                break
            a, b = self.merges[best]
            new, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                    new.append(a + b); i += 2
                else:
                    new.append(seq[i]); i += 1
            seq = new
        r = [self.sym2id[s] for s in seq]
        self.cache[w] = r
        return r

    def decode_ids(self, ids):
        inv = {v: k for k, v in self.sym2id.items()}
        return "".join(inv[i] for i in ids if i in inv).replace(BOW, " ").strip()

    def token_language(self, letter_owner):
        """tok_lang[id] = chỉ số ngôn ngữ sở hữu token (mọi ký tự cùng một ngôn ngữ), -1 nếu không xác định (đặc biệt, '▁', chữ dùng chung)."""
        out = np.full(self.vocab_total, -1, np.int16)
        for s, i in self.sym2id.items():
            owners = {letter_owner.get(ch, -1) for ch in s if ch != BOW}
            if len(owners) == 1 and -1 not in owners:
                out[i] = owners.pop()
        return out

    def to_json(self):
        return dict(base_chars=self.base_chars, merges=[list(m) for m in self.merges])

    @classmethod
    def from_json(cls, d):
        return cls(d["base_chars"], d["merges"])
