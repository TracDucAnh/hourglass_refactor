# BPE chung (và biến thể cho S4), mã hoá corpus, token -> ngôn ngữ, chi phí token (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
from tokenizers import Tokenizer, decoders, models as tk_models, normalizers, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIALS = ["<pad>", "<bos>", "<eos>", "<mask>"]
PAD_ID, BOS_ID, EOS_ID, MASK_ID = range(4)

# Khoá tokenizer: "base" = mọi ngôn ngữ cùng trọng số (theo byte premium) | "<lang>x<w>" = ngôn ngữ đó có trọng số w
# khi train BPE (w > 1: nhiều merge hơn -> ít token / từ hơn). Mọi run dùng "base" trừ lưới S4.
def tok_weights(key):
    if key == "base":
        return {}
    lang, w = key.rsplit("x", 1)
    return {lang: float(w)}

def tok_dir(key):
    return f"{TOK_DIR}/{key}"

def manifest():
    m = read_json(f"{CORPUS_DIR}/manifest.json")
    if not m:
        raise RuntimeError("chưa có corpus: chạy stage prepare_data trước")
    return m

def _corpus_lines(lang, max_bytes=None):
    n = 0
    with open(corpus_path(lang), encoding="utf-8") as f:
        for line in f:
            yield line.rstrip("\n")
            n += len(line.encode("utf-8"))
            if max_bytes is not None and n >= max_bytes:
                return

def train_tokenizer(key):
    """BPE byte-level chung cho mọi ngôn ngữ (một tokenizer cho mọi run -> twin và mô hình chung cùng từ vựng)."""
    out = tok_dir(key)
    if os.path.exists(f"{out}/tokenizer.json"):
        return
    prem, w = {l: v["premium"] for l, v in manifest()["langs"].items()}, tok_weights(key)
    def lines():
        for l in ALL_LANGS:
            yield from _corpus_lines(l, TOK_TRAIN_BYTES_ENG * prem[l] * w.get(l, 1.0))
    tok = Tokenizer(tk_models.BPE())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=SPECIALS, show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    print(f"[tok] train BPE '{key}' (vocab {VOCAB}, trọng số {w or 'đều'})")
    tok.train_from_iterator(lines(), trainer=trainer)
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", bos_token="<bos>", eos_token="<eos>",
                            mask_token="<mask>").save_pretrained(out)

_TOK = {}
def get_tokenizer(key):
    if key not in _TOK:
        _TOK[key] = PreTrainedTokenizerFast.from_pretrained(tok_dir(key))
    return _TOK[key]

def tokens_path(key, lang):
    return f"{tok_dir(key)}/tokens/{lang}.bin"

def encode_corpus(key, lang, batch=2000):
    """Corpus -> dòng token uint16 (mỗi văn bản kết thúc bằng <eos>) + số lần xuất hiện của mỗi token."""
    path = tokens_path(key, lang)
    if os.path.exists(path) and os.path.exists(path + ".counts.npy"):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tok, counts, n = get_tokenizer(key), np.zeros(len(get_tokenizer(key)), np.int64), 0
    assert len(tok) < 65536, "VOCAB phải < 65536 để lưu uint16"
    def flush(buf, f):
        ids = tok(buf, add_special_tokens=False)["input_ids"]
        arr = np.fromiter((t for d in ids for t in (*d, EOS_ID)), np.uint16)
        f.write(arr.tobytes()); np.add.at(counts, arr, 1)
        return len(arr)
    with open(path + ".tmp", "wb") as f:
        buf = []
        for line in tqdm(_corpus_lines(lang), desc=f"encode {key}/{lang}", leave=False, mininterval=5):
            buf.append(line)
            if len(buf) == batch:
                n += flush(buf, f); buf = []
        if buf:
            n += flush(buf, f)
    np.save(path + ".counts.npy", counts)
    os.replace(path + ".tmp", path)
    print(f"[tok] {key}/{lang}: {n / 1e6:.1f}M token")

def load_tokens(key, lang):
    return np.memmap(tokens_path(key, lang), dtype=np.uint16, mode="r")

def build_tokens(needs):
    """needs: {khoá tokenizer: tập ngôn ngữ}. Train tokenizer còn thiếu rồi mã hoá corpus."""
    for key, langs in needs.items():
        train_tokenizer(key)
        for l in sorted(langs):
            encode_corpus(key, l)

def token_language(key, langs, min_share=0.8, min_count=5):
    """tok_lang[id] = chỉ số (trong langs) của ngôn ngữ dùng token đó nhiều nhất, nếu tần suất tương đối của ngôn
    ngữ đó chiếm >= min_share; -1 nếu dùng chung (vd. dấu câu, chữ số) hoặc quá hiếm. Dùng cho logit-lens (S4)."""
    C = np.stack([np.load(tokens_path(key, l) + ".counts.npy").astype(float) for l in langs])     # [L, V]
    rel = C / C.sum(1, keepdims=True).clip(1)
    share = rel / rel.sum(0, keepdims=True).clip(1e-12)
    out = np.where((share.max(0) >= min_share) & (C.sum(0) >= min_count), share.argmax(0), -1)
    out[:len(SPECIALS)] = -1
    return out.astype(np.int16)

def token_cost(key, langs=None):
    """Chi phí token của mỗi ngôn ngữ = số token trung bình mỗi câu FLORES dev. Vì các câu là bản dịch của nhau,
    đây là số token cho CÙNG một nội dung, so được giữa mọi ngôn ngữ (kể cả zho / jpn không có ranh giới từ)."""
    dev, tok = get_flores("dev", langs), get_tokenizer(key)
    return {l: float(np.mean([len(x) for x in tok(v, add_special_tokens=False)["input_ids"]])) for l, v in dev.items()}
