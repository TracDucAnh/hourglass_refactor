# Lưới thí nghiệm + run_id (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(). Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.
FAMILIES = ["decoder", "moe", "diffusion"]
CAUSAL = {"decoder", "moe"}

def make_cfg(grid, family="decoder", kind="joint", d=384, L=2, nest="A", lang=None, frac=None, tok="base", share="eq", seed=0):
    """kind = "joint" (học |L| ngôn ngữ của NESTINGS[nest]) | "twin" (chỉ học `lang`, cùng tokenizer và cùng lượng dữ liệu).
    frac: phần corpus mỗi ngôn ngữ dùng; share: "eq" hoặc "x<m>" (S4_LANG dùng gấp m lần phần corpus)."""
    return dict(grid=grid, family=family, kind=kind, d=d, L=L if kind == "joint" else 1, nest=nest if kind == "joint" else None,
                lang=lang, frac=DATA_FRAC if frac is None else frac, tok=tok, share=share, seed=seed)

def langs_of(c):
    return [c["lang"]] if c["kind"] == "twin" else NESTINGS[c["nest"]][:c["L"]]

def run_id(c):
    """Không chứa tên lưới: cùng cấu hình ở hai lưới (vd. pilot và p3) là cùng một run; twin dùng chung cho mọi |L|."""
    if c["kind"] == "twin":
        return f"{c['family']}_twin_{c['lang']}_d{c['d']}_f{c['frac']:g}_tk{c['tok']}_s{c['seed']}"
    return f"{c['family']}_joint_d{c['d']}_L{c['L']}{c['nest']}_f{c['frac']:g}_tk{c['tok']}_sh{c['share']}_s{c['seed']}"

def ckpt_path(c):
    """Decoder -> CKPT_DIR (Study 4 tự đăng ký; twin_* bị bỏ qua); MoE / diffusion -> OTHER_CKPT_DIR."""
    base = CKPT_DIR if c["family"] == "decoder" else OTHER_CKPT_DIR
    name = run_id(c).replace("decoder_joint_", "").replace("decoder_twin_", "twin_")
    return f"{base}/{name}"

def with_twins(cfgs):
    """Thêm mô hình sinh đôi của mọi ngôn ngữ trong mỗi run joint (cùng họ, d, frac, tokenizer, seed); bỏ trùng."""
    out, seen = [], set()
    for c in cfgs:
        group = [c] + [make_cfg(c["grid"], c["family"], "twin", c["d"], lang=l, frac=c["frac"], tok=c["tok"], seed=c["seed"])
                       for l in langs_of(c)]
        for x in group:
            if run_id(x) not in seen:
                seen.add(run_id(x)); out.append(x)
    return out

def grid_configs(name):
    seeds = range(N_SEEDS)
    if name == "smoke":       # SMOKE: mọi họ, đủ để chạy thử phân tích P3 (2 cỡ |L|) và S4 (1 ô)
        out = with_twins([make_cfg("smoke", "decoder", d=D_P3[0], L=L, seed=s) for L in L_LIST for s in seeds])
        out += with_twins([make_cfg("smoke", f, d=D_FAMILY, L=L_LIST[0]) for f in ("moe", "diffusion")])
        return out + [make_cfg("smoke", "decoder", d=D_S4, L=4, tok=f"{S4_LANG}x{S4_TOK_W[-1]:g}", share=f"x{S4_SHARE_MULT:g}")]
    if name == "pilot_data":  # chọn DATA_FRAC: cỡ nhỏ nhất mà đặc trưng eo, ΔA ổn định và twin đủ tốt
        return with_twins([make_cfg("pilot_data", d=384, L=2, frac=f, seed=s) for f in PILOT_FRACS for s in range(2)])
    if name == "p3":          # P3 (2b): ΔA theo |L| và theo d; hai thứ tự lồng A (cân bằng trật tự từ) và B (dồn khác trật tự)
        return with_twins([make_cfg("p3", d=d, L=L, nest=n, seed=s) for d in D_P3 for L in L_LIST for n in NESTINGS for s in seeds])
    if name == "p3_families": # P3 có đúng ngoài decoder không: MoE và diffusion ở một cỡ, kèm twin cùng họ
        return with_twins([make_cfg("p3_families", f, d=D_FAMILY, L=L, seed=s) for f in ("moe", "diffusion") for L in L_LIST for s in seeds])
    if name == "s2":          # S2: căn chỉnh ở eo theo capacity (bậc thang?) và phương sai giữa seed; không cần twin
        return [make_cfg("s2", d=d, L=4, frac=S2_FRAC, seed=s) for d in D_S2 for s in range(N_SEEDS_S2)]
    if name == "s4":          # S4: 2 x 2 = (dữ liệu của X đều / gấp S4_SHARE_MULT) x (tokenizer làm X tốn / rẻ token)
        assert S4_LANG in NESTINGS["A"][:4] and S4_LANG != "eng_Latn", "S4_LANG phải thuộc 4 ngôn ngữ đầu của NESTINGS['A'] và khác tiếng Anh"
        return [make_cfg("s4", d=D_S4, L=4, tok=f"{S4_LANG}x{w:g}", share=sh, seed=s)
                for sh in ("eq", f"x{S4_SHARE_MULT:g}") for w in S4_TOK_W for s in seeds]
    if name == "holdout_p4":  # P4 (phụ): ô giữ ra để kiểm hàm hiệu chỉnh trên p3 trước Study 5 (d mới; |L| = 6 mới)
        return with_twins([make_cfg("holdout_p4", d=d, L=L, seed=s) for d in D_HOLDOUT for L in L_LIST for s in range(2)]
                          + [make_cfg("holdout_p4", d=d, L=6, seed=s) for d in D_P3 for s in range(2)])
    raise KeyError(name)

ALL_GRIDS = ["smoke", "pilot_data", "p3", "p3_families", "s2", "s4", "holdout_p4"]

def token_needs(grids):
    """{khoá tokenizer: tập ngôn ngữ} mà các lưới cần (để stage build_tokens chỉ mã hoá những gì sẽ dùng)."""
    needs = defaultdict(set)
    for g in grids:
        for c in grid_configs(g):
            needs[c["tok"]] |= set(langs_of(c))
    return dict(needs)
