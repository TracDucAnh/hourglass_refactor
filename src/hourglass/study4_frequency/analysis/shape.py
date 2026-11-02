# Kiểm định hình dạng v6 (Cell C4b)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 29 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C4b — (v6) Kiểm định hình dạng mới trên đường cong bootstrap đã lưu (không cần GPU). CHẠY TRƯỚC C5.
# Vì sao đổi (đọc kèm kết quả v5):
#   - p_LM = max(p_left_up, p_right_down, p_not_interior) đo độ dốc ở 20% NGOÀI CÙNG của lưới x. Nhánh phải rơi đúng vùng spline cong ngược / khối hapax:
#     XLM-R large có p_left_up=0.000, đỉnh 733–1195 rất chắc, nhưng p_right_down=0.56 -> bị gán "không phải U ngược". Một điều kiện đuôi quyết định mọi nhãn.
#   - amp_q >= 0 theo cấu trúc; Kendall với n=4 có p_min=0.083 (không bao giờ < 0.05); cờ robust_to_lang_choice so NHÃN CHỮ nên p=0.04 vs 0.09 đảo nhãn.
# Cách mới (áp đồng loạt cho mọi model, không chọn theo kết quả):
#   - Cửa sổ [xL, xR] theo PHÂN VỊ DỮ LIỆU của log-hạng, quét cả lưới LEFT_QS x RIGHT_QS thay vì một lựa chọn duy nhất.
#   - x̂ = đỉnh của đường cong QUAN SÁT trong cửa sổ, cố định qua các bootstrap -> rise/fall có thể ÂM (không thiên lệch dương như max qua từng bootstrap).
#   - rise = f(x̂) − f(xL); fall = f(x̂) − f(xR); p một phía = (1 + #{<=0}) / (B + 1). Báo độ lớn + CI. Nhãn chỉ để đọc, KHÔNG phải cổng qua/trượt.
#   - Đỉnh dính biên cửa sổ (>= 50% bootstrap) = bị CẮT (censored): "≤ giới hạn trái" / "≥ giới hạn phải", không coi là r* thật.
# Giới hạn cần biết: x̂ chọn trên chính dữ liệu đang kiểm nên hơi lạc quan cho "rise/fall" (không có tách mẫu); giữ nguyên bootstrap theo cụm khái niệm của C5.
# ============================================================
ANCHOR_QS   = (0.02, 0.05, 0.10, 0.80, 0.90, 0.95)   # phân vị của log-hạng (dữ liệu sau prep_main) dùng làm mốc cửa sổ; C5 lưu các mốc này cho từng tập con
LEFT_QS     = (0.02, 0.05, 0.10)
RIGHT_QS    = (0.80, 0.90, 0.95)
MAIN_ANCHOR = (0.05, 0.90)                           # chỉ để HIỂN THỊ; độ bền đọc ở toàn bộ lưới LEFT_QS x RIGHT_QS
EDGE_MARGIN = 0.05                                   # đỉnh nằm trong 5% chiều rộng cửa sổ tính từ mép = "dính biên"
ALPHA_SHAPE = 0.05                                   # chỉ dùng để đặt nhãn đọc và tính tỉ lệ cửa sổ có p < alpha

def pack_subcurves(SUBC):
    """Đóng gói đường cong (quan sát + bootstrap) của mọi tập con để lưu vào npz (C5 gọi). Khoá theo chỉ số, tên gốc nằm trong sc_names."""
    names = list(SUBC)
    out = {"sc_names": np.array(names)}
    for i, nm in enumerate(names):
        d = SUBC[nm]
        out[f"sc_grid__{i}"] = np.asarray(d["grid"], np.float32)
        out[f"sc_fobs__{i}"] = np.asarray(d["f_obs"], np.float32)
        out[f"sc_fboot__{i}"] = np.asarray(d["f_boot"])          # float16 như gam_analysis trả về (sai số lượng tử ~1e-4, nhỏ hơn nhiều so với CI)
        out[f"sc_xq__{i}"] = np.asarray(d["xq"], np.float32)
    return out

_X_DATA = None
def _x_data():
    """log-hạng của các từ trong phân tích chính (sau prep_main). Không phụ thuộc model (cùng ITEMS, cùng cặp)."""
    global _X_DATA
    if _X_DATA is None:
        k0 = done_models()[0]
        pair_muse = [pi for pi in range(P_) if PAIR_STATS.confirm_src.iloc[pi] in MAIN_SRCS]
        main_muse = MAIN_PAIRS == "muse" and 0 < len(pair_muse) < P_
        d0, R0 = word_table(k0, pair_muse) if main_muse else word_table(k0)
        d, _ = prep_main(d0, R0)
        _X_DATA = d.log_rank.values.astype(float)
    return _X_DATA

def load_curve(key, name="all"):
    """Đường cong tại tầng waist của model `key`, tập con `name` -> dict(grid, f_obs[G], f_boot[B,G], xq{phân vị: x}) hoặc None.
    npz mới (C5 v6) có đủ mọi tập con. npz cũ (v5) chỉ có 'all' và 'gam_bare' -> mốc xq tính lại từ _x_data()."""
    p = f"{ANALYSIS_DIR}/s3_{key}{RUN_TAG}.npz"
    if not os.path.exists(p):
        return None
    z = np.load(p)
    if "sc_names" in z.files:
        names = [str(s) for s in z["sc_names"]]
        if name in names:
            i = names.index(name)
            return dict(grid=z[f"sc_grid__{i}"].astype(float), f_obs=z[f"sc_fobs__{i}"].astype(float), f_boot=z[f"sc_fboot__{i}"].astype(np.float32),
                        xq=dict(zip(ANCHOR_QS, z[f"sc_xq__{i}"].astype(float))))
    xq = dict(zip(ANCHOR_QS, np.quantile(_x_data(), ANCHOR_QS)))
    if name == "all":
        return dict(grid=z["grid"].astype(float), f_obs=z["f_obs"][int(z["kw"])].astype(float), f_boot=z["f_boot_w"].astype(np.float32), xq=xq)
    if name == "gam_bare" and "f_boot_w_bare" in z.files:
        return dict(grid=z["grid"].astype(float), f_obs=z["f_obs_bare"].astype(float), f_boot=z["f_boot_w_bare"].astype(np.float32), xq=xq)
    return None

def peak_in_window(g, fo, fb, xL, xR, margin=EDGE_MARGIN):
    """Đỉnh trong cửa sổ [xL, xR]: x̂ (từ đường quan sát), đỉnh của từng bootstrap (xb), tỉ lệ bootstrap có đỉnh dính mép trái / phải."""
    g = np.asarray(g, float)
    gi = np.where((g >= xL) & (g <= xR))[0]
    if len(gi) < 10:
        return None
    kh = int(gi[np.argmax(fo[gi])])
    xb = g[gi[np.argmax(fb[:, gi], axis=1)]]
    w = g[gi[-1]] - g[gi[0]]
    return dict(gi=gi, kh=kh, x_hat=float(g[kh]), xb=xb,
                edge_l=float((xb <= g[gi[0]] + margin * w).mean()), edge_r=float((xb >= g[gi[-1]] - margin * w).mean()))

def shape_contrasts(g, fo, fb, xL, xR, margin=EDGE_MARGIN):
    """rise = f(x̂) − f(xL), fall = f(x̂) − f(xR) với x̂ CỐ ĐỊNH; CI percentile + p một phía (cộng một). p_iut = max(p_rise, p_fall)."""
    pk = peak_in_window(g, fo, fb, xL, xR, margin)
    if pk is None:
        return None
    gi, kh = pk["gi"], pk["kh"]; iL, iR = int(gi[0]), int(gi[-1]); B = fb.shape[0]
    rise_b, fall_b = fb[:, kh] - fb[:, iL], fb[:, kh] - fb[:, iR]
    pv = lambda c: (1.0 + float(c)) / (B + 1.0)
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    xlo, xhi = ci(pk["xb"]); rlo, rhi = ci(rise_b); flo, fhi = ci(fall_b)
    r = dict(x_hat=pk["x_hat"], x_lo=xlo, x_hi=xhi, edge_l=pk["edge_l"], edge_r=pk["edge_r"],
             rise=float(fo[kh] - fo[iL]), rise_lo=rlo, rise_hi=rhi, fall=float(fo[kh] - fo[iR]), fall_lo=flo, fall_hi=fhi,
             p_rise=pv((rise_b <= 0).sum()), p_fall=pv((fall_b <= 0).sum()))
    r["p_iut"] = max(r["p_rise"], r["p_fall"])
    return r

def shape_label(r, a=ALPHA_SHAPE):
    """Nhãn để ĐỌC (không phải cổng): hai câu hỏi tăng / giảm trả lời riêng, đỉnh dính biên được nêu rõ."""
    up, dn = r["p_rise"] < a, r["p_fall"] < a
    if up and dn:
        return "inverted-U"
    if up:
        return "rises; no measurable fall (peak at right limit)" if r["edge_r"] >= 0.5 else "rises; tail flat"
    if dn:
        return "falls from left limit (peak at left limit)" if r["edge_l"] >= 0.5 else "head flat; tail falls"
    return "flat / undetermined"
