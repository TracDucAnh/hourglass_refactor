# Lưu / đọc, đo một mô hình, sinh đôi, vòng chạy (Cell C2)
# Study 2a — Ngôn ngữ tổng hợp: hourglass có điều khiển (P1, P2, P3, S2–S6)
# Nguồn: notebooks/archive/study-2a.ipynb, cell 27 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study2_synthetic` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study2_synthetic.load(...)`.
# ruff: noqa

# ============================================================
# Cell C2 — Lưu / đọc kết quả, đo một mô hình, sinh đôi, vòng chạy
# ============================================================
from hourglass.common.io import read_json

from hourglass.common.io import write_json

def save_npz(rdir, name, arrays):
    os.makedirs(rdir, exist_ok=True)
    tmp = f"{rdir}/{name}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, f"{rdir}/{name}.npz")

def load_npz(rdir, name):
    p = f"{rdir}/{name}.npz"
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def measure(model, W, rdir, variant, family, masks=False):
    """Đo một mô hình trên tập đo chính + tập dò S3, lưu {variant}.npz. Diffusion: thêm các mức che (S6)."""
    Z, _ = extract_eval(model, W["eval_ids"], W["eval_wid"], desc=f"[{variant}] trích eval")
    arrs = compute_metrics(Z, desc=f"[{variant}] số đo")
    _, Zp = extract_eval(model, W["probe_ids"], W["probe_wid"], pick=W["probe_subj"], desc=f"[{variant}] trích probe S3")
    arrs["G_rank"], arrs["rank_counts"] = rank_alignment(Zp, W["probe_rank"])
    save_npz(rdir, variant, arrs)
    if masks:
        for t in tqdm(MASK_LEVELS[1:], desc=f"[{variant}] mức che", unit="mức", leave=False, dynamic_ncols=True):
            Zt, _ = extract_eval(model, W["eval_ids"], W["eval_wid"], t=t, n_draws=N_MASK_DRAWS, desc=f"[t={t:g}] trích eval")
            save_npz(rdir, f"t{t:g}", compute_metrics(Zt, probes=False, desc=f"[t={t:g}] số đo"))
            del Zt
    return Z

def twin_steps(cfg, W, hp):
    """Số bước của mỗi twin. Cũ: STEPS // L (twin chỉ nhận 1/L số lần cập nhật -> huấn luyện thiếu -> A_indep thấp -> ΔA bị thổi phồng).
    Mới: nhiều nhất bằng mô hình chung, nhưng không quá TWIN_MAX_EPOCH epoch trên N_l câu của riêng ngôn ngữ đó (tránh nhớ thuộc lòng)."""
    L = cfg["L"]
    n_l = int(min(int((W["pool_lang"] == l).sum()) for l in range(L)))
    cap = max(1, round(TWIN_MAX_EPOCH * n_l / hp["batch"]))
    floor_ = max(1, round(_S(500, 30) * (BATCH_REF / hp["batch"]) * hp["seq_mult"]))
    return int(min(hp["steps"], max(hp["steps"] // L, floor_, cap)))

def run_twins(cfg, W, rdir, hp):
    """P3 (Study 2c trên tổng hợp): L mô hình đơn ngữ cùng kiến trúc/tokenizer, mỗi cái chỉ thấy câu của đúng một ngôn ngữ (N_POOL/L câu).
    A_indep[pair, tầng] = Procrustes-CV giữa hai mô hình độc lập; ΔA = A_joint - A_indep (tính ở Phần D).
    Lưu val/trn loss của twin để kiểm twin có hội tụ ngang mô hình chung không."""
    L = cfg["L"]
    steps = twin_steps(cfg, W, hp)
    Zs, t0, losses, vals, trns = [], time.time(), [], [], []
    for l in tqdm(range(L), desc=f"sinh đôi đơn ngữ ({steps} bước/mô hình)", unit="mô hình", leave=False, dynamic_ncols=True):
        torch.manual_seed(2000 + cfg["seed"] * 10 + l)
        tw = build_model(cfg["family"], cfg["d"], cfg["task"], W)
        info = train_model(tw, cfg["family"], cfg["task"], W, steps, cfg["seed"] * 10 + l, lang=l, log=True,
                           desc=f"twin {l + 1}/{L}", leave=False, hp=hp, val_langs=[l])
        losses.append(info["final_lm_loss"]); vals.append(info["val_loss"]); trns.append(info["trn_loss"])
        Zl, _ = extract_eval(tw, W["eval_ids"][l:l + 1], W["eval_wid"][l:l + 1], progress=False)
        Zs.append(Zl[:, 0]); del tw
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    pairs = list(itertools.combinations(range(L), 2))
    A_ind = np.full((len(pairs), Zs[0].shape[0]), np.nan, np.float32); C_ind = A_ind.copy()
    for pi, (a, b) in enumerate(tqdm(pairs, desc="Procrustes giữa các cặp sinh đôi", unit="cặp", leave=False, dynamic_ncols=True)):
        A_ind[pi], C_ind[pi] = procrustes_between(Zs[a], Zs[b])
    save_npz(rdir, "twins", dict(A_indep=A_ind, CKA_indep=C_ind, steps=np.array(steps), loss=np.array(losses, np.float32),
                                 val_loss=np.array(vals, np.float32), trn_loss=np.array(trns, np.float32)))
    return round(time.time() - t0, 1), dict(twin_steps=steps, twin_val=vals, twin_trn=trns)

def run_one(cfg):
    rid = run_id(cfg); rdir = f"{RUN_DIR}/{rid}"; meta_p = f"{rdir}/meta.json"
    _m0 = read_json(meta_p, {})
    if (not FORCE_RERUN and _m0.get("status") == "ok" and _m0.get("metric_version") == METRIC_VERSION
            and _m0.get("train_version") == TRAIN_VERSION):   # run đo / huấn luyện bằng phiên bản cũ -> chạy lại
        tqdm.write(f"[skip] {rid}"); return "skipped"
    os.makedirs(rdir, exist_ok=True)
    t0 = time.time()
    tqdm.write("=" * 78); tqdm.write(f"[run] {rid}")
    try:
        if os.path.exists(f"{rdir}/error.txt"):
            os.remove(f"{rdir}/error.txt")
        write_json(meta_p, dict(cfg=cfg, status="running"))
        W = get_world(world_cfg(cfg["L"], cfg["level"], cfg["overlap"], cfg["len_scale"], cfg["seed"],
                                cfg["len_scale_X"], cfg["share_X"]))
        fam, task, d = cfg["family"], cfg["task"], cfg["d"]
        torch.manual_seed(1000 + cfg["seed"])
        model = build_model(fam, d, task, W)
        n_par = model.n_params_nonemb()
        tqdm.write(f"  tham số không-embedding: {n_par / 1e6:.2f}M | fertility: {[round(x, 2) for x in W['meta']['fertility']]}")
        measure(model, W, rdir, "init", fam)                                   # null: mô hình chưa huấn luyện
        hp = train_hparams(cfg.get("bs"), cfg.get("seq_mult"))
        tqdm.write(f"  giao thức v{TRAIN_VERSION}: batch={hp['batch']} x {hp['steps']} bước "
                   f"({hp['batch'] * hp['steps'] / W['pool_ids'].shape[0]:.1f} epoch)  lr_scale={hp['lr_scale']:.2f}  warmup={hp['warmup']}")
        info = train_model(model, fam, task, W, hp["steps"], cfg["seed"], hp=hp)
        val = eval_lm_loss(model, W)
        Vtot = W["meta"]["vocab_total"]
        val_by_lang = [lm_loss_on(model, fam, get_val_pool(W, [l]), Vtot) for l in range(cfg["L"])]
        floor = entropy_floor(W) if fam in ("decoder", "moe") else None
        conv = convergence_summary(info, floor)
        tqdm.write(f"  hội tụ: val={info['val_loss']:.3f} trn={info['trn_loss']:.3f} (gap {conv['val_trn_gap']:+.3f}) | "
                   f"val còn giảm {conv['conv_drop']:.3f} nats trong ~1/6 bước cuối (ngưỡng {CONV_TOL})"
                   + (f" | sàn≈{floor:.2f}, cách sàn {conv['gap_to_floor']:.2f}" if floor is not None else ""))
        if not conv["converged"]:
            tqdm.write("  [CẢNH BÁO] chưa hội tụ theo tiêu chí val -> tăng SEQ_MULT hoặc giảm BATCH (xem pilot_conv)")
        if conv["overfit"]:
            tqdm.write("  [CẢNH BÁO] val >> trn: nghi nhớ thuộc lòng -> cần pool lớn hơn / sinh mẫu mới, không phải thêm epoch")
        measure(model, W, rdir, "trained", fam, masks=(fam == "diffusion"))
        extra = {}
        if cfg["lens"]:
            assert cfg["overlap"] == 0 and fam in ("decoder", "moe")
            save_npz(rdir, "lens", dict(M=logit_lens_mass(model, W["eval_ids"], W["tok_lang"])))
        del model
        if cfg["twins"]:
            sec, tinfo = run_twins(cfg, W, rdir, hp)
            extra["twin_seconds"] = sec; extra.update(tinfo)
            extra["twin_excess_loss"] = float(np.mean(np.array(tinfo["twin_val"]) - np.array(val_by_lang)))
            if extra["twin_excess_loss"] > 0.10:
                tqdm.write(f"  [CẢNH BÁO] twin kém mô hình chung {extra['twin_excess_loss']:.2f} nats (thiếu dữ liệu/bước) -> ΔA có thể bị thổi phồng")
        write_json(meta_p, dict(cfg=cfg, status="ok", seconds=round(time.time() - t0, 1), params_nonemb=n_par,
                                final_lm_loss=info["final_lm_loss"], eval_lm_loss=val, steps=info["steps"], batch=hp["batch"], lr_scale=hp["lr_scale"],
                                seq_mult=hp["seq_mult"], hp=hp, train_version=TRAIN_VERSION, val_loss=info["val_loss"], trn_loss=info["trn_loss"],
                                val_by_lang=val_by_lang, floor_nats=floor, history=info["history"], **conv,
                                fertility=W["meta"]["fertility"], world=W["meta"]["id"], verify=W["meta"]["verify"],
                                finished=datetime.now().isoformat(timespec="seconds"),
                                metric_version=METRIC_VERSION, versions=dict(torch=torch.__version__), **extra))
        tqdm.write(f"[done] {rid} trong {(time.time() - t0) / 60:.1f} phút  (loss LM ≈ {info['final_lm_loss']:.3f}, eval {val})")
        return "ok"
    except Exception as e:
        with open(f"{rdir}/error.txt", "w") as f:
            f.write(traceback.format_exc())
        write_json(meta_p, dict(cfg=cfg, status="error"))
        tqdm.write(f"[ERROR] {rid}: {repr(e)[:300]}  (chi tiết: error.txt) -> chuyển sang run kế tiếp")
        return "error"
    finally:
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def run_grid(name, limit=None):
    cfgs = grid_configs(name)[: limit]
    rep = {}
    bar = tqdm(cfgs, desc=f"lưới {name}", unit="run", dynamic_ncols=True)
    for c in bar:
        rid = run_id(c)
        rep[rid] = run_one(c)
        bar.set_postfix({s: sum(v == s for v in rep.values()) for s in ("ok", "skipped", "error")}, refresh=True)
    print("\nTổng kết", name, {s: sum(v == s for v in rep.values()) for s in ("ok", "skipped", "error")})
    bad = [k for k, v in rep.items() if v == "error"]
    if bad:
        print("Lỗi ở:", bad, "-> sửa rồi chạy lại cell này; run đã xong sẽ được bỏ qua.")
    return rep

def show_status():
    import pandas as pd
    rows = []
    for g in ALL_GRIDS:
        cfgs = grid_configs(g)
        st = [read_json(f"{RUN_DIR}/{run_id(c)}/meta.json", {}).get("status", "missing") for c in cfgs]
        secs = [read_json(f"{RUN_DIR}/{run_id(c)}/meta.json", {}).get("seconds") for c in cfgs]
        secs = [s for s in secs if s]
        rows.append(dict(grid=g, runs=len(cfgs), ok=st.count("ok"), error=st.count("error"), running=st.count("running"),
                         mean_min=round(float(np.mean(secs)) / 60, 1) if secs else None))
    return pd.DataFrame(rows)
