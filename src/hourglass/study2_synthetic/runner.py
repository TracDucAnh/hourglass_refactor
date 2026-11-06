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

def measure(model, W, rdir, variant, family):
    """Đo một mô hình trên tập đo chính + tập dò S3, lưu {variant}.npz.
    Biểu diễn câu: họ hai chiều (encoder, diffusion) = trung bình theo từ; họ causal (decoder, MoE) = trạng thái ở token nội dung
    cuối, vị trí duy nhất đã thấy cả câu (trung bình theo từ trộn các tiền tố khác nhau khi trật tự từ khác nhau).
    Họ causal lưu thêm {variant}_mean.npz = số đo theo trung bình theo từ (như họ hai chiều / Study 1) để đối chiếu cách đo."""
    Z, _, Zl = extract_eval(model, W["eval_ids"], W["eval_wid"], last=True, desc=f"[{variant}] trích eval")
    if model.causal:
        save_npz(rdir, f"{variant}_mean", compute_metrics(Z, desc=f"[{variant}] số đo (trung bình theo từ)"))
    Z = Zl if model.causal else Z
    arrs = compute_metrics(Z, desc=f"[{variant}] số đo")
    _, Zp = extract_eval(model, W["probe_ids"], W["probe_wid"], pick=W["probe_subj"], desc=f"[{variant}] trích probe S3")
    arrs["G_rank_w"], arrs["rank_counts"] = rank_alignment(Zp, W["probe_rank"])
    save_npz(rdir, variant, arrs)
    return Z

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
                                cfg["len_scale_X"], cfg["share_X"], cfg.get("anchor", 0.0)))
        fam, task, d = cfg["family"], cfg["task"], cfg["d"]
        torch.manual_seed(1000 + cfg["seed"])
        model = build_model(fam, d, task, W)
        n_par = model.n_params_nonemb()
        tqdm.write(f"  tham số không-embedding: {n_par / 1e6:.2f}M | fertility: {[round(x, 2) for x in W['meta']['fertility']]}")
        measure(model, W, rdir, "init", fam)                                   # null: mô hình chưa huấn luyện
        sm = cfg.get("seq_mult") if cfg.get("seq_mult") is not None else (SEQ_MULT_FAMILY or {}).get(fam)   # ngân sách theo họ
        hp = train_hparams(cfg.get("bs"), sm)
        hp.update(in_mask=cfg.get("in_mask", 0.0), mlm_corrupt=cfg.get("mlm_corrupt", "mask"))
        tqdm.write(f"  giao thức v{TRAIN_VERSION}: batch={hp['batch']} x {hp['steps']} bước "
                   f"({hp['batch'] * hp['steps'] / W['pool_ids'].shape[0]:.1f} epoch)  lr_scale={hp['lr_scale']:.2f}  warmup={hp['warmup']}")
        info = train_model(model, fam, task, W, hp["steps"], cfg["seed"], hp=hp)
        val = eval_lm_loss(model, W)
        Vtot = W["meta"]["vocab_total"]
        val_by_lang = [lm_loss_on(model, fam, get_val_pool(W, [l]), Vtot, opts=hp) for l in range(cfg["L"])]
        floor = entropy_floor(W) if fam in ("decoder", "moe") else None
        conv = convergence_summary(info, floor)
        tqdm.write(f"  hội tụ: val={info['val_loss']:.3f} trn={info['trn_loss']:.3f} (gap {conv['val_trn_gap']:+.3f}) | "
                   f"val còn giảm {conv['conv_drop']:.3f} nats trong ~1/6 bước cuối (ngưỡng {CONV_TOL})"
                   + (f" | sàn≈{floor:.2f}, cách sàn {conv['gap_to_floor']:.2f}" if floor is not None else ""))
        if not conv["converged"]:
            tqdm.write("  [CẢNH BÁO] chưa hội tụ theo tiêu chí val -> tăng SEQ_MULT hoặc giảm BATCH (xem pilot_conv)")
        if conv["overfit"]:
            tqdm.write("  [CẢNH BÁO] val >> trn: nghi nhớ thuộc lòng -> cần pool lớn hơn / sinh mẫu mới, không phải thêm epoch")
        measure(model, W, rdir, "trained", fam)
        del model
        write_json(meta_p, dict(cfg=cfg, status="ok", seconds=round(time.time() - t0, 1), params_nonemb=n_par,
                                final_lm_loss=info["final_lm_loss"], eval_lm_loss=val, steps=info["steps"], batch=hp["batch"], lr_scale=hp["lr_scale"],
                                seq_mult=hp["seq_mult"], hp=hp, train_version=TRAIN_VERSION, val_loss=info["val_loss"], trn_loss=info["trn_loss"],
                                val_by_lang=val_by_lang, floor_nats=floor, history=info["history"], **conv,
                                fertility=W["meta"]["fertility"], world=W["meta"]["id"], verify=W["meta"]["verify"],
                                finished=datetime.now().isoformat(timespec="seconds"),
                                metric_version=METRIC_VERSION, versions=dict(torch=torch.__version__)))
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
    cfgs = [c for c in grid_configs(name)[: limit] if FAMILY is None or c["family"] == FAMILY]
    if SHARD:
        k, n = map(int, str(SHARD).split("/"))
        cfgs = cfgs[k::n]
    if not cfgs:
        return {}
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
        cfgs = [c for c in grid_configs(g) if FAMILY is None or c["family"] == FAMILY]
        if not cfgs:
            continue
        st = [read_json(f"{RUN_DIR}/{run_id(c)}/meta.json", {}).get("status", "missing") for c in cfgs]
        secs = [read_json(f"{RUN_DIR}/{run_id(c)}/meta.json", {}).get("seconds") for c in cfgs]
        secs = [s for s in secs if s]
        rows.append(dict(grid=g, runs=len(cfgs), ok=st.count("ok"), error=st.count("error"), running=st.count("running"),
                         mean_min=round(float(np.mean(secs)) / 60, 1) if secs else None))
    return pd.DataFrame(rows)
