# Một run: huấn luyện, lưu checkpoint, đo; vòng lưới có resume (Study 2b)
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần torch. Study này không có notebook gốc.
# File này là một phần của workspace `hourglass.study3_natural` (một namespace chung): KHÔNG import trực tiếp.

def save_npz(rdir, name, arrays):
    os.makedirs(rdir, exist_ok=True)
    np.savez_compressed(f"{rdir}/{name}.tmp.npz", **arrays)
    os.replace(f"{rdir}/{name}.tmp.npz", f"{rdir}/{name}.npz")

def run_one(c):
    """Joint: đo bản khởi tạo (null) -> huấn luyện (hoặc nạp checkpoint đã có) -> lưu checkpoint HF -> đo.
    Twin: huấn luyện / nạp -> đo biểu diễn ngôn ngữ của nó. Checkpoint được giữ lại, nên đổi cách đo
    (METRIC_VERSION) chỉ cần đo lại, không phải huấn luyện lại."""
    rid = run_id(c); rdir = f"{RUN_DIR}/{rid}"; meta_p = f"{rdir}/meta.json"
    m0 = read_json(meta_p, {}) or {}
    if (not FORCE_RERUN and m0.get("status") == "ok" and m0.get("train_version") == TRAIN_VERSION
            and m0.get("metric_version") == METRIC_VERSION):
        return "skipped"
    os.makedirs(rdir, exist_ok=True); t0 = time.time()
    lock = f"{rdir}/.lock"                # một twin có thể thuộc nhiều lưới: nhiều GPU chạy song song không huấn luyện trùng
    try:
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        tqdm.write(f"[skip] {rid}: đang chạy ở tiến trình khác (xoá {lock} nếu tiến trình đó đã chết)")
        return "skipped"
    tqdm.write("=" * 78 + f"\n[run] {rid}")
    try:
        write_json(meta_p, dict(cfg=c, status="running"))
        tok, fam = get_tokenizer(c["tok"]), c["family"]
        epochs = (TWIN_EPOCHS if c["kind"] == "twin" else JOINT_EPOCHS) * EPOCH_MULT_FAMILY.get(fam, 1.0)
        st = Streams(c, epochs)
        if c["kind"] == "joint":
            m_init = build_model(fam, c["d"], len(tok), seed=1000 + c["seed"])
            save_npz(rdir, "init", measure_run(m_init, c, "init")); del m_init
        ck = ckpt_path(c); ck_meta = read_json(f"{ck}/hourglass_train.json", {}) or {}
        if not FORCE_RERUN and ck_meta.get("train_version") == TRAIN_VERSION:
            model, info = load_model(ck, fam), ck_meta["info"]
            tqdm.write(f"  nạp checkpoint {ck}")
        else:
            model = build_model(fam, c["d"], len(tok), seed=1000 + c["seed"])
            tqdm.write(f"  {n_params_nonemb(model) / 1e6:.1f}M tham số không-embedding | {st.steps} bước x {BATCH_SEQS} x {SEQ_LEN} "
                       f"({epochs:g} epoch) | token / ngôn ngữ: {st.tokens_per_lang}")
            info = train_model(model, fam, c, st, c["seed"], desc=rid)
            model.save_pretrained(ck); tok.save_pretrained(ck)
            write_json(f"{ck}/hourglass_train.json", dict(cfg=c, train_version=TRAIN_VERSION, info=info))
        tqdm.write(f"  val={info['val_loss']:.3f} | val còn giảm {info['conv_drop']:.3f} nats ở ~15% bước cuối (ngưỡng {CONV_TOL})"
                   + ("" if info["converged"] else " -> CHƯA HỘI TỤ"))
        save_npz(rdir, "trained", measure_run(model, c, "trained"))
        write_json(meta_p, dict(cfg=c, rid=rid, status="ok", langs=langs_of(c), ckpt=ck, params_nonemb=n_params_nonemb(model),
                                train_version=TRAIN_VERSION, metric_version=METRIC_VERSION, epochs=epochs,
                                seconds=round(time.time() - t0, 1), finished=datetime.now().isoformat(timespec="seconds"),
                                **{k: info[k] for k in ("steps", "val_loss", "val_by_lang", "conv_drop", "converged", "tokens_per_lang", "history")}))
        del model
        return "ok"
    except Exception as e:
        with open(f"{rdir}/error.txt", "w") as f:
            f.write(traceback.format_exc())
        write_json(meta_p, dict(cfg=c, status="error"))
        tqdm.write(f"[ERROR] {rid}: {repr(e)[:300]} (chi tiết: error.txt)")
        return "error"
    finally:
        if os.path.exists(lock):
            os.remove(lock)
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def grid_selection(name):
    cfgs = [c for c in grid_configs(name) if FAMILY is None or c["family"] == FAMILY]
    if SHARD:
        k, n = map(int, str(SHARD).split("/"))
        cfgs = cfgs[k::n]
    return cfgs

def run_grid(name):
    rep = {run_id(c): run_one(c) for c in tqdm(grid_selection(name), desc=f"lưới {name}", unit="run", dynamic_ncols=True)}
    print(f"\nTổng kết {name}:", {s: sum(v == s for v in rep.values()) for s in ("ok", "skipped", "error")})
    return rep

def show_status():
    import pandas as pd
    rows = []
    for g in RUN_GRIDS:
        st = [(read_json(f"{RUN_DIR}/{run_id(c)}/meta.json", {}) or {}).get("status", "missing")
              for c in grid_configs(g) if FAMILY is None or c["family"] == FAMILY]
        rows.append(dict(grid=g, runs=len(st), **{s: st.count(s) for s in ("ok", "running", "error", "missing")}))
    print(pd.DataFrame(rows).to_string(index=False))
