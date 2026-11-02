# Lưu / đọc kết quả, resume, vòng chạy (Cell C1)
# Study 1 — Hourglass ở 4 họ mô hình mở (P1, S6)
# Nguồn: notebooks/archive/study-1.ipynb, cell 7 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load().
# File này là một phần của workspace `hourglass.study1_open_models` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study1_open_models.load(...)`.
# ruff: noqa

# ============================================================
# Cell C1 — Lưu / đọc kết quả, resume, vòng chạy
# Một (dataset, run_id, variant) = một file .npz; có file là coi như đã chạy xong (bỏ qua ở lần sau)
# ============================================================
def run_dir(ds, run_id):
    return os.path.join(STUDY_DIR, ds_tag(ds), run_id)

def variant_path(ds, run_id, variant):
    return os.path.join(run_dir(ds, run_id), f"{variant}.npz")

from hourglass.common.io import read_json

from hourglass.common.io import write_json

def update_meta(ds, run_id, **kv):
    os.makedirs(run_dir(ds, run_id), exist_ok=True)
    p = os.path.join(run_dir(ds, run_id), "meta.json")
    meta = read_json(p, {})
    for k, v in kv.items():
        if isinstance(v, dict) and isinstance(meta.get(k), dict):
            meta[k].update(v)
        else:
            meta[k] = v
    write_json(p, meta)

def save_variant(ds, run_id, variant, arrays):
    os.makedirs(run_dir(ds, run_id), exist_ok=True)
    path = variant_path(ds, run_id, variant)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)                      # ghi nguyên tử: file chỉ xuất hiện khi đã ghi xong

def load_variant(ds, run_id, variant="base"):
    p = variant_path(ds, run_id, variant)
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}

def run_one(ds, key, texts, random_init=False):
    cfg = MODELS[key]
    run_id = key + ("__random" if random_init else "")
    levels = plan_levels(cfg, random_init)
    todo = [(t, vs) for t, vs in levels
            if FORCE_RERUN or not all(os.path.exists(variant_path(ds, run_id, v)) for v in vs)]
    if not todo:
        print(f"[skip] {ds_tag(ds)}/{run_id}: đã có đủ kết quả")
        return "skipped"
    print("=" * 78)
    print(f"[run] {ds_tag(ds)}/{run_id}  ({cfg['id']}{' | RANDOM-INIT' if random_init else ''})  levels còn lại: {[t for t, _ in todo]}")
    t_start = time.time()
    model = tok = None
    try:
        err_path = os.path.join(run_dir(ds, run_id), "error.txt")
        if os.path.exists(err_path):
            os.remove(err_path)
        model, tok = load_model(cfg, random_init)
        if SANITY_CHECK:
            sanity_padding_check(model, tok, cfg, texts)
        N = len(texts[LANGS[0]])
        for t, vs in todo:
            print(f"  -> extract (t={t})")
            ext = extract_level(model, tok, cfg, texts, t=t, n_draws=N_MASK_DRAWS if (t and t > 0) else 1)
            K1, _, _, H = ext["all"].shape
            for v in vs:
                print(f"  -> metrics [{v}]")
                buf = ext["masked"] if v.endswith("_masked") else ext["all"]
                arrays = compute_metrics(buf)
                arrays["langs"] = np.array(list(texts))
                if ext["router"] is not None and v == vs[0]:
                    JS, JS_raw, js_langs, rmean = router_js(ext["router"], list(texts))
                    nl = getattr(model.config, "num_hidden_layers", None) or getattr(model.config, "num_layers", 0)
                arrays.update(JS=JS, JS_raw=JS_raw, JS_langs=js_langs, JS_layers=np.array(ext["gate_layers"]),
                              router_mean=rmean, n_layers=np.array(int(nl)))
                save_variant(ds, run_id, v, arrays)
                update_meta(ds, run_id, variants={v: dict(t=t, pool=("masked" if v.endswith("_masked") else "all"),
                                                          done=datetime.now().isoformat(timespec="seconds"))})
                del arrays
            update_meta(ds, run_id, key=key, hf_id=cfg["id"], family=cfg["family"], label=cfg["label"],
                        random_init=random_init, dataset=ds_tag(ds), n_sent=N, langs=list(texts), K1=int(K1),
                        hidden_dim=int(H), last_layer_pre_norm=bool(ext["stats"]["pre_norm_used"]),
                        n_trunc=int(ext["stats"]["n_trunc"]), n_clip=int(ext["stats"]["n_clip"]),
                        max_len=cfg["max_len"],
                        hparams=dict(n_shuffles=N_SHUFFLES, d_pca=D_PCA, csls_k=CSLS_K, probe_seeds=PROBE_SEEDS,
                                     n_mask_draws=N_MASK_DRAWS, mask_levels=MASK_LEVELS, seed=SEED),
                        versions=dict(torch=torch.__version__, transformers=transformers.__version__))
            if ext["stats"]["n_trunc"]:
                print(f"  [note] {ext['stats']['n_trunc']} câu bị cắt ở max_len={cfg['max_len']}")
            del ext
            gc.collect(); torch.cuda.empty_cache()
        update_meta(ds, run_id, status="ok", seconds=round(time.time() - t_start, 1),
                    finished=datetime.now().isoformat(timespec="seconds"))
        print(f"[done] {ds_tag(ds)}/{run_id} trong {(time.time() - t_start) / 60:.1f} phút")
        return "ok"
    except Exception as e:
        import traceback
        os.makedirs(run_dir(ds, run_id), exist_ok=True)
        with open(os.path.join(run_dir(ds, run_id), "error.txt"), "w") as f:
            f.write(traceback.format_exc())
        update_meta(ds, run_id, status="error")
        print(f"[ERROR] {ds_tag(ds)}/{run_id}: {repr(e)[:300]}  (chi tiết trong error.txt) -> chuyển sang model kế tiếp")
        return "error"
    finally:
        del model, tok
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def scope_keys(ds):
    keys = list(MODELS) if RUN_MODELS is None else list(RUN_MODELS)
    if ds == "ntrex" and NTREX_MODEL_KEYS is not None:
        keys = [k for k in keys if k in NTREX_MODEL_KEYS]
    return keys

def run_all(ds_names=None):
    """Chạy tuần tự: tất cả model pretrained trước, rồi các random-init control. Model đã xong thì bỏ qua."""
    report = {}
    for ds in (ds_names or DATASETS_TO_RUN):
        texts = get_texts(ds, N_SENT)
        keys = scope_keys(ds)
        for key in keys:
            report[(ds_tag(ds), key)] = run_one(ds, key, texts, random_init=False)
        if RUN_RANDOM_BASELINES:
            for fam, key in RANDOM_BASELINE.items():
                if key in keys:
                    report[(ds_tag(ds), key + "__random")] = run_one(ds, key, texts, random_init=True)
    bad = [k for k, v in report.items() if v == "error"]
    print("\nTổng kết:", {s: sum(v == s for v in report.values()) for s in ("ok", "skipped", "error")})
    if bad:
        print("Lỗi ở:", bad, "-> sửa rồi chạy lại cell này, các model đã xong sẽ được bỏ qua.")
    return report

def show_status():
    import pandas as pd
    rows = []
    for ds in DATASETS_TO_RUN:
        for key, cfg in MODELS.items():
            if ds == "ntrex" and NTREX_MODEL_KEYS is not None and key not in NTREX_MODEL_KEYS:
                continue
            for rnd in (False, True):
                if rnd and key not in RANDOM_BASELINE.values():
                    continue
                run_id = key + ("__random" if rnd else "")
                need = [v for _, vs in plan_levels(cfg, rnd) for v in vs]
                have = [v for v in need if os.path.exists(variant_path(ds, run_id, v))]
                err = os.path.exists(os.path.join(run_dir(ds, run_id), "error.txt"))
                st = "ok" if len(have) == len(need) else ("partial" if have else ("error" if err else "missing"))
                rows.append(dict(dataset=ds_tag(ds), run=run_id, family=cfg["family"], status=st,
                                 variants=f"{len(have)}/{len(need)}"))
        return pd.DataFrame(rows)
