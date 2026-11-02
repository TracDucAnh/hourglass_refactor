# Lưu / đọc kết quả, resume, vòng chạy (Cell C1)
# Study 4 — Tần suất, loại hình: từ nào được chia sẻ ở eo (S3 trên văn bản tự nhiên)
# Nguồn: notebooks/archive/study-4.ipynb, cell 21 (mã giữ nguyên văn).
# LIB (định nghĩa): luôn được nạp khi gọi load(); cần GPU/torch.
# File này là một phần của workspace `hourglass.study4_frequency` (một namespace chung như kernel notebook):
# KHÔNG import trực tiếp; dùng `hourglass.study4_frequency.load(...)`.
# ruff: noqa

# ============================================================
# Cell C1 — Lưu / đọc kết quả, resume, vòng chạy
# Mỗi model một thư mục runs/{key}/ với retrieval.npz + meta.json; có retrieval.npz là coi như đã xong (bỏ qua ở lần sau).
# Cache biểu diễn mức từ: cache/{key}/{lang}.npy (resume theo từng ngôn ngữ); xoá sau khi đo nếu KEEP_WORD_CACHE=False.
# ============================================================
from hourglass.common.io import read_json

from hourglass.common.io import write_json

def run_dir(key):
    return f"{WORK_DIR}/runs/{key}"

def _retr_current(rd):
    """Kết quả truy hồi hiện hành? (đúng retr_version và hub_mode). Kết quả cũ -> 'stale' -> chạy lại."""
    m = read_json(f"{rd}/meta.json", {})
    return (os.path.exists(f"{rd}/retrieval.npz") and m.get("retr_version") == RETR_VERSION and m.get("hub_mode") == HUB_MODE
            and m.get("items_sig") == ITEMS_SIG          # v4: ITEMS đổi (từ điển/khoá tra) -> kết quả cũ stale
            and m.get("mask_valid") == MASK_VALID_TRANSLATIONS and m.get("dtype") == str(DTYPE))          # v5: đổi mặt nạ / dtype -> stale

TID = {l: type_ids(WT[MEAS][l]) for l in LANGS}          # id dạng từ theo ngôn ngữ (dùng để che ứng viên cùng dạng)

def run_model(key):
    cfg = MODELS[key]; rd = run_dir(key); os.makedirs(rd, exist_ok=True)
    res_path = f"{rd}/retrieval.npz"
    if _retr_current(rd) and not FORCE_RERUN:
        print(f"[skip] {key}: đã có kết quả"); return "skipped"
    print("=" * 78); print(f"[run] {key} ({cfg['id']})"); t_start = time.time()
    model = None
    try:
        tok = load_tokenizer(cfg)
        plans = {l: plan_language(tok, cfg, TEXTS[MEAS][l], WT[MEAS][l], NEED[l]) for l in LANGS}
        os.makedirs(f"{WORK_DIR}/cache/{key}", exist_ok=True)
        cmeta_p = f"{WORK_DIR}/cache/{key}/meta.json"; cmeta = read_json(cmeta_p, {})
        todo = [l for l in LANGS if FORCE_RERUN or not os.path.exists(cache_path(key, l))]
        n_clip = int(cmeta.get("n_clip", 0))
        if todo:
            model, n_nonemb = load_model(cfg)
            if SANITY_CHECK:       # v5: kết quả sanity được ĐỌC và ghi vào meta (trước đây bị bỏ qua, chỉ in cảnh báo rồi chạy tiếp)
                san = {SHORT[l0]: sanity_batch_vs_single(model, tok, cfg, TEXTS[MEAS][l0], WT[MEAS][l0], NEED[l0]) for l0 in SANITY_LANGS}
                bad = [s_ for s_, v in san.items() if not v["ok"]]
                cmeta.update(sanity=san, sanity_failed=bool(bad), sanity_dtype=str(DTYPE))
                if bad:
                    msg = f"{key}: batch-vs-single lệch hệ thống ở {bad} (dtype={DTYPE})"
                    if SANITY_ON_FAIL == "abort":
                        raise RuntimeError("[SANITY] " + msg)
                    print(f"  [SANITY][CẢNH BÁO] {msg}")
                    if SANITY_ON_FAIL == "fp32_check" and DTYPE != torch.float32 and DEVICE == "cuda":
                        del model; model = None; gc.collect(); torch.cuda.empty_cache()
                        m32, _ = load_model(cfg, dtype=torch.float32)
                        san32 = {SHORT[l0]: sanity_batch_vs_single(m32, tok, cfg, TEXTS[MEAS][l0], WT[MEAS][l0], NEED[l0]) for l0 in SANITY_LANGS}
                        del m32; gc.collect(); torch.cuda.empty_cache()
                        cmeta.update(sanity_fp32=san32, sanity_fp32_ok=all(v["ok"] for v in san32.values()))
                        model, n_nonemb = load_model(cfg)
                        if cmeta["sanity_fp32_ok"]:
                            print("  [SANITY] fp32 QUA nhưng bf16 lệch -> sai khác do số học bf16 (không phải lỗi padding/offset). Cân nhắc EXTRACT_DTYPE='float32' cho model này; "
                                  "kết quả hiện tại được đánh dấu numerics_flag=True.")
                        else:
                            print("  [SANITY] fp32 cũng lệch -> nghi lỗi padding/mask/offset hoặc attention-sink; KHÔNG dùng kết quả của model này cho kết luận cho tới khi điều tra.")
                write_json(cmeta_p, cmeta)
            for l in todo:
                t0 = time.time()
                buf, st = extract_language(model, tok, cfg, plans[l], WT[MEAS][l])
                if buf is None:
                    raise RuntimeError(f"{l}: không có từ nào để trích (kiểm tra tách từ / gióng hàng / offset của tokenizer)")
                arr = buf.cpu().numpy(); del buf
                tmp = cache_path(key, l) + ".tmp"
                with open(tmp, "wb") as f:
                    np.save(f, arr)
                os.replace(tmp, cache_path(key, l))
                n_clip += st["n_clip"]
                cmeta.update(n_nonemb=n_nonemb, n_clip=n_clip, K1=int(arr.shape[0]), d=int(arr.shape[2]), pre_norm_used=bool(st["pre_norm_used"]))
                write_json(cmeta_p, cmeta)
                print(f"  [extract] {SHORT[l]}: {plans[l]['n_rows']:,} từ x {arr.shape[0]} tầng x d={arr.shape[2]} "
                      f"({arr.nbytes / 1e9:.1f}GB, {time.time() - t0:.0f}s, cắt cụt {plans[l]['n_trunc']} câu)", flush=True)
                del arr
            del model; model = None
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
        if n_clip:
            print(f"  [warn] {key}: {n_clip:,} giá trị biểu diễn bị cắt/NaN khi ép fp16 (FP16_CLIP) — kiểm tra meta.json:n_clip")
        print("  -> truy hồi CSLS theo tầng")
        out = compute_retrieval(key, ITEMS, plans, TID)
        for l in LANGS:
            out[f"ntok_{SHORT[l]}"] = plans[l]["ntok"]
        tmp = res_path + ".tmp.npz"
        np.savez_compressed(tmp, **out); os.replace(tmp, res_path)
        fert = {SHORT[l]: float(plans[l]["ntok"][plans[l]["ntok"] > 0].mean()) for l in LANGS}    # sub-token / từ (cả từ không thuộc cặp nào)
        write_json(f"{rd}/meta.json", dict(key=key, hf_id=cfg["id"], family=cfg["family"], label=cfg["label"], group=cfg["group"],
                                           n_params_nonemb=cmeta.get("n_nonemb"), K1=int(out["K1"]), d=cmeta.get("d"),
                                           n_pool=int(out["n_pool"]), fertility=fert, n_clip=n_clip, status="ok",
                                           n_rows={SHORT[l]: plans[l]["n_rows"] for l in LANGS},
                                           seconds=round(time.time() - t_start, 1), finished=datetime.now().isoformat(timespec="seconds"),
                                           versions=dict(torch=torch.__version__, transformers=transformers.__version__),
                                           retr_version=RETR_VERSION, hub_mode=HUB_MODE, axis=AXIS, items_sig=ITEMS_SIG,
                                           mask_valid=MASK_VALID_TRANSLATIONS, rr_variant=RR_VARIANT, dtype=str(DTYPE),
                                           sanity=cmeta.get("sanity"), sanity_fp32=cmeta.get("sanity_fp32"), numerics_flag=bool(cmeta.get("sanity_failed", False)),
                                           hparams=dict(csls_k=CSLS_K, pool_n=POOL_N, pool_reps=POOL_REPS, seed=SEED)))
        if not KEEP_WORD_CACHE:
            shutil.rmtree(f"{WORK_DIR}/cache/{key}", ignore_errors=True)
        print(f"[done] {key} trong {(time.time() - t_start) / 60:.1f} phút")
        return "ok"
    except Exception as e:
        import traceback
        with open(f"{rd}/error.txt", "w") as f:
            f.write(traceback.format_exc())
        print(f"[ERROR] {key}: {repr(e)[:300]} (chi tiết trong error.txt) -> chuyển sang model kế tiếp")
        return "error"
    finally:
        del model
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

def run_all():
    report = {k: run_model(k) for k in scope_models()}
    print("\nTổng kết:", {s: sum(v == s for v in report.values()) for s in ("ok", "skipped", "error")})
    return report

def show_status():
    import pandas as pd
    rows = []
    for k in MODELS:
        done = _retr_current(run_dir(k)); stale = os.path.exists(f"{run_dir(k)}/retrieval.npz") and not done
        n_cache = len(glob.glob(f"{WORK_DIR}/cache/{k}/*.npy"))
        err = os.path.exists(f"{run_dir(k)}/error.txt")
        rows.append(dict(model=k, group=MODELS[k]["group"], status="ok" if done else ("stale (cần chạy lại)" if stale else ("error" if err else ("partial" if n_cache else "missing"))),
                         cache=f"{n_cache}/{len(LANGS)}"))
    return pd.DataFrame(rows)
