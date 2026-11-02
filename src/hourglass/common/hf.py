"""Tiện ích Hugging Face dùng chung: fallback trust_remote_code, tìm final norm."""


def _with_trc_fallback(fn, cfg):
    try:
        return fn(cfg["trc"])
    except Exception as e:
        print(f"  [warn] load lỗi với trust_remote_code={cfg['trc']}: {repr(e)[:160]} -> thử lại với {not cfg['trc']}")
        return fn(not cfg["trc"])


def get_final_norm(model, cfg):
    path = cfg.get("final_norm")
    if not path:
        return None
    m = model
    try:
        for p in path.split("."):
            m = getattr(m, p)
        return m
    except AttributeError:
        print(f"  [warn] không thấy final norm '{path}' -> tầng cuối = hidden state sau norm")
        return None
