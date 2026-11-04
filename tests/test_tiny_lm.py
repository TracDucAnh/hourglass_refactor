"""Study 2a — TinyLM: shape đầu ra của 4 họ và overfit được một batch (loss giảm)."""

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def ws():
    from hourglass.study2_synthetic import load

    return load("study2a_smoke", verbose=False)


def _batch(ws, B=8, n_tok=12, V=None):
    g = torch.Generator().manual_seed(0)
    ids = torch.full((B, ws.MAX_LEN), ws.PAD, dtype=torch.long)
    ids[:, 0] = ws.BOS
    ids[:, 1 : n_tok + 1] = torch.randint(ws.SYM0, V, (B, n_tok), generator=g)
    ids[:, n_tok + 1] = ws.EOS
    return ids


@pytest.mark.parametrize("family", ["decoder", "encoder", "moe", "diffusion"])
def test_forward_shapes(ws, family):
    V, d = ws.SYM0 + 50, 64
    model = ws.TinyLM(family, d, V).to(ws.DEVICE)
    ids = _batch(ws, B=3, V=V).to(ws.DEVICE)
    logits, hs, hn = model(ids)
    assert logits.shape == (3, ws.MAX_LEN, V)
    assert len(hs) == ws.K_LAYERS + 1 and all(h.shape == (3, ws.MAX_LEN, d) for h in hs)
    assert hn.shape == (3, ws.MAX_LEN, d)


def test_decoder_overfits_one_batch(ws):
    torch.manual_seed(0)
    V = ws.SYM0 + 40
    model = ws.TinyLM("decoder", 64, V).to(ws.DEVICE)
    ids = _batch(ws, V=V).to(ws.DEVICE)
    n = (ids != ws.PAD).sum(1)
    pool = dict(ids=ids, len=n, sem=torch.zeros_like(n), surf=torch.zeros_like(n))
    idx = torch.arange(ids.shape[0], device=ws.DEVICE)
    g = torch.Generator(device=ws.DEVICE).manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(60):
        loss, lm = ws.compute_loss(model, pool, idx, "decoder", "LM", g, V)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(lm))
    assert losses[-1] < 0.5 * losses[0], losses[::10]


def _capture(ws, family, V, opts):
    """Chạy compute_loss với mô hình giả, trả về tensor đầu vào mà mô hình nhận được."""
    seen = {}

    class Spy(torch.nn.Module):
        def forward(self, x):
            seen["inp"] = x.clone()
            B, T = x.shape
            return torch.zeros(B, T, V, requires_grad=True), None, torch.zeros(B, T, 4)

    ids = _batch(ws, B=64, V=V).to(ws.DEVICE)
    n = (ids != ws.PAD).sum(1)
    pool = dict(ids=ids, len=n, sem=torch.zeros_like(n), surf=torch.zeros_like(n))
    g = torch.Generator(device=ws.DEVICE).manual_seed(0)
    ws.compute_loss(Spy(), pool, torch.arange(64, device=ws.DEVICE), family, "LM", g, V, opts)
    return ids, seen["inp"]


def test_decoder_input_mask_only_touches_content(ws):
    V = ws.SYM0 + 40
    ids, inp = _capture(ws, "decoder", V, {"in_mask": 0.15})
    src = ids[:, :-1]
    masked = inp == ws.MASK
    assert masked.any() and (src[masked] >= ws.SYM0).all()  # chỉ token nội dung bị che
    assert torch.equal(inp[~masked], src[~masked])  # phần còn lại giữ nguyên
    frac = float(masked.sum() / (src >= ws.SYM0).sum())
    assert 0.10 < frac < 0.20, frac
    _, clean = _capture(ws, "decoder", V, None)
    assert not (clean == ws.MASK).any()  # mặc định: không che đầu vào


def test_encoder_shuffle_corruption_has_no_mask_and_stays_in_sentence(ws):
    V = ws.SYM0 + 4000  # nhiều ký hiệu -> mỗi câu có tập token gần như riêng
    ids, inp = _capture(ws, "encoder", V, {"mlm_corrupt": "shuffle"})
    assert not (inp == ws.MASK).any()
    changed = (inp != ids) & (ids >= ws.SYM0)
    # token thay thế phần lớn lấy từ chính câu đó (10% còn lại là token ngẫu nhiên toàn bộ từ vựng, như MLM chuẩn)
    same_row = [bool((ids[b] == inp[b, j]).any()) for b, j in changed.nonzero().tolist()]
    assert changed.any() and sum(same_row) / len(same_row) > 0.8
    _, std = _capture(ws, "encoder", V, None)
    assert (std == ws.MASK).any()  # mặc định: MLM chuẩn với [MASK]
