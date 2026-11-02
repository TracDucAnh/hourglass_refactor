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
