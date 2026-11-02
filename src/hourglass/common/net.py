"""Tải HTTP đơn giản (dùng cho NTREX-128 trên GitHub)."""

import urllib.request


def _http_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()
