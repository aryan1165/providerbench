"""Fetching public benchmark data, cached under ~/.providerbench/cache.

Datasets are fetched by URL from their official homes (Hugging Face datasets-server, GitHub raw);
nothing is vendored into the repo.
"""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .config import HOME

CACHE = HOME / "cache"
HF_ROWS = "https://datasets-server.huggingface.co/rows"
UA = "providerbench/0.1"


TRANSIENT = {408, 425, 429, 500, 502, 503, 504}   # worth waiting out; other 4xx fail at once


def _wait(err, attempt):
    """Seconds before the next try: the server's Retry-After if it gives one, else 2, 4, 8, 16, 30, 30..."""
    ra = getattr(err, "headers", None) and err.headers.get("Retry-After")
    if ra and str(ra).isdigit():
        return min(60, int(ra))
    return min(30, 2 ** (attempt + 1))


def http_get(url, timeout=90, retries=7, binary=False, cache=True):
    """GET with retries; cached forever under CACHE keyed by URL (datasets are versioned).

    Transient failures (timeouts, 429, 5xx such as Hugging Face's 503) are retried with backoff for
    about two minutes; a 404 or other client error fails at once."""
    key = hashlib.sha256(url.encode()).hexdigest()[:24]
    path = CACHE / "http" / key
    if cache and path.exists():
        raw = path.read_bytes()
        return raw if binary else raw.decode("utf-8", "replace")
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
            if cache:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            return raw if binary else raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code not in TRANSIENT:
                break
        except Exception as e:  # noqa: BLE001 - timeouts, resets, DNS: network is flaky, retry
            last = e
        if i < retries - 1:
            time.sleep(_wait(last, i))
    host = urllib.parse.urlsplit(url).netloc
    transient = not isinstance(last, urllib.error.HTTPError) or last.code in TRANSIENT
    hint = " (%s is having trouble; try again in a few minutes, rows already fetched are cached)" % host if transient else ""
    raise RuntimeError("GET %s failed: %s%s" % (url, last, hint))


def hf_count(dataset, split, config="default"):
    q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                "offset": 0, "length": 1})
    return json.loads(http_get(HF_ROWS + "?" + q))["num_rows_total"]


def hf_rows(dataset, split, offset, length, config="default"):
    """Rows [offset, offset+length) of a Hugging Face dataset split, as plain dicts."""
    out = []
    while length > 0:
        n = min(length, 100)
        q = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split,
                                    "offset": offset, "length": n})
        page = json.loads(http_get(HF_ROWS + "?" + q))
        rows = [r["row"] for r in page.get("rows", [])]
        out.extend(rows)
        if len(rows) < n:
            break
        offset, length = offset + n, length - n
    return out


def hf_all(dataset, split, config="default"):
    return hf_rows(dataset, split, 0, hf_count(dataset, split, config), config)


def jsonl(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]
