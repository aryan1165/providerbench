"""Providers, models, keys and where data lives.

providers/<id>.toml   one file per provider                                  in the repo
~/.providerbench/providers/<id>.toml   your own endpoints (self-hosted, Bedrock, ...), never committed
models.toml           canonical model names, aliases, OpenRouter slugs        in the repo
.env                  API keys (KEY_ENV=value), git-ignored; real environment variables win
~/.providerbench/     run history, uploads, prices, dataset cache (PB_HOME overrides)

A target is "<provider id>:<model>", model = a canonical name or the provider's raw model id.
"""
import copy
import fnmatch
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
PROVIDERS_DIR = ROOT / "providers"
HOME = Path(os.environ.get("PB_HOME") or (Path.home() / ".providerbench")).expanduser()
ENV_FILE = ROOT / ".env"
USER_PROVIDERS_DIR = HOME / "providers"


def home(*parts):
    """A path inside the private data directory (created 0700 on first use)."""
    if not HOME.exists():
        HOME.mkdir(parents=True, mode=0o700)
    return HOME.joinpath(*parts)


def load_env():
    """Read .env into os.environ (existing environment variables win)."""
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def save_keys(values):
    """Merge {ENV_NAME: value} into .env, mode 0600. Empty values are ignored. Used by the web UI."""
    lines, seen = [], set()
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            k = line.split("=", 1)[0].strip() if "=" in line else None
            if k in values and values[k]:
                lines.append("%s=%s" % (k, values[k]))
                seen.add(k)
            else:
                lines.append(line)
    for k, v in values.items():
        if v and k not in seen:
            lines.append("%s=%s" % (k, v))
        if v:
            os.environ[k] = v
    fd = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines).strip() + "\n")
    os.chmod(ENV_FILE, 0o600)


def delete_key(env_name):
    """Remove a key from .env and from this process. True if it was there."""
    found = False
    if ENV_FILE.exists():
        kept = []
        for line in ENV_FILE.read_text().splitlines():
            if line.split("=", 1)[0].strip() == env_name:
                found = True
            else:
                kept.append(line)
        fd = os.open(ENV_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("\n".join(kept).strip() + ("\n" if kept else ""))
    found = os.environ.pop(env_name, None) is not None or found
    return found


@dataclass
class Provider:
    id: str
    name: str
    base_url: str
    key_env: str = ""
    docs: str = ""
    headers: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)        # canonical -> provider model id
    thinking_off: dict = field(default_factory=dict)  # model or "*" -> body fragment
    extra_body: dict = field(default_factory=dict)    # model or "*" -> body fragment
    quirks: dict = field(default_factory=dict)
    pricing: list = field(default_factory=list)
    path: str = ""
    custom: bool = False                              # from ~/.providerbench/providers

    @property
    def key(self):
        return os.environ.get(self.key_env, "") if self.key_env else ""

    def has_key(self):
        return not self.key_env or bool(self.key)

    def _base(self, model_id=""):
        base = self.base_url.replace("{model}", model_id)
        for var in re.findall(r"\{([A-Z0-9_]+)\}", base):
            val = os.environ.get(var)
            if not val:
                raise RuntimeError("%s needs the environment variable %s (put it in .env)" % (self.name, var))
            base = base.replace("{%s}" % var, val)
        return base.rstrip("/")

    def host(self):
        return urlsplit(self._base()).hostname

    def url(self, model_id, path="/chat/completions"):
        return self._base(model_id) + path

    def _per_model(self, table, model):
        out = {}
        for pat in sorted(table, key=lambda p: (p != "*", p)):   # "*" first, specific last
            if pat == "*" or fnmatch.fnmatch(model, pat):
                out = deep_merge(out, table[pat])
        return out

    def body_for(self, model, thinking="default"):
        body = self._per_model(self.extra_body, model)
        if thinking == "off":
            body = deep_merge(body, self._per_model(self.thinking_off, model))
        return body

    def info(self):
        return {"id": self.id, "name": self.name, "base_url": self.base_url, "key_env": self.key_env,
                "key_saved": bool(self.key) if self.key_env else True, "models": self.models,
                "docs": self.docs, "quirks": self.quirks, "custom": self.custom}


def deep_merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_providers(include_mock=False):
    """providers/*.toml, then your own in ~/.providerbench/providers; with include_mock (pb serve --mock)
    also the local fake providers."""
    out = {}
    paths = sorted(PROVIDERS_DIR.glob("*.toml"))
    if include_mock or os.environ.get("PB_MOCK") == "1":
        paths += sorted((ROOT / "providerbench" / "mock_providers").glob("*.toml"))
    paths += sorted(USER_PROVIDERS_DIR.glob("*.toml"))
    for path in paths:
        d = tomllib.loads(path.read_text())
        pid = path.stem
        out[pid] = Provider(id=pid, name=d.get("name", pid), base_url=d["base_url"],
                            key_env=d.get("key_env", ""), docs=d.get("docs", ""),
                            headers=d.get("headers", {}), models=d.get("models", {}),
                            thinking_off=d.get("thinking_off", {}), extra_body=d.get("extra_body", {}),
                            quirks=d.get("quirks", {}), pricing=d.get("pricing", []), path=str(path),
                            custom=path.parent == USER_PROVIDERS_DIR)
    return out


def load_models():
    path = ROOT / "models.toml"
    return tomllib.loads(path.read_text()).get("models", {}) if path.exists() else {}


@dataclass
class Target:
    key: str              # "provider:model" as given
    provider: Provider
    model: str            # canonical name, or the raw id when not canonical
    model_id: str         # what is sent as "model"
    canonical: bool

    @property
    def slug(self):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", "%s__%s" % (self.provider.id, self.model))


def resolve(target, providers=None, models=None):
    providers = providers if providers is not None else load_providers()
    models = models if models is not None else load_models()
    if ":" not in target:
        raise ValueError("target %r: use provider:model, e.g. fireworks:glm-5.3-flash" % target)
    pid, model = target.split(":", 1)
    if pid not in providers:
        raise ValueError("unknown provider %r (have: %s)" % (pid, ", ".join(sorted(providers))))
    prov = providers[pid]
    if model in prov.models:
        return Target(target, prov, model, prov.models[model], True)
    if model in models:
        raise ValueError("%s has no id for %s; add it under [models] in %s" % (prov.name, model, prov.path))
    for canon, mid in prov.models.items():   # raw id that happens to be mapped
        if mid == model:
            return Target(target, prov, canon, mid, True)
    return Target(target, prov, model, model, False)


# --------------------------------------------------------------- model-id matching

def norm_id(s):
    """'accounts/fireworks/models/glm-5p3-flash' -> 'glm53flash'."""
    s = s.lower().rsplit("/", 1)[-1]
    s = re.sub(r"(?<=\d)p(?=\d)", ".", s)
    s = re.sub(r"-(it|instruct|chat)$", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def match_canonical(model_id, models=None):
    """Best canonical name for a provider's model id, or None. Exact alias first, then normalised."""
    models = models if models is not None else load_models()
    for canon, m in models.items():
        if model_id == canon or model_id in m.get("aliases", []):
            return canon
    n = norm_id(model_id)
    for canon, m in models.items():
        names = [canon] + m.get("aliases", [])
        if any(norm_id(x) == n for x in names):
            return canon
    for canon, m in models.items():      # date / build suffix: deepseekv41flash1017 vs deepseekv41flash
        for x in [canon] + m.get("aliases", []):
            a = norm_id(x)
            if n.startswith(a) and n[len(a):].isdigit():
                return canon
    return None
