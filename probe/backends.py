"""Model backends: Ollama and any OpenAI-compatible server (llama.cpp, vLLM, ...).

Everything in probe/ goes through `generate()` here, so adding a host is a config
entry rather than a code change. Backends are declared in `probe/backends.json`
(see backends.example.json); secrets are never stored there — an `api_key_env`
field names an environment variable, and the value is read at call time.

    from backends import load_backends, generate, list_models
    backends = load_backends()
    text, tokens, stop = generate(backends["local"], "qwen3:8b", "hello")
"""
import json
import os
import re
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "backends.json")
THINK = re.compile(r"<think>.*?</think>", re.S)

DEFAULTS = {
    "local": {
        "kind": "ollama",
        "url": "http://127.0.0.1:11434",
        "note": "this machine's RTX 4060, 8 GB — fits models up to ~8B",
    }
}


def load_backends():
    """Config file if present, otherwise just the local Ollama."""
    if os.path.exists(CONFIG):
        with open(CONFIG, encoding="utf-8") as fh:
            return json.load(fh)
    return dict(DEFAULTS)


def _auth_header(backend):
    env = backend.get("api_key_env")
    key = os.environ.get(env, "") if env else backend.get("api_key", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _post(url, payload, headers, timeout):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.load(fh)


# Qwen3's official non-thinking format: the chat template with an empty reasoning
# block already closed. Ollama 0.9.6 ignores `think: false` for these models (their
# template predates it), so the prompt is sent raw with the block prefilled; this is
# what llama.cpp's `enable_thinking: false` produces for the 27B.
QWEN3_NO_THINK = ("<|im_start|>user\n{prompt}<|im_end|>\n"
                  "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def _is_qwen3(model):
    return model.startswith("qwen3")


def generate(backend, model, prompt, *, max_tokens=1024, temperature=0.0,
             timeout=1200, seed=None, num_ctx=None, no_think=False, meta=None):
    """-> (text with <think> stripped, tokens generated, stop reason).

    `no_think` disables reasoning for models that have a switch; `num_ctx` sets the
    Ollama context window explicitly instead of inheriting the server default. If a
    dict is passed as `meta`, it receives prompt-token count and the raw response.

    Raises the underlying urllib error so callers can record it; probes treat a
    failure as an unparseable answer rather than a wrong one.
    """
    kind = backend.get("kind", "ollama")
    headers = _auth_header(backend)

    if kind == "ollama":
        options = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = seed
        if num_ctx:
            options["num_ctx"] = num_ctx
        payload = {"model": model, "prompt": prompt, "stream": False,
                   "options": options}
        if no_think and _is_qwen3(model):
            payload["prompt"] = QWEN3_NO_THINK.format(prompt=prompt)
            payload["raw"] = True
        data = _post(backend["url"] + "/api/generate", payload, headers, timeout)
        text = data.get("response", "")
        if meta is not None:
            meta.update(prompt_tokens=data.get("prompt_eval_count", 0), raw=text)
        return THINK.sub("", text).strip(), data.get("eval_count", 0), \
            data.get("done_reason", "")

    if kind == "openai":
        # llama.cpp, vLLM and friends. Chat completions is the endpoint every one
        # of them implements; plain /v1/completions is not always present.
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, "max_tokens": max_tokens,
                   "stream": False}
        if seed is not None:
            payload["seed"] = seed
        if no_think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        data = _post(backend["url"].rstrip("/") + "/v1/chat/completions",
                     payload, headers, timeout)
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        if meta is not None:
            meta.update(prompt_tokens=usage.get("prompt_tokens", 0), raw=text)
        return THINK.sub("", text).strip(), usage.get("completion_tokens", 0), \
            choice.get("finish_reason", "")

    raise ValueError(f"unknown backend kind: {kind!r}")


def list_models(backend, timeout=30):
    """Model identifiers the backend will accept."""
    kind = backend.get("kind", "ollama")
    headers = _auth_header(backend)
    if kind == "ollama":
        req = urllib.request.Request(backend["url"] + "/api/tags", headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as fh:
            return [m["name"] for m in json.load(fh).get("models", [])]
    req = urllib.request.Request(backend["url"].rstrip("/") + "/v1/models",
                                 headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return [m["id"] for m in json.load(fh).get("data", [])]


def describe(backend, timeout=30):
    """Whatever the server will tell us about what it has loaded."""
    out = {}
    headers = _auth_header(backend)
    if backend.get("kind") == "openai":
        for path in ("/props", "/v1/models"):
            try:
                req = urllib.request.Request(backend["url"].rstrip("/") + path,
                                             headers=headers)
                with urllib.request.urlopen(req, timeout=timeout) as fh:
                    out[path] = json.load(fh)
            except Exception as exc:                            # noqa: BLE001
                out[path] = f"{type(exc).__name__}: {exc}"
    else:
        try:
            out["/api/tags"] = list_models(backend, timeout)
        except Exception as exc:                                # noqa: BLE001
            out["/api/tags"] = f"{type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    import sys
    backends = load_backends()
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for name, backend in backends.items():
        if which and name != which:
            continue
        print(f"=== {name}  ({backend.get('kind')}, {backend.get('url')})")
        if backend.get("note"):
            print(f"    {backend['note']}")
        if backend.get("api_key_env"):
            have = "set" if os.environ.get(backend["api_key_env"]) else "NOT SET"
            print(f"    api key env {backend['api_key_env']}: {have}")
        try:
            for m in list_models(backend):
                print(f"    - {m}")
        except Exception as exc:                                # noqa: BLE001
            print(f"    unreachable: {type(exc).__name__}: {exc}")
