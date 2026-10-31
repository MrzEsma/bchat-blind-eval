#!/usr/bin/env python3
"""Preflight check for the two models of a blind evaluation.

For each configured model (OpenAI-compatible /chat/completions endpoint),
four checks, all must pass:
  1. connectivity   — a tiny completion succeeds with the configured key
  2. usage stats    — the response reports prompt/completion token counts
                      (without them the report cannot compute cost)
  3. function calls — the model emits a real tool_call when a tool is offered
  4. tool roundtrip — after feeding the tool result back, a final answer comes
Usage/latency are measured because the final report needs them for costs.

Config file (see SKILL.md):
{
  "base":     {"label": "...", "base_url": "https://.../v1",
               "model": "...", "api_key_env": "EVAL_KEY_BASE",
               "price_per_1m": {"input": 0.0, "output": 0.0}},
  "candidate": {...}
}
api_key_env names an environment variable holding the key (preferred);
a literal "api_key" field also works for local tests.

Exit code 0 iff both models pass everything.
"""
from __future__ import annotations

import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CALC_TOOL = {
    "type": "function",
    "function": {
        "name": "add_numbers",
        "description": "دو عدد را با هم جمع می‌کند",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "number"},
                "b": {"type": "number"},
            },
            "required": ["a", "b"],
        },
    },
}


def load_key(spec: dict) -> str:
    env = spec.get("api_key_env")
    if env:
        val = os.environ.get(env, "")
        if not val:
            raise SystemExit(f"environment variable {env} is not set")
        return val
    return spec.get("api_key", "")


def chat(spec: dict, key: str, messages: list, tools=None, max_tokens=800,
         timeout=120):
    url = spec["base_url"].rstrip("/") + "/chat/completions"
    body = {"model": spec["model"], "messages": messages,
            "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {key}"})
    # free-tier endpoints often answer 429 for a moment; one blip must not
    # fail the preflight, so retry with backoff (honoring Retry-After)
    for attempt in range(5):
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return data, (time.time() - t0) * 1000.0
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            if e.code in (408, 429, 500, 502, 503, 504) and attempt < 4:
                wait = e.headers.get("Retry-After")
                try:
                    wait = min(float(wait), 60) if wait else 5 * 2 ** attempt
                except ValueError:
                    wait = 5 * 2 ** attempt
                print(f"    (HTTP {e.code}, retrying in {wait:.0f}s)",
                      flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code}: {detail}") from e
        except (OSError, http.client.HTTPException, ValueError) as e:
            # connection refused/reset, timeouts, truncated or non-JSON body
            if attempt == 4:
                raise RuntimeError(f"{type(e).__name__}: {e}") from e
            wait = 5 * 2 ** attempt
            print(f"    ({type(e).__name__}, retrying in {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def check_model(spec: dict) -> dict:
    out = {"label": spec.get("label") or spec["model"], "checks": {},
           "usage_fields": [], "ok": True}
    key = load_key(spec)

    # 1) connectivity + usage
    try:
        data, ms = chat(spec, key, [{"role": "user", "content": "سلام"}],
                        max_tokens=30)
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        out["checks"]["connectivity"] = f"ok ({ms:.0f} ms)"
        out["usage_fields"] = sorted(k for k, v in usage.items()
                                     if isinstance(v, int))
        for need, alt in (("prompt_tokens", "input_tokens"),
                          ("completion_tokens", "output_tokens")):
            if need not in usage and alt not in usage:
                out["checks"]["usage_stats"] = (
                    f"MISSING ({need}) — costs cannot be computed")
                out["ok"] = False
        if "usage_stats" not in out["checks"]:
            out["checks"]["usage_stats"] = "ok"
    except Exception as e:  # noqa: BLE001
        out["checks"]["connectivity"] = f"FAIL: {e}"
        out["ok"] = False
        return out

    # 2) function calling
    try:
        data, ms = chat(
            spec, key,
            [{"role": "user",
              "content": "حاصل ۲+۲ را فقط و فقط با ابزار add_numbers حساب کن؛ "
                         "بدون ابزار پاسخ نده."}],
            tools=[CALC_TOOL], max_tokens=300)
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            out["checks"]["function_calling"] = (
                f"FAIL: no tool_calls (content={str(msg.get('content'))[:80]!r})")
            out["ok"] = False
            return out
        fn = calls[0]["function"]
        args_ok = "a" in json.loads(fn.get("arguments") or "{}")
        out["checks"]["function_calling"] = (
            f"ok ({fn['name']}"
            f"{', args ok' if args_ok else ', args MISSING'}) ({ms:.0f} ms)")
        if not args_ok:
            out["ok"] = False

        # 3) tool roundtrip
        msgs = [{"role": "user", "content": "حاصل ۲+۲ را با ابزار حساب کن."},
                {"role": "assistant", "content": msg.get("content") or "",
                 "tool_calls": calls},
                {"role": "tool", "tool_call_id": calls[0]["id"],
                 "content": "4"}]
        data, ms = chat(spec, key, msgs, tools=[CALC_TOOL], max_tokens=200)
        final = (data["choices"][0]["message"].get("content") or "").strip()
        out["checks"]["tool_roundtrip"] = (
            f"ok ({ms:.0f} ms)" if final else "FAIL: no final content")
        if not final:
            out["ok"] = False
    except Exception as e:  # noqa: BLE001
        out["checks"]["function_calling"] = f"FAIL: {e}"
        out["ok"] = False
    return out


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "..", "models.json")
    cfg = json.load(open(cfg_path))
    results = {side: check_model(cfg[side]) for side in ("base", "candidate")}
    art = os.path.abspath(cfg_path) + ".preflight.json"
    json.dump(results, open(art, "w"), ensure_ascii=False, indent=1)

    for side, r in results.items():
        print(f"--- {side}: {r['label']}")
        for k, v in r["checks"].items():
            print(f"    {k:18} {v}")
        if r["usage_fields"]:
            print(f"    usage fields      {', '.join(r['usage_fields'])}")
        print(f"    overall           {'PASS' if r['ok'] else 'FAIL'}")
    print(f"preflight artifact: {art}")
    return 0 if all(r["ok"] for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
