#!/usr/bin/env python3
"""Blind A/B execution of the 120-case dataset against two models.

Every case is sent to both models with the SAME system prompt, the SAME
frozen vendor world, and the SAME mock tool outputs (fixtures recorded in
the dataset — no real API is ever called). Per case, which model is "A" and
which is "B" is decided by a seeded coin flip and written to blinding.json.
Judge views (judge/<case>.md) contain only A/B labels and both sides'
tool calls + final replies, so a judge that has not opened blinding.json
stays blind.

Usage:
  python3 run_eval.py --config models.json --out runs/myrun --next 10
  python3 run_eval.py ... --next 10        # next 10 not-yet-completed cases
  python3 run_eval.py ... --start 21 --end 30   # dataset positions 21..30
  python3 run_eval.py ... --limit 5        # first 5 cases (smoke run)
  python3 run_eval.py ... --case representative_001
Re-running the same --out directory skips already-completed cases; a case
whose one side finished before a crash/429 reuses that side (only the
missing side is re-run). The A/B coin is per case id and seeded, so it does
not depend on which batch a case runs in.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import random
import secrets
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Reasoning blocks some models put inside the visible content. The closing
# tag is matched loosely (any </...>) because models misspell it (</thoth>).
REASON_TAGS = r"(?:think|thinking|thought|thoughts|reasoning|reflection)"
REASON_BLOCK_RE = re.compile(
    rf"<{REASON_TAGS}\b[^>]*>.*?</\s*[a-z_-]{{2,20}}\s*>", re.S | re.I)
REASON_OPEN_RE = re.compile(rf"<{REASON_TAGS}\b[^>]*>", re.I)
REASON_CLOSE_RE = re.compile(rf"</\s*{REASON_TAGS}[\w-]*\s*>", re.I)
EMPTY_REPLY = "[پاسخ خالی — مدل متنی برای مشتری تولید نکرد]"
NO_FINAL_REPLY = "[دور ابزار تمام شد بدون پاسخ نهایی]"

DEFAULT_RESULTS = {
    "get_product_info": "[Tool Result]: اطلاعاتی برای این محصول پیدا نشد. "
                        "این سؤال را با label مناسب به غرفه‌دار ارجاع بده.",
    "search_products_in_basalam":
        "[Tool Result]: محصولی مطابق این جستجو در غرفه پیدا نشد.",
    "get_similar_products":
        "[Tool Result]: محصول مشابه قابل‌پیشنهادی پیدا نشد.",
    "google_search":
        "[Tool Result]: دامنه این نتیجه: اطلاعات عمومی درباره نوع کالا یا یک "
        "ویژگی کلی است؛ این متن مشخصات محصول خاص غرفه، قیمت، موجودی یا "
        "تنوع‌های قابل خرید آن را اثبات نمی‌کند.\n\n(نتیجه جستجوی نمونه — "
        "منبع مشخص و تأییدشده‌ای برای ادعای دقیق پیدا نشد.)",
    # same text as the dataset's escalate success fixtures
    "escalate_to_vendor":
        "پیگیری داخلی ثبت و اطلاع داده شده است.\n"
        "در پاسخ نهایی، اول‌شخص و طبیعی بگو موضوع را بررسی می‌کنی و همین‌جا "
        "خبر می‌دهی؛ مثلاً: «اجازه بدید بررسی کنم و همین‌جا خبر می‌دم.»\n"
        "سازوکار داخلی پیگیری، هر عنوان انسانی و هر پاسخ‌دهندهٔ دوم را نام "
        "نبر.\n"
        "تا روشن‌شدن موضوع، نتیجهٔ تأییدنشده یا زمان قطعی نساز.\n"
        "اگر مشتری دوباره همین موضوع را پیگیری کرد و همچنان به بررسی نیاز "
        "بود، این Tool را با label برابر direct_contact دوباره اجرا کن، کوتاه "
        "عذرخواهی کن و همان لحن اول‌شخص را نگه دار.\n"
        "ارتباط مستقیم را پیش از درخواست صریح مشتری پیشنهاد نده.",
    "get_customer_order_status":
        '[Tool Result]: {"order_status": "در حال آماده‌سازی", '
        '"shipping_method": "پست پیشتاز", "estimated_delivery": '
        '"حدوداً ۳ روز کاری دیگر"}',
}


def clean_reply(raw):
    """Strip reasoning the model leaked into its visible reply.

    Returns (clean_text, flags). flags lists what happened so the report and
    judge view can show it instead of silently passing an empty answer.
    """
    flags = []
    text = raw or ""
    stripped = REASON_BLOCK_RE.sub("", text)
    if stripped != text:
        flags.append("reasoning_stripped")
    # a bare closing tag: everything before it was reasoning
    closes = list(REASON_CLOSE_RE.finditer(stripped))
    if closes:
        stripped = stripped[closes[-1].end():]
        flags.append("reasoning_stripped")
    # an opening tag that never closed: the rest is reasoning
    op = REASON_OPEN_RE.search(stripped)
    if op:
        stripped = stripped[:op.start()]
        flags.append("unclosed_reasoning")
    clean = stripped.strip()
    if not clean:
        flags.append("empty_reply")
        clean = EMPTY_REPLY
    return clean, sorted(set(flags))


def coin(seed, case_id):
    """Per-case seeded coin: which model is shown as A. Independent of batch
    order and filters, so every case gets a fair flip whenever it runs."""
    return random.Random(f"{seed}:{case_id}").choice(["base", "candidate"])


def load_blinding(path, all_ids, seed, already_run=()):
    """Load (or create) blinding.json and make sure EVERY dataset case has an
    entry. Existing entries are never changed. A case that was already run by
    an older version of this script without an entry was run with A=base (the
    old default), so it gets "base", not a fresh coin."""
    if os.path.exists(path):
        blind = json.load(open(path))
    else:
        # the seed is secret (stored only here): with a fixed public default
        # anyone could recompute the A/B mapping from the code
        blind = {"_note": "DO NOT read this file before judging is complete",
                 "seed": seed if seed is not None else secrets.randbits(64),
                 "A": {}}
    if blind.get("seed") is None:
        # an older file without a stored seed: its entries were made from the
        # public default seed and can be recomputed; new ones get a secret seed
        blind["seed"] = seed if seed is not None else secrets.randbits(64)
        print("WARNING: blinding.json had no stored seed (older version); its "
              "existing A/B entries can be recomputed from public code",
              flush=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(blind, fh, ensure_ascii=False, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    seed = blind["seed"]
    missing = [cid for cid in all_ids if cid not in blind["A"]]
    if missing:
        for cid in missing:
            blind["A"][cid] = ("base" if cid in already_run
                               else coin(seed, cid))
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(blind, fh, ensure_ascii=False, indent=1)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    return blind


def load_cases(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def render_system_prompt(case, template, schemas):
    world = case["vendor"]["knowledge_base"]
    shop = re.search(r"نام غرفه\*?\*?:?\*?\*?\s*(.+)", world)
    cat = re.search(r"محصولات تخصصی\*?\*?:?\*?\*?\s*(.+)", world)
    return (template
            .replace("{{SHOP_NAME}}", shop.group(1).strip() if shop else "غرفه")
            .replace("{{SHOP_CATEGORY}}",
                     cat.group(1).strip()[:80] if cat else "محصولات غرفه")
            .replace("{{VENDOR_BLOCK}}", world)
            .replace("{{CUSTOMER_CITY}}", case.get("customer_city", "تهران")))


ESCALATE_LABELS = json.load(open(os.path.join(ROOT, "rig", "tool_schemas.json")))[
    "escalate_to_vendor"]["parameters"]["properties"]["label"].get("enum", [])


def _norm(v):
    """Compare tool args loosely on type/spacing (900000001 == "900000001")."""
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items() if x not in (None, "")}
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    s = str(v).replace("\u200c", " ").strip().lower()
    return re.sub(r"\s+", " ", s)


def _stem(w):
    for suf in ("‌های", "‌ها", "های", "ها", "ی"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[:-len(suf)]
    return w


def _words(text):
    return {_stem(w) for w in re.findall(r"[\w\u0600-\u06FF]+",
                                         _norm(text))}


def mock_tool(case, name, args):
    """Deterministic frozen tool output.

    exact fixture -> normalized subset match -> tool-specific identity match
    (same product_id; near-identical search query) -> fixed fail-closed
    default. A fixture is never handed out for unrelated arguments: a model
    that searches for the wrong thing gets "not found", not the right answer.
    """
    fx = [f for f in case.get("tool_fixtures", []) if f["tool"] == name]
    if not isinstance(args, dict):
        args = {"_raw": args}
    na = _norm(args)
    for f in fx:
        if _norm(f.get("args") or {}) == na:
            return f["result"]
    for f in fx:
        fa = _norm(f.get("args") or {})
        if na and fa and all(na.get(k) == v for k, v in fa.items()):
            return f["result"]
    if name in ("get_product_info", "get_similar_products"):
        for f in fx:
            if _norm((f.get("args") or {}).get("product_id")) == \
                    na.get("product_id") and na.get("product_id"):
                return f["result"]
    elif name in ("search_products_in_basalam", "google_search"):
        q = _words(args.get("query", ""))
        best, key = None, (0.0, 0, 0)
        for f in fx:
            fq = _words((f.get("args") or {}).get("query", ""))
            if q and fq:
                j = len(q & fq) / len(q | fq)
                if q <= fq or fq <= q:  # one query refines the other
                    j = max(j, 0.6 if len(q & fq) >= 2 else j)
                # ties: more shared words first, then a result that found
                # something over a "not found" fixture
                found = 0 if "پیدا نشد" in f["result"][:200] else 1
                k = (j, len(q & fq), found)
                if k > key:
                    best, key = f, k
        score = key[0]
        if best is not None and score >= 0.6:
            return best["result"]
    elif name == "escalate_to_vendor":
        if not (args.get("label") and args.get("reason")) or \
                args.get("label") not in ESCALATE_LABELS:
            return ("[Tool Result]: validation error for escalate_to_vendor: "
                    "`reason` and a valid `label` (one of "
                    f"{', '.join(ESCALATE_LABELS)}) are required; nothing was "
                    "registered.")
        # the ack does not depend on wording, only on a valid label
        valid = [f for f in fx if (f.get("args") or {}).get("label")
                 and "validation error" not in f["result"]]
        if args.get("label") and args.get("reason"):
            same = [f for f in valid
                    if f["args"].get("label") == args.get("label")]
            if same or valid:
                return (same or valid)[0]["result"]
    elif name == "get_customer_order_status" and fx:
        return fx[0]["result"]
    return DEFAULT_RESULTS.get(
        name, "[Tool Result]: ابزار خروجی معتبری برنگرداند.")


def customer_message(m):
    parts = []
    body = m.get("content") or ""
    has_link = "basalam.com/p/" in body
    if (m.get("message_type") == "PRODUCT" or m.get("product_id")) and not has_link:
        parts.append(f"لینک محصول: basalam.com/p/{m.get('product_id', '')}")
    elif m.get("message_type") == "PICTURE" and not body.strip():
        parts.append("(مشتری یک تصویر ارسال کرده است)")
    q = m.get("quoted_message")
    if q:
        user_part = body.strip() or "(متنی ندارد؛ فقط پیام بالا را نقل کرده است)"
        body = (f"پیام کاربر در پاسخ به پیام نقل‌قول‌شده‌ی زیر است. "
                f"نقش پیام نقل‌قول‌شده: {q.get('role', 'assistant')}\n\n"
                f"> {q.get('content', '')}\n\nپیام کاربر:\n{user_part}")
    parts.append(body.strip())
    return "\n".join(p for p in parts if p.strip())


def seller_message(m):
    """A prior message from the shop side (the bot or the human vendor).

    Product cards have no text in the dataset; render the card as its link so
    the model (and the judge) sees what was actually sent instead of "".
    """
    body = (m.get("content") or "").strip()
    if (m.get("message_type") == "PRODUCT" or m.get("product_id")) \
            and "basalam.com/p/" not in body:
        card = f"(کارت محصول ارسال شد: basalam.com/p/{m.get('product_id', '')})"
        body = f"{card}\n{body}".strip()
    elif m.get("message_type") == "PICTURE" and not body:
        body = "(یک تصویر ارسال شد)"
    if m.get("role") == "vendor":
        body = f"(پیام خود غرفه‌دار)\n{body}"
    return body


def history_messages(case):
    """The prior conversation exactly as both models receive it."""
    out = []
    for m in case.get("prior_messages", []):
        if m.get("role") in ("assistant", "vendor"):
            out.append({"role": "assistant", "content": seller_message(m)})
        else:
            out.append({"role": "user", "content": customer_message(m)})
    return out


class Client:
    def __init__(self, spec, max_retries=6, max_wait=120):
        self.spec = spec
        self.tag = "model"  # what log lines call this client: "side A"/"side B"
        self.max_retries = max_retries
        self.max_wait = max_wait
        env = spec.get("api_key_env")
        self.key = (os.environ.get(env, "") if env
                    else spec.get("api_key", ""))
        if not self.key:
            raise SystemExit(f"missing API key for {spec.get('label')}")

    def _wait(self, attempt, retry_after=None):
        if retry_after:
            try:
                return min(float(retry_after), self.max_wait)
            except ValueError:
                pass
        return min(self.max_wait, 5 * 2 ** attempt) * random.uniform(0.8, 1.2)

    def chat(self, messages, tools, max_tokens=2000, timeout=180):
        """POST /chat/completions with backoff on 429/5xx/network errors.

        Honors Retry-After; waits 5s, 10s, 20s, ... (capped at max_wait) for
        up to max_retries retries before giving up on this request.
        """
        url = self.spec["base_url"].rstrip("/") + "/chat/completions"
        body = {"model": self.spec["model"], "messages": messages,
                "max_tokens": max_tokens, "tools": tools}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        for attempt in range(self.max_retries + 1):
            t0 = time.time()
            last_try = attempt == self.max_retries
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read())
                if "choices" not in data and data.get("error"):
                    # some gateways return 200 with an error body
                    raise RuntimeError(f"API error: {str(data['error'])[:300]}")
                return data, (time.time() - t0) * 1000
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                if e.code in (408, 429, 500, 502, 503, 504) and not last_try:
                    w = self._wait(attempt, e.headers.get("Retry-After"))
                    log(f"    HTTP {e.code} on {self.tag}; "
                        f"retry {attempt + 1}/{self.max_retries} in {w:.0f}s")
                    time.sleep(w)
                    continue
                raise RuntimeError(f"HTTP {e.code}: {detail}") from e
            except (OSError, http.client.HTTPException, ValueError,
                    RuntimeError) as e:  # URLError, socket.timeout, ...
                if last_try:
                    raise
                w = self._wait(attempt)
                log(f"    {type(e).__name__} on {self.tag}; retry "
                    f"{attempt + 1}/{self.max_retries} in {w:.0f}s")
                time.sleep(w)
        raise RuntimeError("unreachable")


def log(msg):
    print(msg, flush=True)


def scrub(text, cfg):
    """Remove model/provider identity from an error so it cannot unblind a
    side. Full ids and labels go first (longest first), then their parts,
    the base_url host, well-known model family names, and provider fields of
    gateway error bodies. Both sides are scrubbed the same way."""
    text = re.sub(r'"(provider_name|provider|model|model_id)"\s*:\s*"[^"]*"',
                  r'"\1": "<hidden>"', text, flags=re.I)
    needles = set()
    for side in ("base", "candidate"):
        spec = cfg.get(side) or {}
        model, label = str(spec.get("model", "")), str(spec.get("label", ""))
        needles |= {model, label, model.split("/")[-1], model.split(":")[0],
                    model.split("/")[-1].split(":")[0]}
        needles.add(re.sub(r"^https?://", "", str(spec.get("base_url", "")))
                    .split("/")[0])
        needles |= set(re.split(r"[/:\s_.-]+", model + " " + label))
    needles |= {"gpt", "claude", "qwen", "llama", "gemini", "gemma",
                "mistral", "deepseek", "nemotron", "glm", "kimi", "minimax",
                "grok", "phi", "anthropic", "openai", "google", "nvidia",
                "meta", "moonshot", "z-ai", "x-ai"}
    for n in sorted((n for n in needles if len(n) >= 3 and n.lower() not in
                     ("free", "chat", "latest", "instruct", "preview",
                      "api", "www", "the")), key=len, reverse=True):
        text = re.sub(re.escape(n), "<hidden>", text, flags=re.I)
    return text


def write_json_atomic(path, obj, mode=None):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, ensure_ascii=False)
    if mode:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def dataset_hash(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def add_usage(total, usage):
    if not usage:
        return
    for k in ("prompt_tokens", "input_tokens"):
        if isinstance(usage.get(k), int):
            total["input"] += usage[k]
    for k in ("completion_tokens", "output_tokens"):
        if isinstance(usage.get(k), int):
            total["output"] += usage[k]
    for k in ("total_tokens",):
        if isinstance(usage.get(k), int):
            total["total"] += usage[k]
    for k in ("prompt_tokens_details", "completion_tokens_details"):
        d = usage.get(k) or {}
        if isinstance(d.get("cached_tokens"), int):
            total["cached"] += d["cached_tokens"]
        if isinstance(d.get("reasoning_tokens"), int):
            total["reasoning"] += d["reasoning_tokens"]


def run_case_model(client, case, template, tools, max_rounds=10):
    """Run all turns of a case for one model. Returns a record dict."""
    system = render_system_prompt(case, template, tools)
    usage = {"input": 0, "output": 0, "total": 0, "cached": 0, "reasoning": 0}
    tool_log, replies, raw_replies, flags, lat = [], [], [], [], 0.0
    messages = [{"role": "system", "content": system}] + history_messages(case)

    for ti, turn in enumerate(case.get("turns", [])):
        messages.append({"role": "user", "content": customer_message(turn)})
        reply = None
        for _ in range(max_rounds):
            data, ms = client.chat(messages, tools)
            lat += ms
            add_usage(usage, data.get("usage"))
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                reply = msg.get("content") or ""
                break
            messages.append({"role": "assistant", "content": msg.get("content"),
                             "tool_calls": calls})
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {"_raw": fn.get("arguments")}
                res = mock_tool(case, fn.get("name", ""), args)
                tool_log.append({"turn": ti, "tool": fn.get("name"),
                                 "args": args, "result": res})
                messages.append({"role": "tool", "tool_call_id": c.get("id"),
                                 "name": fn.get("name"),
                                 "content": res})
        if reply is None:
            raw, clean, f = "", NO_FINAL_REPLY, ["no_final_reply"]
        else:
            raw = reply
            clean, f = clean_reply(reply)
        raw_replies.append(raw)
        replies.append(clean)
        flags.append(f)
        messages.append({"role": "assistant",
                         "content": "" if clean in (EMPTY_REPLY, NO_FINAL_REPLY)
                         else clean})
    return {"status": "completed", "replies": replies,
            "raw_replies": raw_replies, "reply_flags": flags,
            "tool_calls": tool_log, "usage": usage, "latency_ms": round(lat)}


def judge_view(case, a, b):
    """Everything a blind judge needs, in one file, with only A/B labels.

    Shows the conversation exactly as the models received it, every tool
    call with its FULL frozen result, the replies, any reply problems, and
    the vendor knowledge base (needed to check policy answers).
    """
    lines = [f"# {case['id']}  (difficulty {case['difficulty']}, "
             f"cohort {case['cohort']}, tags {', '.join(case['tags'])})", "",
             f"شهر مشتری: {case.get('customer_city', '')}", "",
             "## PRIOR CONVERSATION (as the models received it)"]
    for m in history_messages(case):
        who = "customer" if m["role"] == "user" else "shop"
        lines.append(f"  {who}: {m['content']}")
    lines.append("## NEW CUSTOMER MESSAGE(S)")
    for i, t in enumerate(case.get("turns", [])):
        lines.append(f"  customer[turn {i}]: {customer_message(t)}")
    for side, rec in (("A", a), ("B", b)):
        lines += ["", f"=== SIDE {side} ==="]
        for i, tc in enumerate(rec["tool_calls"]):
            lines.append(f"{side}[tool {i + 1}, turn {tc.get('turn', 0)}] "
                         f"{tc['tool']}"
                         f"({json.dumps(tc['args'], ensure_ascii=False)}) ->")
            lines.append("    " + tc["result"].replace("\n", "\n    "))
        flags = rec.get("reply_flags") or [[] for _ in rec["replies"]]
        for j, r in enumerate(rec["replies"]):
            note = (f"  ⚠ {', '.join(flags[j])}" if j < len(flags) and flags[j]
                    else "")
            lines.append(f"{side}[reply {j}]{note} {r}")
    lines += ["",
              "<!-- رأی را با vote.py ثبت کن؛ این فایل را ویرایش نکن -->",
              "---", "## VENDOR KNOWLEDGE BASE (reference for policy checks)",
              case["vendor"]["knowledge_base"]]
    return "\n".join(lines)


def select_cases(all_cases, args, done):
    if args.case:
        sel = [c for c in all_cases if c["id"] == args.case]
        if not sel:
            raise SystemExit(f"unknown case id: {args.case}")
        return sel
    sel = all_cases
    if args.start or args.end:
        start = max(args.start or 1, 1)
        end = args.end or len(all_cases)
        sel = all_cases[start - 1:end]
    elif args.limit:
        sel = all_cases[:args.limit]
    if args.next:
        sel = [c for c in sel if c["id"] not in done][:args.next]
    return sel


def public_config(cfg):
    """Model config without secrets, stored in the run dir for report.py."""
    out = {}
    for side in ("base", "candidate"):
        spec = dict(cfg[side])
        spec.pop("api_key", None)
        out[side] = spec
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "models.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--next", type=int, default=0,
                    help="run the next N cases that are not completed yet")
    ap.add_argument("--start", type=int, default=0,
                    help="first dataset position to run (1-based)")
    ap.add_argument("--end", type=int, default=0,
                    help="last dataset position to run (inclusive)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N dataset cases")
    ap.add_argument("--case", default="")
    ap.add_argument("--seed", type=int, default=None,
                    help="only to reproduce a run; by default a secret random "
                         "seed is created and kept in blinding.json")
    ap.add_argument("--max-retries", type=int, default=6,
                    help="retries per API request on 429/5xx/network errors")
    ap.add_argument("--max-wait", type=int, default=120,
                    help="longest single backoff sleep in seconds")
    ap.add_argument("--allow-changes", action="store_true",
                    help="continue a run although models/dataset changed")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    all_cases = load_cases(os.path.join(ROOT, "dataset", "cases.jsonl"))
    template = open(os.path.join(ROOT, "rig", "prompt_template.md")).read()
    schemas = json.load(open(os.path.join(ROOT, "rig", "tool_schemas.json")))
    tools = [{"type": "function",
              "function": {"name": n, **s}} for n, s in schemas.items()]

    out = args.out
    data_path = os.path.join(ROOT, "dataset", "cases.jsonl")
    os.makedirs(os.path.join(out, "judge"), exist_ok=True)
    private = os.path.join(out, ".partial")
    os.makedirs(private, exist_ok=True)
    os.chmod(private, 0o700)

    # a run must not silently mix models or dataset versions
    meta_path = os.path.join(out, "run_meta.json")
    meta = {"base_model": cfg["base"]["model"],
            "candidate_model": cfg["candidate"]["model"],
            "base_url": cfg["base"]["base_url"],
            "candidate_url": cfg["candidate"]["base_url"],
            "dataset_sha256_16": dataset_hash(data_path),
            "prompt_sha256_16": dataset_hash(
                os.path.join(ROOT, "rig", "prompt_template.md")),
            "tools_sha256_16": dataset_hash(
                os.path.join(ROOT, "rig", "tool_schemas.json"))}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        diff = [k for k in meta if old.get(k) != meta[k]]
        if diff and not args.allow_changes:
            raise SystemExit(
                f"this run was started with a different {', '.join(diff)} "
                f"({old}); use a new --out, or --allow-changes to mix")
    elif os.path.exists(os.path.join(out, "results.jsonl")) and \
            not args.allow_changes:
        raise SystemExit(
            "this run dir has results but no run_meta.json (made by an older "
            "version), so it cannot prove the same models/dataset were used; "
            "start a new --out, or pass --allow-changes to continue anyway")
    else:
        write_json_atomic(meta_path, meta)
    if not os.path.exists(os.path.join(out, "run_config.json")):
        write_json_atomic(os.path.join(out, "run_config.json"),
                          public_config(cfg))

    results_path = os.path.join(out, "results.jsonl")
    rows = {}
    if os.path.exists(results_path):
        for l in open(results_path):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "completed":
                rows[(r["case_id"], r["variant"])] = r
    # done = both sides recorded (the rows are written last, after the judge
    # file, so a crash in between just re-runs nothing and rebuilds the view)
    done = {cid for cid, v in rows if v == "A" and (cid, "B") in rows}

    blind = load_blinding(os.path.join(out, "blinding.json"),
                          [c["id"] for c in all_cases], args.seed,
                          already_run={cid for cid, _ in rows})
    by_id = {c["id"]: c for c in all_cases}
    for cid in sorted(done):
        jpath = os.path.join(out, "judge", f"{cid}.md")
        if cid in by_id and not os.path.exists(jpath):
            open(jpath, "w").write(judge_view(by_id[cid], rows[(cid, "A")],
                                              rows[(cid, "B")]))
            log(f"{cid}: judge view rebuilt")
    cases = select_cases(all_cases, args, done)

    clients = {s: Client(cfg[s], args.max_retries, args.max_wait)
               for s in ("base", "candidate")}
    f = open(results_path, "a")
    todo = [c for c in cases if c["id"] not in done]
    log(f"{len(done)}/{len(all_cases)} cases already completed; "
        f"running {len(todo)} now")

    ok = 0
    for n, case in enumerate(todo, 1):
        cid = case["id"]
        a_side = blind["A"][cid]
        b_side = "candidate" if a_side == "base" else "base"
        recs, failed = {}, None
        for which, label in ((a_side, "A"), (b_side, "B")):
            # a side that already finished (before a crash) is reused, so a
            # retry only re-runs and re-pays for the side that failed. Files
            # are named by A/B, never by model, so they cannot unblind.
            ppath = os.path.join(private, f"{cid}.{label}.json")
            if os.path.exists(ppath):
                try:
                    recs[which] = json.load(open(ppath))
                    log(f"[{n}/{len(todo)}] {cid} side {label}: reused saved run")
                    continue
                except json.JSONDecodeError:
                    os.remove(ppath)
            clients[which].tag = f"{cid} side {label}"
            t0 = time.time()
            try:
                recs[which] = run_case_model(clients[which], case, template,
                                             tools)
            except Exception as e:  # noqa: BLE001
                failed = f"side {label}: {scrub(str(e), cfg)[:300]}"
                break
            write_json_atomic(ppath, recs[which], 0o600)
            log(f"[{n}/{len(todo)}] {cid} side {label}: done in "
                f"{time.time() - t0:.0f}s")
        if failed:
            log(f"[{n}/{len(todo)}] {cid}: FAILED ({failed}) — re-run later "
                f"with --case {cid}; finished sides are kept")
            f.write(json.dumps({"case_id": cid, "variant": None,
                                "status": "failed", "error": failed},
                               ensure_ascii=False) + "\n")
            f.flush()
            continue
        jpath = os.path.join(out, "judge", f"{cid}.md")
        open(jpath, "w").write(judge_view(case, recs[a_side], recs[b_side]))
        for side in ("A", "B"):
            which = a_side if side == "A" else b_side
            rec = dict(recs[which])
            rec.update({"case_id": cid, "variant": side})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        for label in ("A", "B"):
            try:
                os.remove(os.path.join(private, f"{cid}.{label}.json"))
            except FileNotFoundError:
                pass
        ok += 1
        empties = sum("empty_reply" in fl or "no_final_reply" in fl
                      for side in (a_side, b_side)
                      for fl in recs[side]["reply_flags"])
        na, nb = len(recs[a_side]["tool_calls"]), len(recs[b_side]["tool_calls"])
        log(f"[{n}/{len(todo)}] {cid}: ok (A tools={na}, B tools={nb}"
            + (f", EMPTY replies={empties}" if empties else "") + ")")
    f.close()
    left = len(all_cases) - len(done) - ok
    log(f"run dir: {out} — cases still to run: {left}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
