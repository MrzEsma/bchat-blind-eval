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
  python3 run_eval.py --config models.json --out runs/myrun
  python3 run_eval.py ... --limit 5        # first 5 cases (smoke run)
  python3 run_eval.py ... --case representative_001
Re-running the same --out directory skips already-completed cases.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

THOUGHT_RE = re.compile(r"<thought>.*?</thought>", re.S | re.I)
THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)

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
    "escalate_to_vendor":
        "[Tool Result]: پیگیری داخلی ثبت و اطلاع داده شد.\n"
        "در پاسخ نهایی، اول‌شخص و طبیعی بگو موضوع را بررسی می‌کنی و همین‌جا "
        "خبر می‌دهی؛ مثلاً: «اجازه بدید بررسی کنم و همین‌جا خبر می‌دم.»\n"
        "سازوکار داخلی پیگیری، هر عنوان انسانی و هر پاسخ‌دهندهٔ دوم را نام "
        "نبر. تا روشن‌شدن موضوع، نتیجهٔ تأییدنشده یا زمان قطعی نساز.",
    "get_customer_order_status":
        '[Tool Result]: {"order_status": "در حال آماده‌سازی", '
        '"shipping_method": "پست پیشتاز", "estimated_delivery": '
        '"حدوداً ۳ روز کاری دیگر"}',
}


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


def mock_tool(case, name, args):
    """Deterministic frozen tool output: exact fixture -> subset -> default."""
    fx = [f for f in case.get("tool_fixtures", []) if f["tool"] == name]
    try:
        args = json.loads(json.dumps(args))
    except (TypeError, ValueError):
        pass
    for f in fx:
        if f.get("args") == args:
            return f["result"]
    for f in fx:
        fa = f.get("args") or {}
        if isinstance(args, dict) and args and all(
                args.get(k) == v for k, v in fa.items() if v not in (None, "")):
            return f["result"]
    if len(fx) == 1:
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
        body = (f"پیام کاربر در پاسخ به پیام نقل‌قول‌شده‌ی زیر است. "
                f"نقش پیام نقل‌قول‌شده: {q.get('role', 'assistant')}\n\n"
                f"> {q.get('content', '')}\n\nپیام کاربر:\n{body}")
    parts.append(body.strip())
    return "\n".join(p for p in parts if p.strip())


class Client:
    def __init__(self, spec):
        self.spec = spec
        env = spec.get("api_key_env")
        self.key = (os.environ.get(env, "") if env
                    else spec.get("api_key", ""))
        if not self.key:
            raise SystemExit(f"missing API key for {spec.get('label')}")

    def chat(self, messages, tools, max_tokens=2000, timeout=180):
        url = self.spec["base_url"].rstrip("/") + "/chat/completions"
        body = {"model": self.spec["model"], "messages": messages,
                "max_tokens": max_tokens, "tools": tools}
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"})
        last = None
        for attempt in range(3):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    return json.loads(resp.read()), (time.time() - t0) * 1000
            except urllib.error.HTTPError as e:
                detail = e.read()[:300].decode(errors="replace")
                last = RuntimeError(f"HTTP {e.code}: {detail}")
                if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise last
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt < 2:
                    time.sleep(3)
                    continue
                raise
        raise last


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
    tool_log, replies, lat = [], [], 0.0
    messages = [{"role": "system", "content": system}]
    for m in case.get("prior_messages", []):
        if m.get("role") == "assistant":
            messages.append({"role": "assistant",
                             "content": m.get("content", "")})
        else:
            messages.append({"role": "user",
                             "content": customer_message(m)})

    for turn in case.get("turns", []):
        messages.append({"role": "user", "content": customer_message(turn)})
        reply = ""
        for _ in range(max_rounds):
            data, ms = client.chat(messages, tools)
            lat += ms
            add_usage(usage, data.get("usage"))
            msg = data["choices"][0]["message"]
            calls = msg.get("tool_calls") or []
            if not calls:
                reply = (msg.get("content") or "").strip()
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
                tool_log.append({"tool": fn.get("name"), "args": args,
                                 "result": res[:1500]})
                messages.append({"role": "tool", "tool_call_id": c.get("id"),
                                 "name": fn.get("name"),
                                 "content": res})
        else:
            reply = "[دور ابزار تمام شد بدون پاسخ نهایی]"
        clean = THINK_RE.sub("", THOUGHT_RE.sub("", reply)).strip()
        replies.append(clean)
        messages.append({"role": "assistant", "content": clean})
    return {"status": "completed", "replies": replies, "tool_calls": tool_log,
            "usage": usage, "latency_ms": round(lat)}


def judge_view(case, a, b):
    lines = [f"# {case['id']}  (difficulty {case['difficulty']}, "
             f"cohort {case['cohort']}, tags {', '.join(case['tags'])})", "",
             f"شهر مشتری: {case.get('customer_city','')}", "",
             "## PRIOR CONVERSATION"]
    for m in case.get("prior_messages", []):
        who = "customer" if m.get("role") != "assistant" else "assistant"
        lines.append(f"  {who}: {(m.get('content') or '')[:600]}")
    lines.append("## NEW CUSTOMER MESSAGE(S)")
    for t in case.get("turns", []):
        lines.append(f"  customer: {customer_message(t)}")
    for side, rec in (("A", a), ("B", b)):
        lines += ["", f"=== SIDE {side} ==="]
        for i, tc in enumerate(rec["tool_calls"]):
            lines.append(f"{side}[tool {i + 1}] {tc['tool']}"
                         f"({json.dumps(tc['args'], ensure_ascii=False)}) -> "
                         f"{tc['result'][:400]}")
        for j, r in enumerate(rec["replies"]):
            lines.append(f"{side}[reply {j}] {r}")
    lines += ["", "## VERDICT: a_better | b_better | both_good | both_bad",
              "## REASON (short Persian):"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "models.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--case", default="")
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    cases = load_cases(os.path.join(ROOT, "dataset", "cases.jsonl"))
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
    elif args.limit:
        cases = cases[:args.limit]
    template = open(os.path.join(ROOT, "rig", "prompt_template.md")).read()
    schemas = json.load(open(os.path.join(ROOT, "rig", "tool_schemas.json")))
    tools = [{"type": "function",
              "function": {"name": n, **s}} for n, s in schemas.items()]

    out = args.out
    os.makedirs(os.path.join(out, "judge"), exist_ok=True)
    results_path = os.path.join(out, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        for l in open(results_path):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "completed":
                done.add(r["case_id"])

    blind_path = os.path.join(out, "blinding.json")
    blind = {}
    if os.path.exists(blind_path):
        blind = json.load(open(blind_path))
    else:
        rng = random.Random(args.seed)
        # per case, a fair coin decides which model is shown as "A"
        blind = {"_note": "DO NOT read this file before judging is complete",
                 "A": {c["id"]: rng.choice(["base", "candidate"])
                       for c in cases}}
        json.dump(blind, open(blind_path, "w"), ensure_ascii=False, indent=1)
        os.chmod(blind_path, 0o600)

    clients = {s: Client(cfg[s]) for s in ("base", "candidate")}
    f = open(results_path, "a")

    for case in cases:
        cid = case["id"]
        if cid in done:
            print(f"{cid}: already completed, skipping")
            continue
        a_side = blind["A"].get(cid, "base")
        b_side = "candidate" if a_side == "base" else "base"
        recs, failed = {}, None
        try:
            recs[a_side] = run_case_model(clients[a_side], case, template,
                                          tools)
            recs[b_side] = run_case_model(clients[b_side], case, template,
                                          tools)
        except Exception as e:  # noqa: BLE001
            failed = str(e)[:300]
        if failed:
            print(f"{cid}: FAILED ({failed})")
            f.write(json.dumps({"case_id": cid, "variant": "A",
                                "status": "failed", "error": failed},
                               ensure_ascii=False) + "\n")
            f.flush()
            continue
        for side in ("A", "B"):
            which = a_side if side == "A" else b_side
            rec = dict(recs[which])
            rec.update({"case_id": cid, "variant": side})
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        f.flush()
        jpath = os.path.join(out, "judge", f"{cid}.md")
        open(jpath, "w").write(judge_view(case, recs[a_side], recs[b_side]))
        na, nb = len(recs[a_side]["tool_calls"]), len(recs[b_side]["tool_calls"])
        print(f"{cid}: ok (A tools={na}, B tools={nb})")
    f.close()
    print(f"run dir: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
