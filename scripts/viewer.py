#!/usr/bin/env python3
"""Build a self-contained HTML viewer for a finished run.

  python3 viewer.py --run runs/myrun            # -> runs/myrun/viewer.html

The page shows every case: the conversation the models received, then the
two answers side by side with each model's name, its tool calls (with the
frozen results) and the judge's verdict and reason. A "blind" switch on the
page hides the names and shuffles the sides so a human can pick a winner
first and reveal afterwards; human picks are kept in that browser only.

Like report.py this UNBLINDS the run, so it refuses while verdicts are still
pending unless --allow-partial is given. --fragment writes the page without
the <html>/<head> wrapper (for hosts that add their own).
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEMPLATE = os.path.join(HERE, "viewer_template.html")


def lines(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="",
                    help="models.json (default: <run>/run_config.json)")
    ap.add_argument("--out", default="")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--fragment", action="store_true")
    args = ap.parse_args()
    run = args.run

    import sys
    sys.path.insert(0, HERE)
    from run_eval import history_messages, customer_message  # noqa: E402

    cfg_path = args.config or os.path.join(run, "run_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "models.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    names = {s: (cfg.get(s) or {}).get("label") or s
             for s in ("base", "candidate")}
    models = {s: (cfg.get(s) or {}).get("model") or "" for s in names}

    meta_path = os.path.join(run, "run_meta.json")
    if os.path.exists(meta_path):
        import hashlib
        want = json.load(open(meta_path)).get("dataset_sha256_16")
        have = hashlib.sha256(open(os.path.join(
            ROOT, "dataset", "cases.jsonl"), "rb").read()).hexdigest()[:16]
        if want and want != have:
            print("WARNING: dataset/cases.jsonl changed since this run was "
                  "made; conversations shown may differ from what the models "
                  "saw")
    recs = {}
    for r in lines(os.path.join(run, "results.jsonl")):
        if r.get("status") == "completed":
            recs[(r["case_id"], r["variant"])] = r
    votes = {v["case_id"]: v for v in lines(os.path.join(run, "votes.jsonl"))}
    cases = {}
    for l in open(os.path.join(ROOT, "dataset", "cases.jsonl")):
        c = json.loads(l)
        cases[c["id"]] = c
    completed = sorted({cid for cid, v in recs if v == "A"
                        and (cid, "B") in recs and cid in cases})
    unknown = sorted({cid for cid, _ in recs if cid not in cases})
    if unknown:
        print(f"WARNING: {len(unknown)} result ids are not in the dataset "
              f"(e.g. {unknown[0]}); left out")
    pending = [c for c in completed if c not in votes]
    not_run = [c for c in cases if c not in completed]
    if (pending or not_run) and not args.allow_partial:
        raise SystemExit(f"run is not finished ({len(not_run)} cases not "
                         f"run, {len(pending)} not judged); the viewer "
                         "unblinds the run. Finish or pass --allow-partial.")
    blind = json.load(open(os.path.join(run, "blinding.json")))["A"]

    data = []
    for cid in sorted(completed, key=lambda x: list(cases).index(x)):
        c = cases[cid]
        a_owner = blind[cid]
        sides = {}
        for variant in ("A", "B"):
            r = recs.get((cid, variant))
            if not r:
                continue
            owner = a_owner if variant == "A" else (
                "candidate" if a_owner == "base" else "base")
            sides[owner] = {
                "replies": r.get("replies", []),
                "flags": r.get("reply_flags", []),
                "tools": [{"turn": t.get("turn", 0), "tool": t["tool"],
                           "args": t["args"], "result": t["result"]}
                          for t in r.get("tool_calls", [])],
                "latency_ms": r.get("latency_ms", 0),
                "usage": r.get("usage", {}),
            }
        v = votes.get(cid)
        winner = None
        if v:
            if v["verdict"] == "a_better":
                winner = a_owner
            elif v["verdict"] == "b_better":
                winner = "candidate" if a_owner == "base" else "base"
            else:
                winner = v["verdict"]
        data.append({
            "id": cid, "difficulty": c["difficulty"], "cohort": c["cohort"],
            "tags": c["tags"], "city": c.get("customer_city", ""),
            "history": [{"who": "customer" if m["role"] == "user" else "shop",
                         "text": m["content"]} for m in history_messages(c)],
            "turns": [customer_message(t) for t in c.get("turns", [])],
            "kb": c["vendor"]["knowledge_base"],
            "sides": sides, "winner": winner,
            "reason": (v or {}).get("reason", ""),
        })

    payload = json.dumps({"names": names, "models": models,
                          "run": os.path.basename(os.path.normpath(run)),
                          "cases": data}, ensure_ascii=False)
    # no raw "<" inside the <script> data block (</script>, <!--, ...)
    payload = payload.replace("<", "\\u003c")
    html = open(TEMPLATE).read().replace("__DATA__", payload)
    if args.fragment:
        start = html.index("<!--FRAGMENT-START-->")
        end = html.index("<!--FRAGMENT-END-->")
        html = html[start:end]
    out = args.out or os.path.join(run, "viewer.html")
    open(out, "w").write(html)
    print(f"viewer written: {out} ({len(data)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
