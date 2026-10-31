#!/usr/bin/env python3
"""Unblind a finished run and write the final evaluation report.

  python3 report.py --run runs/myrun [--config models.json]

Reads results.jsonl, votes.jsonl and (only now) blinding.json, then writes
runs/<run>/report.md with:
  - overall winner and vote counts (wins / both_good / both_bad)
  - per-topic (tag) win rates -> each model's strengths and weaknesses
  - per-difficulty and per-cohort breakdown
  - token usage and cost per model (needs prices in models.json)
  - latency, tool-call and escalation stats, empty replies, failures
Model labels/prices come from runs/<run>/run_config.json (written by
run_eval.py, no keys) unless --config is given.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "—"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="",
                    help="models.json (default: <run>/run_config.json)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="unblind even though some cases have no verdict")
    args = ap.parse_args()
    run = args.run

    cfg_path = args.config or os.path.join(run, "run_config.json")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(ROOT, "models.json")
    if not os.path.exists(cfg_path):
        raise SystemExit("no model config found: pass --config models.json")
    cfg = json.load(open(cfg_path))

    def lines(p):
        if not os.path.exists(p):
            return []
        out = []
        for l in open(p):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
        return out

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
    results = lines(os.path.join(run, "results.jsonl"))
    votes = {v["case_id"]: v for v in lines(os.path.join(run, "votes.jsonl"))}
    blind = json.load(open(os.path.join(run, "blinding.json")))["A"]

    cases = {}
    for l in open(os.path.join(ROOT, "dataset", "cases.jsonl")):
        c = json.loads(l)
        cases[c["id"]] = c

    recs = {}
    for r in results:
        if r.get("status") == "completed":
            recs[(r["case_id"], r["variant"])] = r
    completed = sorted({cid for cid, v in recs if v == "A"
                        and (cid, "B") in recs and cid in cases})
    unknown = sorted({cid for cid, _ in recs if cid not in cases})
    if unknown:
        print(f"WARNING: {len(unknown)} result ids are not in the dataset "
              f"(e.g. {unknown[0]}); ignored")
    judged = [c for c in completed if c in votes]
    unjudged = [c for c in completed if c not in votes]

    not_run = [c for c in cases if c not in completed]
    if (unjudged or not_run) and not args.allow_partial:
        raise SystemExit(
            f"run is not finished: {len(not_run)} of {len(cases)} cases not "
            f"run yet, {len(unjudged)} run but not judged. The report "
            "unblinds the run, so it waits until every case is run and "
            "judged (or pass --allow-partial).")

    name = {"base": cfg.get("base", {}).get("label", "base"),
            "candidate": cfg.get("candidate", {}).get("label", "candidate")}
    price = {s: (cfg.get(s, {}) or {}).get("price_per_1m") or {}
             for s in ("base", "candidate")}

    # ---- votes ----
    wins = defaultdict(int)
    both = defaultdict(int)
    tag_wins = defaultdict(lambda: defaultdict(int))
    tag_decided = defaultdict(int)
    diff_stats = defaultdict(lambda: defaultdict(int))
    for cid in judged:
        v = votes[cid]["verdict"]
        a_owner = blind[cid]
        b_owner = "candidate" if a_owner == "base" else "base"
        if v == "a_better":
            wins[a_owner] += 1
        elif v == "b_better":
            wins[b_owner] += 1
        else:
            both[v] += 1
        for tag in cases[cid]["tags"]:
            if v in ("both_good", "both_bad"):
                tag_decided[tag] += 0
            else:
                tag_decided[tag] += 1
                winner = a_owner if v == "a_better" else b_owner
                tag_wins[tag][winner] += 1
        winner = (a_owner if v == "a_better" else
                  b_owner if v == "b_better" else v)
        diff_stats[f"difficulty {cases[cid]['difficulty']}"][winner] += 1
        diff_stats[f"cohort {cases[cid]['cohort']}"][winner] += 1

    # ---- usage / cost / latency / tools ----
    stats = {}
    for side in ("base", "candidate"):
        rs = [recs[(cid, "A")] for cid in completed if blind.get(cid) == side]
        rs += [recs[(cid, "B")] for cid in completed
               if blind.get(cid) != side and (cid, "B") in recs]
        usage = defaultdict(int)
        lat = []
        ntools = 0
        nesc = 0
        nempty = 0
        for r in rs:
            for fl in r.get("reply_flags") or []:
                if "empty_reply" in fl or "no_final_reply" in fl:
                    nempty += 1
            for k, val in (r.get("usage") or {}).items():
                usage[k] += val
            lat.append(r.get("latency_ms") or 0)
            for tc in r.get("tool_calls") or []:
                ntools += 1
                if tc.get("tool") == "escalate_to_vendor":
                    nesc += 1
        lat.sort()
        p50 = lat[len(lat) // 2] if lat else 0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0
        cost = None
        if price[side].get("input") is not None:
            cost = (usage["input"] / 1e6) * float(price[side]["input"]) + \
                   (usage["output"] / 1e6) * float(price[side]["output"])
        stats[side] = {"n": len(rs), "usage": dict(usage), "p50": p50,
                       "p95": p95, "tools": ntools, "esc": nesc, "cost": cost,
                       "empty": nempty}

    # a case that failed once and later completed is not a failure
    failed = [r for r in results if r.get("status") == "failed"
              and r["case_id"] not in completed]

    # ---- report ----
    o = []
    a = o.append
    a(f"# گزارش ارزیابی کور — {os.path.basename(os.path.normpath(run))}")
    a("")
    a(f"- پرونده‌ها: {len(completed)} از {len(cases)} اجرا شده "
      f"(قضاوت‌شده: {len(judged)}"
      + (f"، بدون قضاوت: {len(unjudged)}" if unjudged else "") + ")")
    a(f"- مدل پایه: **{name['base']}** | مدل کاندید: **{name['candidate']}**")
    a("")
    a("## نتیجه کلی")
    a("")
    a("| مدل | برد | both_good | both_bad | نرخ برد (بدون مساوی‌ها) |")
    a("|---|---|---|---|---|")
    decided = wins["base"] + wins["candidate"]
    for s in ("base", "candidate"):
        a(f"| {name[s]} | {wins[s]} | {both['both_good']} "
          f"| {both['both_bad']} | {pct(wins[s], decided)} |")
    winner = max(("base", "candidate"), key=lambda s: wins[s])
    if decided and wins[winner] > wins["candidate" if winner == "base" else "base"]:
        margin = wins[winner] - wins["candidate" if winner == "base" else "base"]
        a("")
        a(f"**برنده: {name[winner]}** با {wins[winner]} برد در برابر "
          f"{decided - wins[winner]} برد حریف (اختلاف {margin}).")
    elif decided:
        a("")
        a("**نتیجه: مساوی** — تعداد بردها برابر است.")
    a("")

    a("## نقاط قوت و ضعف هر مدل (بر اساس برچسب موضوعی)")
    a("")
    for s in ("base", "candidate"):
        other = "candidate" if s == "base" else "base"
        strengths, weak = [], []
        for tag, w in sorted(tag_wins.items()):
            n = tag_decided[tag]
            if n < 3:
                continue
            wr = w[s] / n
            if wr >= 0.6:
                strengths.append(f"{tag} ({w[s]}/{n})")
            elif wr <= 0.4:
                weak.append(f"{tag} ({w[s]}/{n})")
        a(f"**{name[s]}**")
        a(f"- قوی در: {', '.join(strengths) if strengths else '—'}")
        a(f"- ضعیف در: {', '.join(weak) if weak else '—'}")
        a("")

    a("## جزئیات موضوعی (بردها در هر برچسب)")
    a("")
    a("| موضوع | پرونده‌های قضاوت‌شده | برد base | برد candidate |")
    a("|---|---|---|---|")
    for tag, w in sorted(tag_wins.items(), key=lambda kv: -tag_decided[kv[0]]):
        a(f"| {tag} | {tag_decided[tag]} | {w['base']} | {w['candidate']} |")
    a("")

    a("## به تفکیک سختی و دسته")
    a("")
    a(f"| گروه | برد {name['base']} | برد {name['candidate']} | both_good | both_bad |")
    a("|---|---|---|---|---|")
    for g in sorted(diff_stats):
        d = diff_stats[g]
        a(f"| {g} | {d['base']} | {d['candidate']} | {d['both_good']} "
          f"| {d['both_bad']} |")
    a("")

    a("## هزینه و مصرف")
    a("")
    a("| مدل | توکن ورودی | توکن خروجی | کش‌شده | هزینه (دلار) |")
    a("|---|---|---|---|---|")
    for s in ("base", "candidate"):
        u = stats[s]["usage"]
        c = f"${stats[s]['cost']:.4f}" if stats[s]["cost"] is not None \
            else "قیمت وارد نشده"
        a(f"| {name[s]} | {u.get('input', 0):,} | {u.get('output', 0):,} "
          f"| {u.get('cached', 0):,} | {c} |")
    a("")

    a("## سرعت و ابزارها")
    a("")
    a("| مدل | تأخیر میانه (ثانیه) | تأخیر ۹۵٪ | فراخوانی ابزار "
      "| ارجاع به غرفه‌دار | پاسخ خالی |")
    a("|---|---|---|---|---|---|")
    for s in ("base", "candidate"):
        st = stats[s]
        a(f"| {name[s]} | {st['p50'] / 1000:.1f} | {st['p95'] / 1000:.1f} "
          f"| {st['tools']} | {st['esc']} | {st['empty']} |")
    a("")
    if failed:
        a(f"## خطاها")
        a("")
        for r in failed[:20]:
            a(f"- {r['case_id']}: {str(r.get('error'))[:160]}")
        a("")
    if unjudged:
        a("## بدون قضاوت")
        a("")
        a(", ".join(unjudged))
        a("")

    a("## دلایل قضاوت (خلاصه هر پرونده)")
    a("")
    for cid in judged:
        a(f"- **{cid}** [{votes[cid]['verdict']}]: {votes[cid]['reason']}")

    out = os.path.join(run, "report.md")
    open(out, "w").write("\n".join(o) + "\n")
    print(f"report written: {out}")
    if unjudged:
        print(f"WARNING: {len(unjudged)} completed cases have no verdict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
