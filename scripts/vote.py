#!/usr/bin/env python3
"""Record a blind judge verdict for one case (append-only, duplicate-safe).

  python3 vote.py --run runs/myrun --case representative_001 \
      --verdict a_better --reason "پاسخ A موجودی را از ابزار خواند، B حدس زد"

  python3 vote.py --run runs/myrun --pending   # cases still needing a verdict

Verdicts: a_better | b_better | both_good | both_bad
The judge must NOT open runs/<run>/blinding.json before all votes are in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

VERDICTS = {"a_better", "b_better", "both_good", "both_bad"}
# Persian/Arabic, Latin, digits, punctuation, quotes and ZWNJ are fine; any
# other script (CJK, Devanagari, Cyrillic...) in a reason is almost always a
# judge glitch or a shell-quoting accident and would be printed verbatim in
# the final report.
ODD_SCRIPT_RE = re.compile(
    r"[^\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u0000-\u024F"
    r"\u2000-\u206F\u00A0-\u00FF\u200c\u200d«»…–—"
    r"\u2190-\u23FF\u2460-\u27BF]")  # arrows, math, ✓ ✗ and similar


def load_lines(path):
    if not os.path.exists(path):
        return []
    out = []
    for l in open(path):
        l = l.strip()
        if l:
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--case")
    ap.add_argument("--verdict")
    ap.add_argument("--reason", default="")
    ap.add_argument("--pending", action="store_true")
    ap.add_argument("--allow-mixed-script", action="store_true",
                    help="accept a reason that contains non-Persian/Latin "
                         "script characters")
    args = ap.parse_args()

    results = load_lines(os.path.join(args.run, "results.jsonl"))
    votes = load_lines(os.path.join(args.run, "votes.jsonl"))
    voted = {v["case_id"] for v in votes if v.get("verdict")}
    completed = sorted({r["case_id"] for r in results
                        if r.get("status") == "completed"})

    if args.pending or not args.case:
        todo = [c for c in completed if c not in voted]
        print(f"voted {len(voted)}/{len(completed)}; pending: {len(todo)}")
        for c in todo:
            print("  " + c)
        return 0

    if args.case in voted:
        print(f"{args.case}: already voted — append-only, refusing")
        return 1
    data = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "dataset", "cases.jsonl")
    if os.path.exists(data):
        ids = {json.loads(l)["id"] for l in open(data) if l.strip()}
        if args.case not in ids:
            print(f"{args.case}: not a case id of this dataset")
            return 1
    if args.case not in completed:
        print(f"{args.case}: not a completed case in this run")
        return 1
    if args.verdict not in VERDICTS:
        print(f"verdict must be one of {sorted(VERDICTS)}")
        return 1
    if len(args.reason.strip()) < 10:
        print("reason is required (>= 10 chars) — every vote needs a why")
        return 1

    odd = sorted(set(ODD_SCRIPT_RE.findall(args.reason)))
    if odd and not args.allow_mixed_script:
        print(f"reason contains unexpected characters {''.join(odd)!r} — "
              "fix the text (tip: put the reason in single quotes) or pass "
              "--allow-mixed-script")
        return 1

    with open(os.path.join(args.run, "votes.jsonl"), "a") as f:
        f.write(json.dumps({"case_id": args.case, "verdict": args.verdict,
                            "reason": args.reason.strip()},
                           ensure_ascii=False) + "\n")
    print(f"{args.case}: {args.verdict} recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
