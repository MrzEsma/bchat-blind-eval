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
import sys

VERDICTS = {"a_better", "b_better", "both_good", "both_bad"}


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
    if args.case not in completed:
        print(f"{args.case}: not a completed case in this run")
        return 1
    if args.verdict not in VERDICTS:
        print(f"verdict must be one of {sorted(VERDICTS)}")
        return 1
    if len(args.reason.strip()) < 10:
        print("reason is required (>= 10 chars) — every vote needs a why")
        return 1

    with open(os.path.join(args.run, "votes.jsonl"), "a") as f:
        f.write(json.dumps({"case_id": args.case, "verdict": args.verdict,
                            "reason": args.reason.strip()},
                           ensure_ascii=False) + "\n")
    print(f"{args.case}: {args.verdict} recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
