#!/usr/bin/env python3
"""Sanity checks for dataset/cases.jsonl. Run after any rebuild.

  python3 lint_dataset.py            # exit 1 if any problem is found

Checks:
  - every case has the fields the rig needs, 6-tool fixtures only
  - a case's text never contains another shop's/owner's codename, and no
    codename from the first release is left (either would decode a word of
    a real shop name); a case naming its own shop or house brand is fine
  - no synthetic codename glued inside another word: the anonymizer once
    replaced substrings, turning «پیگیری» into «پیدیبا» and «سنگین» into
    «سآذین»; a codename may only appear as a whole word
  - no characters from scripts the dataset never uses (CJK, Devanagari, ...)
  - no shop-side message that is empty and is not a product card / picture
  - escalate_to_vendor success fixtures use one of the current texts and do
    not tell the model to name a human («مدیر غرفه», «همکارم»), which the
    system prompt forbids
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

TOOLS = {"get_product_info", "search_products_in_basalam",
         "get_similar_products", "google_search", "escalate_to_vendor",
         "get_customer_order_status"}
# synthetic names the anonymizer uses for shops/owners/brands
_CN = json.load(open(os.path.join(ROOT, "dataset", "codenames.json")))
NAME_CODENAMES = _CN["shop_and_owner_codenames"]
OLD_CODENAMES = _CN["old_codenames"]
CODENAMES = NAME_CODENAMES + OLD_CODENAMES
NAME_LINE = re.compile(r"(نام|مدیر) غرفه\s*\**\s*:")
LETTER = r"[\u0621-\u063A\u0641-\u064A\u0671-\u06D3]"  # letters, not ، ؛ ؟
GLUED = re.compile(
    rf"(?:{LETTER}(?:{'|'.join(map(re.escape, CODENAMES))})"
    rf"|(?:{'|'.join(map(re.escape, CODENAMES))}){LETTER})")
# real words that merely contain a codename
ALLOWED = {"بزارین", "جوانکننده", "مهتابی", "مهتابی،فلاسک،", "بلند", "بلندتر",
           "بلندترین", "بلندمدت", "بلندند", "پلاستیک", "پلاستیکی", "گسترش",
           "گسترده", "می‌مالند", "سرسیلندر", "سیلندر"}
WORD = re.compile(r"[\u0600-\u06FF\u200c]+")
ODD = re.compile(r"[ऀ-ॿ぀-ヿ㐀-鿿가-힯"
                 r"Ѐ-ӿ]")
HUMAN_TITLE = re.compile(r"عنوان (مدیر غرفه|همکارم)")


def norm(text):
    """Arabic letter forms -> Persian, so «ريبان» matches «ریبان»."""
    return text.replace("ي", "ی").replace("ك", "ک").replace("ى", "ی")


def own_use_key(cid, line, start, end):
    """codename with one word of context on each side, for the review list."""
    before = WORD.findall(line[:start])[-1:] or [""]
    after = WORD.findall(line[end:])[:1] or [""]
    return f"{cid}|{before[0]}|{line[start:end]}|{after[0]}"


def phrase_free(line, phrases):
    """The line with the case's full shop/owner names (and «غرفه X» style
    mentions of them) blanked out."""
    for ph in sorted(filter(None, phrases), key=len, reverse=True):
        line = line.replace(ph, " ")
    return line


def strings(x, path=()):
    if isinstance(x, str):
        yield path, norm(x)
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from strings(v, path + (k,))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from strings(v, path + (i,))


def main() -> int:
    path = os.path.join(ROOT, "dataset", "cases.jsonl")
    problems = []
    ids = set()
    whole = 0
    reviewed = set(_CN.get("reviewed_own_uses", []))
    unreviewed = []
    # «سلام X عزیز», «X جان», «خانم/آقای X» with X not a made-up codename
    greet = re.compile(r"(?:سلام|ممنون|مرسی)\s+(\S+(?:\s\S+)?)\s+(?:عزیز|جان)"
                       r"|(?:خانم|آقای)\s+(\S+)"
                       r"|(?:^|[\s،,.!؟])(\S+)\s+جان(?![\u0600-\u06FF])")
    honor = re.compile(r"(?:استاد|جناب|دکتر|حاج|حاجی|مهندس)\s+(\S+)")
    phone = re.compile(r"(?<!\d)(?:\+98|0098|0)?9\d{9}(?!\d)")
    card = re.compile(r"(?<!\d)\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}(?!\d)")
    coupon = re.compile(r"(?<![A-Za-z0-9])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)"
                        r"[A-Za-z0-9]{6,}(?![A-Za-z0-9])")
    greet_ok = {"دوست", "عزیز", "گرامی", "محترم", "وقت", "و", "هم", "سلام",
                "های", "هایی", "هست", "ها", "نوش", "باشید", "داره", "باشه",
                "دل", "دلم", "قربونت"}
    loose = re.compile(rf"(?<!{LETTER})(?:{'|'.join(map(re.escape, NAME_CODENAMES))})(?!{LETTER})")
    old = re.compile(rf"(?<!{LETTER})(?:{'|'.join(map(re.escape, OLD_CODENAMES))})(?!{LETTER})")
    for n, line in enumerate(open(path), 1):
        c = json.loads(line)
        cid = c.get("id", f"line {n}")
        own = set()
        own_phrases = []
        kb_lines = c.get("vendor", {}).get("knowledge_base", "").split("\n")
        for i, kb_line in enumerate(kb_lines):
            if NAME_LINE.search(kb_line):
                own |= set(WORD.findall(kb_line.split(":", 1)[1]))
                ws = [w for w in WORD.findall(kb_line.split(":", 1)[1])
                      if w not in ("گالری", "فروشگاه", "غرفه")]
                own_phrases.append(" ".join(ws))
                # some knowledge bases put the owner's name on the next line
                nxt = kb_lines[i + 1] if i + 1 < len(kb_lines) else ""
                if nxt.strip() and not nxt.lstrip().startswith(("-", "*", "#")):
                    own |= set(WORD.findall(nxt))
                    own_phrases.append(" ".join(WORD.findall(nxt)))
        if cid in ids:
            problems.append(f"{cid}: duplicate id")
        ids.add(cid)
        for k in ("cohort", "difficulty", "tags", "vendor", "turns",
                  "tool_fixtures"):
            if k not in c:
                problems.append(f"{cid}: missing field {k}")
        for f in c.get("tool_fixtures", []):
            if f.get("tool") not in TOOLS:
                problems.append(f"{cid}: fixture for unknown tool {f.get('tool')}")
            if f.get("tool") == "escalate_to_vendor" and \
                    HUMAN_TITLE.search(f.get("result", "")):
                problems.append(f"{cid}: escalate fixture names a human title")
        for m in c.get("prior_messages", []):
            if m.get("role") in ("assistant", "vendor") and \
                    not (m.get("content") or "").strip() and \
                    m.get("message_type") not in ("PRODUCT", "PICTURE") and \
                    not m.get("product_id"):
                problems.append(f"{cid}: empty shop-side message")
        for p, s in strings(c):
            for line in s.split("\n"):
                if NAME_LINE.search(line):
                    continue
                for m in old.finditer(line):
                    problems.append(f"{cid} {'/'.join(map(str, p))}: old "
                                    f"codename left: …{line[max(0, m.start() - 15):m.end() + 15]}…")
                for m in phone.finditer(line):
                    problems.append(f"{cid} {'/'.join(map(str, p))}: phone "
                                    f"number: {m.group()}")
                for m in card.finditer(line):
                    problems.append(f"{cid} {'/'.join(map(str, p))}: card-like "
                                    f"number: {m.group()}")
                if "کد" in line or "code" in line.lower():
                    for m in coupon.finditer(line):
                        t = m.group()
                        if not re.fullmatch(r"code\d+|[A-Z]{1,6}-?\d{1,6}|"
                                            r"[A-Za-z]{1,4}\d{1,5}[A-Za-z]?\d*",
                                            t):
                            problems.append(f"{cid} {'/'.join(map(str, p))}: "
                                            f"possible real code: {t}")
                for m in honor.finditer(line):
                    if m.group(1) not in greet_ok | own and \
                            m.group(1) not in ("سفارش", "عالی", "شما"):
                        problems.append(f"{cid} {'/'.join(map(str, p))}: "
                                        f"title + name?: …{m.group()}…")
                for m in greet.finditer(line):
                    name = WORD.findall(m.group(1) or m.group(2) or "") or \
                        WORD.findall(m.group(3) or "")[-1:]
                    if name and not set(name) <= greet_ok | own and \
                            not name[0].startswith(("های", "هایی")):
                        problems.append(f"{cid} {'/'.join(map(str, p))}: "
                                        f"possible personal name: …{m.group()}…")
                for m in loose.finditer(line):
                    whole += 1
                    if m.group() in own:
                        # a case's own codename is fine only where a review
                        # confirmed it names the shop or its house brand
                        stripped = phrase_free(line, own_phrases)
                        if m.group() in stripped:
                            k = own_use_key(cid, line, m.start(), m.end())
                            if k not in reviewed:
                                unreviewed.append(k)
                                problems.append(
                                    f"{cid} {'/'.join(map(str, p))}: own "
                                    f"codename outside the shop name, not "
                                    f"reviewed: …{line[max(0, m.start() - 15):m.end() + 15]}…")
                    # a case may name its OWN shop/owner (or use its own shop
                    # codename as its house brand); any other codename in the
                    # text would decode a word of some real shop name
                    if m.group() not in own:
                        problems.append(
                            f"{cid} {'/'.join(map(str, p))}: codename of "
                            f"another shop/owner used in text: "
                            f"…{line[max(0, m.start() - 15):m.end() + 15]}…")
            for m in GLUED.finditer(s):
                w = next((x.group() for x in WORD.finditer(s)
                          if x.start() <= m.start() < x.end()), "")
                if any(w.strip("\u200c").startswith(x) for x in ALLOWED):
                    continue
                problems.append(f"{cid} {'/'.join(map(str, p))}: codename "
                                f"glued into a word: …{s[max(0, m.start() - 12):m.end() + 12]}…")
            for m in ODD.finditer(s):
                problems.append(f"{cid} {'/'.join(map(str, p))}: unexpected "
                                f"script {m.group()!r}: …{s[max(0, m.start() - 12):m.end() + 12]}…")
    if "--write-reviewed" in sys.argv:
        others = [x for x in problems if "not reviewed" not in x]
        if others:
            print("fix these first; --write-reviewed only records reviewed "
                  "own-codename uses:")
            for x in others[:50]:
                print(x)
            return 1
        _CN["reviewed_own_uses"] = sorted(reviewed | set(unreviewed))
        json.dump(_CN, open(os.path.join(ROOT, "dataset", "codenames.json"),
                            "w"), ensure_ascii=False, indent=0)
        print(f"marked {len(unreviewed)} own-codename uses as reviewed")
        return 0
    for p in problems[:200]:
        print(p)
    print(f"{len(ids)} cases, {len(problems)} problems")
    print(f"info: {whole} uses of a case's own shop/owner codename in its "
          "text (shop mentions and house brands; review that none stands for "
          "a common word)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
