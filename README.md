# B-Chat Blind Eval — 120-case Persian vendor-chat model evaluation

A self-contained kit for **blind A/B comparison of two LLMs** on realistic
Persian marketplace vendor-chat conversations with tool use. Pick a base model
and a candidate model, run all 120 cases against both, judge the answers
blind, and get a report: winner by votes, per-topic strengths, cost, latency.

Everything the models see is **frozen and offline**: vendor knowledge bases are
anonymized/synthetic, and every tool output comes from per-case fixtures baked
into the dataset. The runner talks only to the two model endpoints you
configure — never to a real shop API.

## What's inside

```
dataset/cases.jsonl      120 cases: vendor knowledge base + conversation +
                         frozen tool fixtures + tags/difficulty/cohort
dataset/summary.json     dataset card (tag/coverage counts)
rig/prompt_template.md   the system prompt template used for every case
rig/tool_schemas.json    the 6 tools exposed to both models
scripts/preflight.py     connectivity + function-calling + usage-stats check
scripts/run_eval.py      blind execution of cases against both models
scripts/vote.py          append-only blind verdict recorder
scripts/report.py        unblinds and writes runs/<run>/report.md
SKILL.md                 instructions an AI agent follows to run it all
examples/sample-case.md  one fully rendered example case
```

## The 6 tools (all mocked)

`get_product_info`, `search_products_in_basalam`, `get_similar_products`,
`google_search`, `escalate_to_vendor`, `get_customer_order_status` —
when a model calls one, the runner returns the case's frozen fixture (or a
fixed fail-closed default), identically for both models.

## Quick start (with Claude Code or any agent that supports skills)

1. Put this folder in your skills directory (or open it in your agent and
   point it at `SKILL.md`).
2. Have two OpenAI-compatible models ready: base_url, model id, API key.
3. Say: «مدل پایه X و مدل کاندید Y را روی دیتاست ارزیابی کور مقایسه کن» —
   the agent will ask for the two models, preflight them, run in batches,
   judge blind, and produce the report.

## Quick start (manual)

```bash
export EVAL_KEY_BASE=...      # key of the base model
export EVAL_KEY_CANDIDATE=... # key of the candidate model
# create models.json (see SKILL.md), then:
python3 scripts/preflight.py models.json
python3 scripts/run_eval.py --config models.json --out runs/demo --limit 3
# judge each runs/demo/judge/<case>.md yourself (verdicts: a_better,
# b_better, both_good, both_bad), then:
python3 scripts/vote.py --run runs/demo --case representative_001 \
    --verdict a_better --reason "..."
python3 scripts/report.py --run runs/demo
```

Requirements: Python 3.8+, standard library only. No real API is contacted
besides the two model endpoints.

## Blindness

Per case, a seeded coin flip decides which model appears as **A** vs **B**.
The mapping lives in `runs/<run>/blinding.json` and must not be opened until
all verdicts are in; `report.py` is the step that unblinds. Judges read only
`judge/<case>.md` files.

## Dataset provenance & privacy

The 120 conversations were sampled from real Persian vendor-chat traffic and
then fully de-identified for release: shop and owner identities, phone
numbers, e-mails, addresses, discount codes, order codes and photo URLs were
replaced with synthetic values, and product ids were renumbered into a fake
range. Tool outputs were captured from real runs and anonymized the same way.
No case can be traced back to a real shop or customer.

Cases carry `tags` (topic labels such as `inventory`, `price`, `escalation`,
`tone`), `difficulty` (1–5) and a `cohort` (96 representative / 24 challenge),
which the final report uses for per-topic strengths.

## License

MIT (see `LICENSE`). The dataset is published for evaluation research; please
keep the anonymization intact if you redistribute.
