# Daily X Monitor — @molt_cornelius

## Goal
Fetch and summarize recent posts from @molt_cornelius every day using Claude Code's `/loop` command. No X API key required.

---

## Stack
- `feedparser` — pulls posts from RSSHub (free, no auth)
- `anthropic` — Python SDK for summarization via Claude API
- `/loop` — Claude Code's built-in scheduler to trigger daily

---

## Script Behaviour (single file: `monitor.py`)
1. Fetch last 10 posts from `rsshub.app/twitter/user/molt_cornelius`
2. Format posts into a prompt and send to Claude API
3. Print summary to terminal
4. Save dated output to `summary_YYYYMMDD.txt`
5. Print token usage + estimated cost after each run

---

## Setup Steps
1. `pip install feedparser anthropic`
2. `export ANTHROPIC_API_KEY=sk-...`
3. Place `monitor.py` in your project folder
4. In Claude Code: `/loop every 24h: python monitor.py`

---

## Expected Cost
| Factor | Value |
|---|---|
| Posts per run | ~10 |
| Input tokens | ~300 |
| Output tokens | ~150 |
| Cost per run (Sonnet) | ~$0.001–0.003 |
| Monthly total | < $0.10 |
