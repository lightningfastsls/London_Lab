import os
import re
import sys
from pathlib import Path
from html import unescape
import feedparser
import anthropic
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
from datetime import datetime

# Fix Windows console encoding for Unicode output
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load .env from repo root
load_dotenv(Path(__file__).parent / ".env")

# --- Config ---
HANDLE = "molt_cornelius"
RSS_URL = "https://rss.app/feeds/K1nc8lgREqH3S7PR.xml"
MAX_POSTS = 10
MODEL = "claude-sonnet-4-6"
ARTICLE_MAX_CHARS = 3000


def strip_html(html):
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_article_text(browser, url):
    """Use headless browser to fetch X article content."""
    try:
        page = browser.new_page()
        page.goto(url, timeout=30000)
        page.wait_for_timeout(5000)
        text = page.inner_text("body")
        page.close()
        # Clean up X chrome (nav, login prompts, etc.)
        # Article content starts after the handle line
        marker = f"@{HANDLE}"
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
        return text[:ARTICLE_MAX_CHARS].strip()
    except Exception as e:
        print(f"    (failed to fetch: {e})")
        return ""


def fetch_posts():
    print(f"Fetching posts from @{HANDLE}...")
    feed = feedparser.parse(RSS_URL)

    # Launch browser once, reuse for all articles
    pw = sync_playwright().start()
    browser = pw.chromium.launch(headless=True)

    posts = []
    for entry in feed.entries[:MAX_POSTS]:
        text = strip_html(entry.get("summary", ""))
        link = entry.get("link", "")

        # If the post is mostly an article link, fetch the actual content
        if re.search(r"x\.com/i/article/\d+", text) and link:
            print(f"  -> Fetching article: {link}")
            article_text = fetch_article_text(browser, link)
            if article_text:
                text = article_text

        posts.append({
            "text": text,
            "link": link,
            "published": entry.get("published", ""),
        })

    browser.close()
    pw.stop()
    print(f"  -> {len(posts)} posts fetched")
    return posts


def summarize(posts):
    if not posts:
        return "No posts found to summarize."

    formatted = "\n".join(
        f"- [{p['published']}] {p['text']} ({p['link']})"
        for p in posts
    )

    client = anthropic.Anthropic()

    prompt = f"""You are summarizing recent activity from @{HANDLE} on X (Twitter).

Here are their last {len(posts)} posts:
{formatted}

Write a concise 3-5 sentence summary covering:
- Main themes or topics they posted about
- Tone and any notable highlights
- Any standout post worth flagging"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    usage = response.usage
    print(f"  -> Tokens used: {usage.input_tokens} input / {usage.output_tokens} output")
    cost = usage.input_tokens * 3e-6 + usage.output_tokens * 15e-6
    print(f"  -> Estimated cost: ${cost:.5f}")

    return response.content[0].text


def run():
    print(f"\n=== X Monitor: @{HANDLE} ===")
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    posts = fetch_posts()
    summary = summarize(posts)

    print(f"\n--- Summary ---\n{summary}\n")

    filename = f"summary_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"@{HANDLE} -- {datetime.now().strftime('%Y-%m-%d')}\n\n")
        f.write(summary + "\n\n")
        f.write("--- Raw posts ---\n")
        for p in posts:
            f.write(f"[{p['published']}] {p['text'][:500]}\n{p['link']}\n\n")

    print(f"Saved to {filename}")


if __name__ == "__main__":
    run()
