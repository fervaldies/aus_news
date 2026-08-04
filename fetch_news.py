"""
fetch_news.py — aus_news
------------------------
Fetches top Australian news headlines using GNews search API (free),
then translates them to Spanish using GitHub Models (free).

Required environment variables:
  GNEWS_API_KEY   — free API key from gnews.io
  GITHUB_TOKEN    — automatically available in GitHub Actions
"""

import sys
import json
import re
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

GNEWS_API_KEY      = os.environ.get("GNEWS_API_KEY", "")
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL       = "gemini-3.5-flash-lite"
GEMINI_URL         = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

ANTHROPIC_API_KEY  = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL    = "claude-haiku-4-5-20251001"


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_json(text):
    """Extract the first complete JSON object from text, ignoring any preamble."""
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "").strip()
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in response: {repr(text)}")
    return text[start:end + 1]


def clean_title(title):
    """
    Remove trailing source attribution like ' - ABC News' or ' - Reuters'.
    Only strips if the part after the LAST dash is short (likely a source name)
    and does not look like part of a sentence.
    """
    if " - " in title:
        parts = title.rsplit(" - ", 1)
        suffix = parts[1].strip()
        # Only strip if suffix looks like a news source:
        # short (under 30 chars) and no lowercase common sentence words
        sentence_words = {"the", "a", "an", "and", "or", "but", "in",
                          "on", "at", "to", "of", "for", "is", "are",
                          "was", "were", "not", "new", "old", "it"}
        words = suffix.lower().split()
        looks_like_source = (
            len(suffix) < 30 and
            not any(w in sentence_words for w in words)
        )
        if looks_like_source:
            return parts[0].strip()
    return title.strip()


def gemini_call(messages, max_tokens=600):
    """Make a call to Gemini API and return the response text."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    prompt_text = "\n\n".join(m["content"] for m in messages)

    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt_text}]}],
        "generationConfig": {"maxOutputTokens": max_tokens}
    }).encode("utf-8")

    url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                result = json.loads(r.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            print(f"  ❌ Gemini API error {e.code}: {error_body}")
            transient = e.code == 429 or 500 <= e.code < 600
            if transient and attempt < 2:
                wait = 5 * (attempt + 1)
                print(f"  ⏳ Gemini transient error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def claude_call(messages, max_tokens=600):
    """Fallback: call Claude Haiku via the Anthropic API."""
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY is not set")

    prompt_text = "\n\n".join(m["content"] for m in messages)

    payload = json.dumps({
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt_text}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type":      "application/json",
            "x-api-key":         ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read().decode("utf-8"))
    return result["content"][0]["text"]


def github_models_call(messages, max_tokens=600):
    """Try Gemini first (free); fall back to Claude Haiku if Gemini is down.
    Keeps the same function name/interface so callers don't need to change."""
    try:
        return gemini_call(messages, max_tokens)
    except Exception as e:
        print(f"  ⚠️ Gemini unavailable after retries ({e}) — falling back to Claude Haiku...")
        return claude_call(messages, max_tokens)


# ── news fetching ─────────────────────────────────────────────────────────────

import random

# ── topic pool ──────────────────────────────────────────────────────────────

AUSTRALIA_TOPIC_POOL = [
    "Sydney",
    "Melbourne",
    "Brisbane",
    "Perth",
    "Adelaide",
    "Canberra",
    "Australian government",
    "Australian economy",
    "Australian politics",
    "Australian sport",
    "Australian rules football OR AFL",
    "NRL",
    "Australian weather OR bushfire OR flood",
    "Australian immigration",
    "Australian housing market",
    "Australian technology",
    "Australian health",
    "Australian education",
    "Australian environment OR climate",
    "Australian crime",
    "Australian business",
    "Queensland",
    "New South Wales",
    "Victoria Australia",
    "Western Australia",
    "Australian tourism",
    "Australian mining",
    "Australian wildlife",
    "Australian indigenous",
    "Australian defence OR military",
]


def fetch_australia_news():
    """
    Fetch headlines specifically ABOUT Australia using GNews search endpoint.
    Always includes the general 'Australia' query (so top national stories
    never get missed), plus 2 randomly chosen topics from a larger pool each
    day for variety. Free GNews tier caps each request at 10 articles
    regardless of the 'max' param, so variety comes from multiple queries.
    """
    if not GNEWS_API_KEY:
        raise ValueError("GNEWS_API_KEY is not set")

    random_topics = random.sample(AUSTRALIA_TOPIC_POOL, 2)
    queries = ["Australia"] + random_topics

    print(f"🎲 Today's topics: {queries}")

    all_headlines = []
    seen = set()

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(2)  # avoid tripping GNews's short-window rate limit

        params = urllib.parse.urlencode({
            "q":      q,
            "lang":   "en",
            "max":    "10",   # free tier caps at 10 regardless
            "apikey": GNEWS_API_KEY
        })
        url = f"https://gnews.io/api/v4/search?{params}"

        print(f"📰 Fetching news for query: {q!r}...")
        data = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=15) as r:
                    data = json.loads(r.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < 2:
                    wait = 5 * (attempt + 1)
                    print(f"  ⏳ Rate limited, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ⚠️ Query {q!r} failed: {e}")
                    break
            except Exception as e:
                print(f"  ⚠️ Query {q!r} failed: {e}")
                break

        if data is None:
            continue

        articles = data.get("articles", [])
        print(f"  → {len(articles)} articles")

        for article in articles:
            title = clean_title(article.get("title", ""))
            if title and len(title) > 10 and title not in seen:
                seen.add(title)
                all_headlines.append(title)

    if len(all_headlines) < 5:
        raise ValueError(f"Only {len(all_headlines)} valid headlines after merging — need at least 5")

    print(f"✅ Total unique headlines collected: {len(all_headlines)}")
    return all_headlines


def pick_best_5_australia(headlines):
    """Use GitHub Models to pick the 5 most important Australia-specific stories."""
    numbered = "\n".join(f"{i}. {h}" for i, h in enumerate(headlines))
    print("🤖 Picking best 5 via GitHub Models...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "You are an Australian news editor. Pick the 5 best stories from the list.\n\n"
            "STEP 1 — Remove duplicates FIRST. Group headlines that describe the same "
            "underlying event (same place/person + same topic = same story, even if the "
            "wording, angle, or numbers differ). Example: 'Sydney floods force evacuations' "
            "and 'Thousands flee Sydney flooding' are the SAME story — keep only ONE. "
            "From each group pick the single clearest headline.\n\n"
            "STEP 2 — From the de-duplicated stories, pick the 5 most important. "
            "Each of the 5 MUST be about a different event. Never select two headlines "
            "that share the same place and topic.\n\n"
            "AVOID vague headlines with no specific person, place, company, or concrete "
            "event, minor sport or celebrity items, and anything not specifically about "
            "Australia.\n\n"
            "PREFER headlines naming a specific Australian person, place, company, or "
            "concrete event, with variety of topics.\n\n"
            "Return ONLY raw JSON, no markdown, no backticks:\n"
            '{"selected_indexes": [0, 1, 2, 3, 4]}\n\n'
            f"Indexes are 0-based. Stories:\n\n{numbered}"
        )
    }], max_tokens=100)
    text    = extract_json(text)
    indexes = json.loads(text)["selected_indexes"]
    selected = [headlines[i] for i in indexes if i < len(headlines)]
    if len(selected) < 5:
        for h in headlines:
            if h not in selected:
                selected.append(h)
            if len(selected) == 5:
                break
    return [{"title": t} for t in selected[:5]]


# ── translation ───────────────────────────────────────────────────────────────

def translate_to_spanish(headlines):
    """Translate headlines to Spanish (Spain) using GitHub Models."""
    headlines_text = "\n".join(f"- {n['title']}" for n in headlines)

    print("🌐 Translating via GitHub Models...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "Translate these headlines to Spanish from Spain. "
            "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
            '{"news": [{"title": "translated"}, {"title": "translated"}, '
            '{"title": "translated"}, {"title": "translated"}, {"title": "translated"}]}\n\n'
            f"Headlines:\n{headlines_text}"
        )
    }])

    return extract_json(text)


def rewrite_headlines(headlines):
    """Rewrite raw API headlines to sound natural and punchy for social media."""
    numbered = "\n".join(f"{i+1}. {n['title']}" for i, n in enumerate(headlines))
    print("✍️ Rewriting headlines for natural language...")
    text = github_models_call([{
        "role": "user",
        "content": (
            "Rewrite these 5 news headlines to sound natural and punchy for social media. "
            "Rules: remove prefixes like 'LIVE UPDATES:', 'Study:', 'Report:', 'Breaking:'. "
            "Remove dates in parentheses. Replace semicolons with a comma or 'and'. "
            "Simplify scientific jargon into plain language. Remove marketing-speak. "
            "Keep each headline under 20 words and factually accurate. "
            "Return ONLY raw JSON, no markdown, no backticks:\n"
            '{"news": [{"title": "rewritten"}, {"title": "rewritten"}, '
            '{"title": "rewritten"}, {"title": "rewritten"}, {"title": "rewritten"}]}\n\n'
            f"Headlines:\n{numbered}"
        )
    }])
    text = extract_json(text)
    rewritten = json.loads(text)["news"]
    return rewritten

# ── main ──────────────────────────────────────────────────────────────────────

def get_news(day_name):
    # Step 1 — fetch Australia-specific news
    all_headlines = fetch_australia_news()

    # Step 2 — pick 5 best Australia-specific stories
    en_news  = pick_best_5_australia(all_headlines)
    en_news  = rewrite_headlines(en_news)
    en_data  = {"news": en_news}
    print(f"✅ Selected {len(en_data['news'])} headlines:")
    for n in en_data["news"]:
        print(f"  - {n['title']}")

    # Step 3 — translate to Spanish
    es_text = translate_to_spanish(en_data["news"])
    es_data = json.loads(es_text)
    print("✅ Translation complete")

    # Step 4 — write YML files
    date_str = datetime.now().strftime("%Y-%m-%d")

    def build_yml(data):
        lines = [f"date: {date_str}", f"day: {day_name}", "news:"]
        for item in data["news"]:
            title = item["title"].replace('"', "'")
            lines.append(f'  - title: "{title}"')
        return "\n".join(lines) + "\n"

    with open(f"{day_name}NewsEN.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(en_data))
    with open(f"{day_name}NewsES.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(es_data))

    print(f"✅ Created {day_name}NewsEN.yml and {day_name}NewsES.yml")


if __name__ == "__main__":
    day_name = sys.argv[1]
    get_news(day_name)
