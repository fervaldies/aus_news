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
import urllib.request
import urllib.parse
from datetime import datetime

GNEWS_API_KEY      = os.environ.get("GNEWS_API_KEY", "")
GITHUB_TOKEN       = os.environ.get("GITHUB_TOKEN", "")
GITHUB_MODELS_URL  = "https://models.inference.ai.azure.com/chat/completions"
GITHUB_MODEL       = "gpt-4o-mini"


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


def github_models_call(messages, max_tokens=600):
    """Make a call to GitHub Models API and return the response text."""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set")

    payload = json.dumps({
        "model": GITHUB_MODEL,
        "messages": messages,
        "max_tokens": max_tokens
    }).encode("utf-8")

    req = urllib.request.Request(
        GITHUB_MODELS_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GITHUB_TOKEN}"
        }
    )

    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read().decode("utf-8"))

    return result["choices"][0]["message"]["content"]


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

    for q in queries:
        params = urllib.parse.urlencode({
            "q":      q,
            "lang":   "en",
            "max":    "10",   # free tier caps at 10 regardless
            "apikey": GNEWS_API_KEY
        })
        url = f"https://gnews.io/api/v4/search?{params}"

        print(f"📰 Fetching news for query: {q!r}...")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  ⚠️ Query {q!r} failed: {e}")
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
