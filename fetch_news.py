"""
fetch_news.py — aus_news
------------------------
Fetches top Australian news headlines using GNews API (free),
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
import urllib.error
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
    """Remove source attribution like ' - ABC News' from the end of a headline."""
    if " - " in title:
        title = title.rsplit(" - ", 1)[0]
    return title.strip()


# ── news fetching ─────────────────────────────────────────────────────────────

def fetch_australia_news():
    """Fetch top Australian headlines from GNews API and return as dict."""
    if not GNEWS_API_KEY:
        raise ValueError("GNEWS_API_KEY is not set")

    url = (
        f"https://gnews.io/api/v4/top-headlines"
        f"?country=au&lang=en&max=10&apikey={GNEWS_API_KEY}"
    )
    print(f"📰 Fetching from GNews API...")
    with urllib.request.urlopen(url, timeout=15) as r:
        data = json.loads(r.read().decode("utf-8"))

    articles = data.get("articles", [])
    print(f"  GNews returned {len(articles)} articles")

    if len(articles) < 5:
        raise ValueError(f"GNews returned only {len(articles)} articles — need at least 5")

    headlines = []
    for article in articles[:5]:
        title = clean_title(article.get("title", ""))
        if title:
            headlines.append({"title": title})

    if len(headlines) < 5:
        raise ValueError(f"Only {len(headlines)} valid headlines after cleaning")

    return {"news": headlines}


# ── translation ───────────────────────────────────────────────────────────────

def translate_with_github_models(headlines):
    """Translate headlines to Spanish (Spain) using GitHub Models (free)."""
    if not GITHUB_TOKEN:
        raise ValueError("GITHUB_TOKEN is not set")

    headlines_text = "\n".join(f"- {n['title']}" for n in headlines)

    payload = json.dumps({
        "model": GITHUB_MODEL,
        "messages": [{
            "role": "user",
            "content": (
                "Translate these headlines to Spanish from Spain. "
                "Return ONLY raw JSON, no markdown, no backticks, no explanation:\n"
                '{"news": [{"title": "translated"}, {"title": "translated"}, '
                '{"title": "translated"}, {"title": "translated"}, {"title": "translated"}]}\n\n'
                f"Headlines:\n{headlines_text}"
            )
        }],
        "max_tokens": 600
    }).encode("utf-8")

    req = urllib.request.Request(
        GITHUB_MODELS_URL,
        data=payload,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {GITHUB_TOKEN}"
        }
    )

    print("🌐 Translating via GitHub Models...")
    with urllib.request.urlopen(req, timeout=30) as r:
        result = json.loads(r.read().decode("utf-8"))

    text = result["choices"][0]["message"]["content"]
    return extract_json(text)


# ── main ──────────────────────────────────────────────────────────────────────

def get_news(day_name):
    # Step 1 — fetch Australian news
    en_data = fetch_australia_news()
    print(f"✅ {len(en_data['news'])} headlines fetched:")
    for n in en_data["news"]:
        print(f"  - {n['title']}")

    # Step 2 — translate to Spanish
    es_text = translate_with_github_models(en_data["news"])
    es_data = json.loads(es_text)
    print("✅ Translation complete")

    # Step 3 — write YML files
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
