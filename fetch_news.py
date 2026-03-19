import anthropic
import sys
import json
from datetime import datetime

def get_news(day_name):
    client = anthropic.Anthropic()

    # --- Fetch 2 Australian news in English ---
    en_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{
            "role": "user",
            "content": (
                "Search for the 2 most important news today about or related to Australia. "
                "Return ONLY a valid JSON object, no markdown, no explanation:\n"
                '{"news": [{"title": "headline under 20 words"}, {"title": "headline under 20 words"}]}'
            )
        }]
    )

    en_text = "".join(b.text for b in en_response.content if hasattr(b, "text"))
    en_data = json.loads(en_text)

    # --- Translate to Spanish (Spain) ---
    headlines = "\n".join(f"- {n['title']}" for n in en_data["news"])

    es_response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                "Translate these headlines to Spanish from Spain. "
                "Return ONLY a valid JSON object, no markdown, no explanation:\n"
                '{"news": [{"title": "translated"}, {"title": "translated"}]}\n\n'
                f"Headlines:\n{headlines}"
            )
        }]
    )

    es_text = "".join(b.text for b in es_response.content if hasattr(b, "text"))
    es_data = json.loads(es_text)

    # --- Write YML files ---
    date_str = datetime.now().strftime("%Y-%m-%d")

    def build_yml(data):
        lines = [f"date: {date_str}", f"day: {day_name}", "news:"]
        for item in data["news"]:
            lines.append(f'  - title: "{item["title"]}"')
        return "\n".join(lines) + "\n"

    with open(f"{day_name}NewsEN.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(en_data))

    with open(f"{day_name}NewsES.yml", "w", encoding="utf-8") as f:
        f.write(build_yml(es_data))

    print(f"✅ Created {day_name}NewsEN.yml and {day_name}NewsES.yml")

if __name__ == "__main__":
    day_name = sys.argv[1]
    get_news(day_name)
