import anthropic
import sys
import json
from datetime import datetime

def call_with_tools(client, messages):
    """Handle the tool use loop and return final text."""
    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages
        )

        # If done, extract and return text
        if response.stop_reason == "end_turn":
            return "".join(b.text for b in response.content if hasattr(b, "text"))

        # Otherwise, process tool use and continue the loop
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": block.input.get("query", "")
                })
        messages.append({"role": "user", "content": tool_results})

def get_news(day_name):
    client = anthropic.Anthropic()

    # --- Fetch 2 Australian news in English ---
    en_messages = [{
        "role": "user",
        "content": (
            "Search for the 2 most important news today about or related to Australia. "
            "Return ONLY a valid JSON object, no markdown, no explanation:\n"
            '{"news": [{"title": "headline under 20 words"}, {"title": "headline under 20 words"}]}'
        )
    }]

    en_text = call_with_tools(client, en_messages)
    print(f"EN response: {en_text}")
    en_data = json.loads(en_text)

    # --- Translate to Spanish (Spain) ---
    headlines = "\n".join(f"- {n['title']}" for n in en_data["news"])

    es_response = client.messages.create(
        model="claude-haiku-4-5-20251001",
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
    print(f"ES response: {es_text}")
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
