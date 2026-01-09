#!/usr/bin/env python3
"""Generate unique topic labels per model (GPT, Claude, Gemini) for Reddit clusters."""

import os
import json
import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai

TOPIC_PROMPT = """Analyze these Reddit posts/comments from a cluster and provide a SHORT (2-4 word) topic label that captures what they're discussing.

Posts/Comments:
{texts}

Respond with ONLY the topic label, nothing else. Examples: "Trade Rumors", "Team Chemistry", "Kerr Criticism", "Fan Frustration", "Contract Talk", "Dynasty Debate"."""


async def label_with_gpt(texts: list[str]) -> str:
    """Generate topic label with GPT."""
    client = AsyncOpenAI()
    sample = "\n".join(f"- {t[:200]}" for t in texts[:10])

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(texts=sample)}],
            temperature=0.3,
            max_tokens=20
        )
        return response.choices[0].message.content.strip().strip('"').strip("'")
    except Exception as e:
        print(f"GPT error: {e}")
        return "Unknown Topic"


async def label_with_claude(texts: list[str]) -> str:
    """Generate topic label with Claude."""
    client = AsyncAnthropic()
    sample = "\n".join(f"- {t[:200]}" for t in texts[:10])

    try:
        response = await client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=20,
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(texts=sample)}]
        )
        return response.content[0].text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Claude error: {e}")
        return "Unknown Topic"


async def label_with_gemini(texts: list[str]) -> str:
    """Generate topic label with Gemini."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    sample = "\n".join(f"- {t[:200]}" for t in texts[:10])

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            TOPIC_PROMPT.format(texts=sample),
            generation_config=genai.types.GenerationConfig(temperature=0.3, max_output_tokens=20)
        )
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Gemini error: {e}")
        return "Unknown Topic"


async def main():
    data_dir = Path(__file__).parent.parent / "reddit" / "data"

    # Load points data
    with open(data_dir / "points_data.json") as f:
        points = json.load(f)

    print(f"Loaded {len(points)} points")

    # Group by cluster
    clusters = {}
    for p in points:
        cid = p["cluster"]
        if cid not in clusters:
            clusters[cid] = []
        clusters[cid].append(p["sentence"])

    print(f"Found {len(clusters)} clusters")

    # Generate labels per model
    topic_names = {"gpt": {}, "claude": {}, "gemini": {}}

    for cluster_id, texts in clusters.items():
        if len(texts) < 3:
            continue

        print(f"\nCluster {cluster_id} ({len(texts)} items):")

        # Get labels from all 3 models in parallel
        gpt_label, claude_label, gemini_label = await asyncio.gather(
            label_with_gpt(texts),
            label_with_claude(texts),
            label_with_gemini(texts)
        )

        topic_names["gpt"][str(cluster_id)] = gpt_label
        topic_names["claude"][str(cluster_id)] = claude_label
        topic_names["gemini"][str(cluster_id)] = gemini_label

        print(f"  GPT:    {gpt_label}")
        print(f"  Claude: {claude_label}")
        print(f"  Gemini: {gemini_label}")

        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)

    # Save
    with open(data_dir / "topic_names.json", "w") as f:
        json.dump(topic_names, f, indent=2)

    print("\n" + "=" * 60)
    print("Topic labels generated!")
    print("=" * 60)
    print(f"Saved to: {data_dir / 'topic_names.json'}")
    print("\nRestart the Reddit server to see new labels:")
    print("  pkill -f 'reddit/server.py' && python reddit/server.py")


if __name__ == "__main__":
    asyncio.run(main())
