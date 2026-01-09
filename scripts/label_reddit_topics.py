#!/usr/bin/env python3
"""Generate unique topic labels per model (GPT, Claude, Gemini) for Reddit clusters."""

import os
import json
import asyncio
from pathlib import Path
from litellm import acompletion

# Model IDs for litellm
MODELS = {
    "gpt": "openai/gpt-5.1",
    "claude": "anthropic/claude-sonnet-4-5",
    "gemini": "gemini/gemini-2.5-flash"
}

TOPIC_PROMPT = """Analyze these Reddit posts/comments from a cluster and provide a SHORT (2-4 word) topic label that captures what they're discussing.

Posts/Comments:
{texts}

Respond with ONLY the topic label, nothing else. Examples: "Trade Rumors", "Team Chemistry", "Kerr Criticism", "Fan Frustration", "Contract Talk", "Dynasty Debate"."""


async def get_topic_label(texts: list[str], model_key: str) -> str:
    """Generate topic label using specified model via litellm."""
    model_id = MODELS[model_key]
    sample = "\n".join(f"- {t[:200]}" for t in texts[:10])

    try:
        # GPT-5.1 with reasoning_effort="none" - no reasoning overhead
        extra_params = {"reasoning_effort": "none"} if "gpt-5.1" in model_id else {}

        response = await acompletion(
            model=model_id,
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(texts=sample)}],
            timeout=90,
            max_tokens=50,
            **extra_params
        )
        content = response.choices[0].message.content
        if content:
            return content.strip().strip('"').strip("'")[:30]
        else:
            return "Unknown Topic"
    except Exception as e:
        print(f"{model_key.upper()} error: {e}")
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
        tasks = [get_topic_label(texts, model) for model in MODELS.keys()]
        results = await asyncio.gather(*tasks)

        for model, label in zip(MODELS.keys(), results):
            topic_names[model][str(cluster_id)] = label
            print(f"  {model.upper()}: {label}")

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
