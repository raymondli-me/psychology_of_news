#!/usr/bin/env python3
"""Generate unique topic labels per model (GPT, Claude, Gemini) for YouTube clusters."""

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

TOPIC_PROMPT = """Analyze these YouTube video/comment samples about Draymond Green trade discussions and provide a SHORT (2-4 word) topic label.

Samples:
{texts}

Respond with ONLY the topic label, nothing else. Examples: "Trade Rumors", "Kerr Criticism", "Fan Reactions", "Dynasty Debate"."""


async def get_topic_label(texts: str, model_key: str) -> str:
    """Generate topic label using specified model via litellm."""
    model_id = MODELS[model_key]

    try:
        # GPT-5.1 with reasoning_effort="none" - no reasoning overhead
        extra_params = {"reasoning_effort": "none"} if "gpt-5.1" in model_id else {}

        response = await acompletion(
            model=model_id,
            messages=[{"role": "user", "content": TOPIC_PROMPT.format(texts=texts[:2000])}],
            timeout=90,
            max_tokens=50,
            **extra_params
        )
        content = response.choices[0].message.content
        if content:
            return content.strip().strip('"').strip("'")[:30]
        else:
            return f"Topic"
    except Exception as e:
        print(f"{model_key.upper()} error for cluster: {e}")
        return f"Topic"


async def main():
    data_dir = Path(__file__).parent.parent / "youtube" / "data"

    # Load points data
    with open(data_dir / "points_data.json") as f:
        points = json.load(f)

    # Group texts by cluster
    cluster_texts = {}
    for point in points:
        cluster_id = point["cluster"]
        if cluster_id not in cluster_texts:
            cluster_texts[cluster_id] = []
        cluster_texts[cluster_id].append(point["sentence"])

    # Sample texts for each cluster
    sampled = {}
    for cluster_id, texts in cluster_texts.items():
        sample = texts[:10]
        sampled[cluster_id] = "\n---\n".join(sample)

    print(f"Generating topic labels for {len(sampled)} clusters with 3 models...")

    # Generate labels for all clusters
    topic_names = {"gpt": {}, "claude": {}, "gemini": {}}

    for cluster_id, sample_text in sampled.items():
        print(f"\nCluster {cluster_id}:")

        # Get labels from all 3 models in parallel
        tasks = [get_topic_label(sample_text, model) for model in MODELS.keys()]
        results = await asyncio.gather(*tasks)

        for model, label in zip(MODELS.keys(), results):
            topic_names[model][str(cluster_id)] = label
            print(f"  {model.upper()}: {label}")

        await asyncio.sleep(0.5)

    # Save
    with open(data_dir / "topic_names.json", "w") as f:
        json.dump(topic_names, f, indent=2)

    print("\n" + "=" * 60)
    print("Topic labels generated!")
    print("=" * 60)
    print(f"Saved to: {data_dir / 'topic_names.json'}")


if __name__ == "__main__":
    asyncio.run(main())
