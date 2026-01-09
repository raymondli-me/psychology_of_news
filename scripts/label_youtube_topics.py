#!/usr/bin/env python3
"""Generate unique topic labels per model (GPT, Claude, Gemini) for YouTube clusters."""

import os
import json
import asyncio
from pathlib import Path
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai

# Load env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")


async def label_with_gpt(cluster_texts: dict[int, str]) -> dict[int, str]:
    """Use GPT to label clusters."""
    client = AsyncOpenAI()
    labels = {}

    for cluster_id, sample_text in cluster_texts.items():
        prompt = f"""Based on these YouTube video/comment samples about Draymond Green trade discussions, give this cluster a short, descriptive label (2-4 words max).

Samples:
{sample_text[:2000]}

Respond with ONLY the label, nothing else."""

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20
            )
            labels[cluster_id] = response.choices[0].message.content.strip().strip('"')
        except Exception as e:
            print(f"GPT error for cluster {cluster_id}: {e}")
            labels[cluster_id] = f"Cluster {cluster_id}"

    return labels


async def label_with_claude(cluster_texts: dict[int, str]) -> dict[int, str]:
    """Use Claude to label clusters."""
    client = AsyncAnthropic()
    labels = {}

    for cluster_id, sample_text in cluster_texts.items():
        prompt = f"""Based on these YouTube video/comment samples about Draymond Green trade discussions, give this cluster a short, descriptive label (2-4 words max).

Samples:
{sample_text[:2000]}

Respond with ONLY the label, nothing else."""

        try:
            response = await client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=20,
                messages=[{"role": "user", "content": prompt}]
            )
            labels[cluster_id] = response.content[0].text.strip().strip('"')
        except Exception as e:
            print(f"Claude error for cluster {cluster_id}: {e}")
            labels[cluster_id] = f"Cluster {cluster_id}"

    return labels


async def label_with_gemini(cluster_texts: dict[int, str]) -> dict[int, str]:
    """Use Gemini to label clusters."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")
    labels = {}

    for cluster_id, sample_text in cluster_texts.items():
        prompt = f"""Based on these YouTube video/comment samples about Draymond Green trade discussions, give this cluster a short, descriptive label (2-4 words max).

Samples:
{sample_text[:2000]}

Respond with ONLY the label, nothing else."""

        try:
            response = await asyncio.to_thread(
                model.generate_content, prompt
            )
            labels[cluster_id] = response.text.strip().strip('"')
        except Exception as e:
            print(f"Gemini error for cluster {cluster_id}: {e}")
            labels[cluster_id] = f"Cluster {cluster_id}"

    return labels


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
        sample = texts[:10]  # First 10 items
        sampled[cluster_id] = "\n---\n".join(sample)

    print("Generating topic labels with 3 models...")

    # Run all labeling concurrently
    gpt_labels, claude_labels, gemini_labels = await asyncio.gather(
        label_with_gpt(sampled),
        label_with_claude(sampled),
        label_with_gemini(sampled)
    )

    # Combine into topic_names format
    topic_names = {
        "gpt": {str(k): v for k, v in gpt_labels.items()},
        "claude": {str(k): v for k, v in claude_labels.items()},
        "gemini": {str(k): v for k, v in gemini_labels.items()}
    }

    # Save
    with open(data_dir / "topic_names.json", "w") as f:
        json.dump(topic_names, f, indent=2)

    print("\nTopic labels generated:")
    for model in ["gpt", "claude", "gemini"]:
        print(f"\n{model.upper()}:")
        for cluster_id, name in topic_names[model].items():
            print(f"  Cluster {cluster_id}: {name}")


if __name__ == "__main__":
    asyncio.run(main())
