"""Topic naming using LLMs."""

import os
import asyncio
from typing import List, Dict
import google.generativeai as genai


TOPIC_PROMPT = """Analyze these texts from a cluster and provide a short (2-4 word) topic label.

Texts:
{texts}

Respond with ONLY the topic label, nothing else. Examples: "Trade Rumors", "Team Chemistry", "Contract Issues", "Fan Reactions"."""


async def name_cluster_gemini(texts: List[str], max_samples: int = 10) -> str:
    """Generate a topic name for a cluster using Gemini."""
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Sample texts if too many
    sample = texts[:max_samples] if len(texts) > max_samples else texts
    texts_str = "\n".join(f"- {t[:200]}" for t in sample)

    try:
        response = await asyncio.to_thread(
            model.generate_content,
            TOPIC_PROMPT.format(texts=texts_str),
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=20
            )
        )
        return response.text.strip().strip('"').strip("'")
    except Exception as e:
        print(f"Topic naming error: {e}")
        return "Unknown Topic"


async def name_all_clusters(
    texts: List[str],
    labels: List[int],
    min_cluster_size: int = 3
) -> Dict[int, str]:
    """Generate topic names for all clusters."""
    from collections import defaultdict

    # Group texts by cluster
    cluster_texts = defaultdict(list)
    for text, label in zip(texts, labels):
        if label >= 0:  # Skip noise
            cluster_texts[label].append(text)

    # Name each cluster
    topic_names = {}
    for cluster_id, cluster_txts in cluster_texts.items():
        if len(cluster_txts) >= min_cluster_size:
            name = await name_cluster_gemini(cluster_txts)
            topic_names[cluster_id] = name
            print(f"Cluster {cluster_id}: {name}")
            await asyncio.sleep(0.5)  # Rate limit

    return topic_names
