"""
Triple-LLM sentence rating using litellm.
Based on the working code from run_analysis.py.
"""

import asyncio
import re
from typing import Optional
import numpy as np
from litellm import acompletion

from .config import Config


async def rate_sentence_single(
    text: str,
    model_name: str,
    model_id: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 10
) -> tuple:
    """Rate one sentence with one model. Returns (model_name, score)."""

    async with semaphore:
        retries = 3
        backoff = 2

        # GPT-5-mini needs more tokens for reasoning
        if "gpt-5" in model_id.lower():
            max_tokens = 1000

        for attempt in range(retries):
            try:
                response = await acompletion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=120,
                    max_tokens=max_tokens
                )
                content = response.choices[0].message.content
                if content:
                    content = content.strip()
                    match = re.search(r'\b(\d+)\b', content)
                    if match:
                        score = int(match.group(1))
                        return (model_name, min(max(score, 1), 10))
                return (model_name, 5)  # Default if no number found

            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "rate" in err_str or "quota" in err_str:
                    wait_time = backoff ** (attempt + 1)
                    print(f"    {model_name}: Rate limit, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                elif attempt < retries - 1:
                    await asyncio.sleep(1)
                else:
                    print(f"    {model_name}: Error after {retries} tries: {str(e)[:60]}")
                    return (model_name, 5)

        return (model_name, 5)


async def rate_sentence_all_models(
    text: str,
    config: Config,
    semaphores: dict
) -> dict:
    """Rate one sentence with all configured models in parallel."""
    prompt = config.get_prompt(text)

    tasks = [
        rate_sentence_single(
            text=text,
            model_name=m.name,
            model_id=m.model_id,
            prompt=prompt,
            semaphore=semaphores[m.name],
            max_tokens=m.max_tokens
        )
        for m in config.models
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    scores = {}
    for r in results:
        if isinstance(r, tuple):
            scores[r[0]] = r[1]
        else:
            print(f"    Exception: {r}")

    # Fill in missing with 5
    for m in config.models:
        if m.name not in scores:
            scores[m.name] = 5

    return scores


async def rate_batch(
    batch: list,
    config: Config,
    semaphores: dict
) -> list:
    """Rate a batch of sentence dicts with all models."""
    tasks = [
        rate_sentence_all_models(item['text'], config, semaphores)
        for item in batch
    ]
    all_scores = await asyncio.gather(*tasks)

    for item, scores in zip(batch, all_scores):
        for name, score in scores.items():
            item[f'{name}_score'] = score

        # Calculate mean
        score_values = [item[f'{m.name}_score'] for m in config.models]
        item['mean_score'] = np.mean(score_values)

    return batch


async def rate_sentences(
    sentences: list,
    config: Config,
    progress_callback=None
) -> list:
    """
    Rate all sentences with all models.

    Args:
        sentences: List of dicts with 'text', 'source', 'article_title' keys
        config: Config object with models and prompt
        progress_callback: Optional callback(current, total)

    Returns:
        List of dicts with added score columns
    """
    # Create semaphores per model
    semaphores = {
        m.name: asyncio.Semaphore(config.max_concurrent_per_model)
        for m in config.models
    }

    all_results = []
    total = len(sentences)

    for i in range(0, total, config.batch_size):
        batch = sentences[i:i + config.batch_size]
        batch_num = i // config.batch_size + 1
        total_batches = (total + config.batch_size - 1) // config.batch_size

        print(f"Batch {batch_num}/{total_batches} ({len(batch)} sentences)...")

        try:
            rated_batch = await rate_batch(batch, config, semaphores)
            all_results.extend(rated_batch)

            # Show sample
            if rated_batch:
                sample = rated_batch[0]
                scores_str = ", ".join(
                    f"{m.name}={sample[f'{m.name}_score']}"
                    for m in config.models
                )
                print(f"  Sample: {scores_str}")

            if progress_callback:
                progress_callback(min(i + config.batch_size, total), total)

        except Exception as e:
            print(f"  Batch error: {e}")
            import traceback
            traceback.print_exc()

    return all_results
