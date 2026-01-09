"""Triple LLM rating for sentiment analysis."""

import os
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
import json

# LLM clients
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai


@dataclass
class RatingResult:
    """Rating result from a single model."""
    model: str
    score: int
    raw_response: str


# Rating prompt template
RATING_PROMPT = """Rate how likely the following text implies that Draymond Green will be traded.

Scale:
1 = No trade implication at all
2-3 = Minimal/unlikely trade signals
4-5 = Neutral/ambiguous
6-7 = Moderate trade signals
8-9 = Strong trade indicators
10 = Trade very likely/imminent

Text: "{text}"

Respond with ONLY a single number from 1-10. Nothing else."""


class TripleRater:
    """Rate texts using GPT, Claude, and Gemini."""

    def __init__(self):
        # Initialize clients
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.anthropic = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        # Configure Gemini
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))
        self.gemini = genai.GenerativeModel("gemini-2.0-flash")

        # Model IDs for display
        self.model_ids = {
            "gpt": "gpt-4o-mini",
            "claude": "claude-3-5-haiku-latest",
            "gemini": "gemini-2.0-flash"
        }

    async def rate_gpt(self, text: str) -> RatingResult:
        """Rate with GPT."""
        try:
            response = await self.openai.chat.completions.create(
                model=self.model_ids["gpt"],
                messages=[{"role": "user", "content": RATING_PROMPT.format(text=text)}],
                temperature=0,
                max_tokens=10
            )
            raw = response.choices[0].message.content.strip()
            score = int("".join(c for c in raw if c.isdigit())[:2] or "5")
            score = max(1, min(10, score))
            return RatingResult(model="gpt", score=score, raw_response=raw)
        except Exception as e:
            print(f"GPT error: {e}")
            return RatingResult(model="gpt", score=5, raw_response=str(e))

    async def rate_claude(self, text: str) -> RatingResult:
        """Rate with Claude."""
        try:
            response = await self.anthropic.messages.create(
                model=self.model_ids["claude"],
                max_tokens=10,
                messages=[{"role": "user", "content": RATING_PROMPT.format(text=text)}]
            )
            raw = response.content[0].text.strip()
            score = int("".join(c for c in raw if c.isdigit())[:2] or "5")
            score = max(1, min(10, score))
            return RatingResult(model="claude", score=score, raw_response=raw)
        except Exception as e:
            print(f"Claude error: {e}")
            return RatingResult(model="claude", score=5, raw_response=str(e))

    async def rate_gemini(self, text: str) -> RatingResult:
        """Rate with Gemini."""
        try:
            response = await asyncio.to_thread(
                self.gemini.generate_content,
                RATING_PROMPT.format(text=text),
                generation_config=genai.types.GenerationConfig(
                    temperature=0,
                    max_output_tokens=10
                )
            )
            raw = response.text.strip()
            score = int("".join(c for c in raw if c.isdigit())[:2] or "5")
            score = max(1, min(10, score))
            return RatingResult(model="gemini", score=score, raw_response=raw)
        except Exception as e:
            print(f"Gemini error: {e}")
            return RatingResult(model="gemini", score=5, raw_response=str(e))

    async def rate_single(self, text: str) -> Dict[str, int]:
        """Rate a single text with all three models."""
        results = await asyncio.gather(
            self.rate_gpt(text),
            self.rate_claude(text),
            self.rate_gemini(text)
        )
        return {r.model: r.score for r in results}

    async def rate_batch(
        self,
        texts: List[str],
        batch_size: int = 10,
        progress_callback=None
    ) -> List[Dict[str, int]]:
        """Rate a batch of texts with rate limiting."""
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await asyncio.gather(*[self.rate_single(t) for t in batch])
            results.extend(batch_results)

            if progress_callback:
                progress_callback(len(results), len(texts))
            else:
                print(f"Rated {len(results)}/{len(texts)} texts...")

            # Rate limit pause between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(1)

        return results


async def rate_texts(texts: List[str], batch_size: int = 10) -> List[Dict[str, int]]:
    """Rate a list of texts with all three models."""
    rater = TripleRater()
    return await rater.rate_batch(texts, batch_size)
