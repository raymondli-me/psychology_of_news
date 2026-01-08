"""
Director Agent: Controls the narrative visualization.
Uses litellm with model-specific tweaks from LLM_MODEL_GUIDE.md
"""
import json
import logging
from typing import List, Dict, Any
from litellm import acompletion
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Available chat models (latest models)
DIRECTOR_MODELS = {
    "gemini": "gemini/gemini-2.5-flash",  # Default - fast & cheap
    "claude": "anthropic/claude-sonnet-4-5",
    "gpt": "openai/gpt-5-nano",  # May have issues with structured JSON output
}

class DirectorAction(BaseModel):
    type: str  # "focus_topic", "highlight", "reset", "filter"
    target: Any = None
    description: str = ""

class DirectorResponse(BaseModel):
    answer: str
    actions: List[DirectorAction]
    model_used: str = ""

class DirectorAgent:
    def __init__(self, default_model: str = "gpt"):
        """
        Initialize Director with a default model.

        Args:
            default_model: "gpt", "claude", or "gemini" (or full model_id)
        """
        self.default_model = default_model
        
    def _get_model_id(self, model: str) -> str:
        """Resolve model shortname to full model_id."""
        if model in DIRECTOR_MODELS:
            return DIRECTOR_MODELS[model]
        return model  # Assume it's already a full model_id

    async def direct(
        self,
        query: str,
        context_sentences: List[Dict],
        topic_names: Dict[str, str],
        model: str = None  # "gpt", "claude", "gemini", or full model_id
    ) -> DirectorResponse:
        """
        Generate a response and stage directions based on the query and context.
        """
        
        # Format context for prompt
        context_str = ""
        for i, item in enumerate(context_sentences):
            cluster_name = topic_names.get(str(item['cluster']), f"Topic {item['cluster']}")
            context_str += f"[{i}] Cluster: {cluster_name} | Score: {item.get('mean_score', 5):.1f} | Text: {item['text'][:200]}...\n"

        topics_str = "\n".join([f"ID {k}: {v}" for k, v in topic_names.items()])
        
        prompt = f"""You are the Director of an interactive 3D data visualization.
User Query: "{query}"

DATA CONTEXT (Top relevant sentences):
{context_str}

AVAILABLE TOPICS:
{topics_str}

YOUR JOB:
1. Answer the user's question using the provided data context. Be concise and insightful.
2. DIRECT the visualization to support your answer using specific actions.

AVAILABLE ACTIONS:
- "focus_topic": target = topic_id (integer). Zooms camera to that cluster. Use when discussing a specific topic.
- "highlight_points": target = [list of sentence_ids]. Highlights specific data points.
- "filter_score": target = {{ "min": x, "max": y }}. Hides points outside this score range.
- "reset": target = null. Resets view.

RETURN JSON ONLY:
{{
  "answer": "Your text response here...",
  "actions": [
    {{ "type": "focus_topic", "target": 3, "description": "Focusing on the relevant cluster" }}
  ]
}}
"""
        # Resolve model
        model_key = model or self.default_model
        model_id = self._get_model_id(model_key)

        try:
            kwargs = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "timeout": 120
            }

            # Apply model-specific tweaks (from LLM_MODEL_GUIDE.md)
            model_lower = model_id.lower()

            # GPT-5: needs more tokens, only supports temperature=1
            if "gpt-5" in model_lower:
                kwargs["max_tokens"] = 2000
                kwargs["temperature"] = 1

            # Gemini 2.5+: disable thinking to avoid parsing bug
            elif "gemini-2.5" in model_lower or "gemini-3" in model_lower:
                kwargs["reasoning_effort"] = "none"
                kwargs["temperature"] = 0.3
                kwargs["response_format"] = {"type": "json_object"}

            # Claude: no response_format support via litellm, parse manually
            elif "claude" in model_lower or "anthropic" in model_lower:
                kwargs["temperature"] = 0.3
                # Don't set response_format for Claude

            # Others: standard settings
            else:
                kwargs["temperature"] = 0.3

            response = await acompletion(**kwargs)
            content = response.choices[0].message.content

            # Extract JSON from response (handles markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            data = json.loads(content)
            return DirectorResponse(**data, model_used=model_id)

        except Exception as e:
            logger.error(f"Director error ({model_id}): {e}")
            # Fallback
            return DirectorResponse(
                answer=f"I found some data, but had trouble with the {model_key} model. Try a different one?",
                actions=[],
                model_used=model_id
            )
