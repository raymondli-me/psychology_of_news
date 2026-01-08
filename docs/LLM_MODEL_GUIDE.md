# LLM Model Configuration Guide

> Lessons learned from debugging triple-LLM rating system (Jan 2026)
> Reference this before making model changes!

---

## Working Model Configuration (Tested Jan 2026)

```python
models = [
    ModelConfig(name="GPT", model_id="openai/gpt-5-nano"),
    ModelConfig(name="Claude", model_id="anthropic/claude-sonnet-4-5"),
    ModelConfig(name="Gemini", model_id="gemini/gemini-2.5-flash"),
]
```

---

## Model-Specific Rules

### OpenAI GPT-5 Models

| Model | Cost (per 1M) | Status | Notes |
|-------|---------------|--------|-------|
| `openai/gpt-5-nano` | $0.05 in / $0.40 out | **RECOMMENDED** | Cheapest, fastest |
| `openai/gpt-5-mini` | $0.25 in / $2.00 out | Works | Returns empty with low max_tokens |
| `openai/gpt-5` | $1.25 in / $10.00 out | Works | Full model |
| `openai/gpt-4o-mini` | - | Works | Older, but reliable |

**Critical Rules for GPT-5:**

1. **max_tokens must be 1000+** for GPT-5 models
   - With low max_tokens (10-100), GPT-5 returns EMPTY responses
   - This is because GPT-5 uses internal reasoning tokens
   ```python
   if "gpt-5" in model_id.lower():
       max_tokens = 1000
   ```

2. **reasoning_effort parameter does NOT work via chat completions**
   - Only works via Responses API (`openai/responses/gpt-5-nano`)
   - gpt-5-nano already defaults to `reasoning_effort=none` internally
   - Don't try to set it - you'll get `BadRequestError: Unsupported value`

3. **Environment variable:** `OPENAI_API_KEY`

---

### Anthropic Claude Models

| Model | Status | Notes |
|-------|--------|-------|
| `anthropic/claude-sonnet-4-5` | **RECOMMENDED** | Works reliably |
| `anthropic/claude-opus-4-5` | Works | More expensive |

**Rules for Claude:**
- No special handling needed
- Works with standard max_tokens (10-100 is fine)
- **Environment variable:** `ANTHROPIC_API_KEY`

---

### Google Gemini Models

| Model | Status | Notes |
|-------|--------|-------|
| `gemini/gemini-2.5-flash` | **RECOMMENDED** | Latest, requires `reasoning_effort="none"` |
| `gemini/gemini-2.0-flash` | Works | Fallback if 2.5 has issues |
| `gemini/gemini-2.5-flash-preview-09-2025` | BROKEN | litellm parsing bug |
| `gemini/gemini-1.5-flash` | 404 | Deprecated/removed |
| `gemini/gemini-1.5-pro` | 404 | Deprecated/removed |
| `gemini/gemini-3.0-*` | 404 | Doesn't exist yet |

**Critical Rules for Gemini 2.5:**

1. **MUST pass `reasoning_effort="none"`** for Gemini 2.5+ models
   - Gemini 2.5 has a "thinking" feature that returns special response format
   - litellm has a parsing bug with this format
   - Setting `reasoning_effort="none"` disables thinking and avoids the bug
   - Also makes requests ~96% cheaper according to Google docs
   ```python
   if "gemini-2.5" in model_id.lower() or "gemini-3" in model_id.lower():
       kwargs["reasoning_effort"] = "none"
   ```

2. **The litellm error looks like:**
   ```
   GeminiException - Received={'candidates': [{'content': {'role': 'model'}, ...
   Error converting to valid response block=cannot access local variable 'thought_signatures'
   ```

3. **Sync vs Async behavior differs:**
   - Gemini 2.5 may work in sync `completion()` but fail in async `acompletion()`
   - Always test with async if your app uses async

4. **Environment variable:** `GOOGLE_API_KEY`

---

## Async/Event Loop Issues

### nest_asyncio Bug

**Problem:** `nest_asyncio.apply()` breaks litellm's async timeout handling

**Symptoms:**
- All API calls timeout with: `Timeout context manager should b...`
- Direct sync calls work, but async batching fails
- Happens when nest_asyncio is applied globally

**Solution:** Only apply nest_asyncio in Jupyter/Colab environments:
```python
def _in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False

# Only apply in notebooks
if _in_notebook():
    import nest_asyncio
    nest_asyncio.apply()
```

**Why it matters:**
- Jupyter/Colab have a running event loop, need nest_asyncio for `asyncio.run()`
- CLI/scripts don't have this issue
- nest_asyncio somehow interferes with litellm's internal timeout handling

---

## API Key Names

| Provider | Environment Variable | Colab Secret Name |
|----------|---------------------|-------------------|
| OpenAI | `OPENAI_API_KEY` | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` | `CLAUDE_API_KEY` (common alias) |
| Google | `GOOGLE_API_KEY` | `GOOGLE_API_KEY` |
| Event Registry | `EVENT_REGISTRY_API_KEY` | `EVENT_REGISTRY_NEWS` |

---

## Common Errors and Solutions

### 1. Empty Response from GPT-5
```
GPT-5-mini: (empty)
```
**Cause:** max_tokens too low
**Fix:** Set `max_tokens=1000` for GPT-5 models

### 2. Gemini Parsing Error
```
GeminiException - Received={'candidates': ...
cannot access local variable 'thought_signatures'
```
**Cause:** Gemini 2.5's thinking feature
**Fix:** Pass `reasoning_effort="none"`

### 3. All Timeouts in Async
```
Timeout context manager should b...
```
**Cause:** nest_asyncio applied globally
**Fix:** Only apply nest_asyncio in Jupyter/Colab

### 4. 404 Model Not Found
```
models/gemini-1.5-flash is not found
```
**Cause:** Model deprecated or name changed
**Fix:** Check current model names, use `gemini-2.0-flash` or `gemini-2.5-flash`

### 5. Unsupported Value Error (OpenAI)
```
BadRequestError: OpenAIException - Unsupported value
```
**Cause:** Using `reasoning_effort` with chat completions endpoint
**Fix:** Don't pass `reasoning_effort` to GPT models via chat completions

---

## Testing New Models

Before deploying a new model, test it with:

```python
import litellm

# Sync test
response = litellm.completion(
    model="provider/model-name",
    messages=[{"role": "user", "content": "Reply with just: 7"}],
    max_tokens=1000,  # High for GPT-5
    timeout=60
)
print(response.choices[0].message.content)

# Async test (important - behavior may differ!)
import asyncio
from litellm import acompletion

async def test():
    response = await acompletion(
        model="provider/model-name",
        messages=[{"role": "user", "content": "Reply with just: 7"}],
        max_tokens=1000,
        timeout=60
    )
    print(response.choices[0].message.content)

asyncio.run(test())
```

---

## Version Info

- **Date:** January 2026
- **litellm version:** Check with `pip show litellm`
- **Models tested:** gpt-5-nano, claude-sonnet-4-5, gemini-2.5-flash

---

## Quick Reference

```python
# Working rater configuration
async def rate_sentence(model_id, prompt):
    kwargs = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "timeout": 120,
        "max_tokens": 10
    }

    # GPT-5 needs more tokens
    if "gpt-5" in model_id.lower():
        kwargs["max_tokens"] = 1000

    # Gemini 2.5+ needs reasoning disabled
    if "gemini-2.5" in model_id.lower() or "gemini-3" in model_id.lower():
        kwargs["reasoning_effort"] = "none"

    response = await acompletion(**kwargs)
    return response.choices[0].message.content
```
