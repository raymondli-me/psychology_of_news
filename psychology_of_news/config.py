"""
Configuration for psychology_of_news analysis.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Config for a single LLM."""
    name: str
    model_id: str
    max_tokens: int = 10  # GPT-5-mini needs more, handled in rater


@dataclass
class Config:
    """
    Main configuration for news analysis.

    Example:
        config = Config(
            topic="Draymond Green trade",
            input_csv="/path/to/scraped_articles.csv",
            output_dir="/path/to/output"
        )
    """

    # What we're analyzing
    topic: str = "Draymond Green trade"

    # The rating prompt - {topic} and {text} are replaced
    prompt_template: str = """Rate this sentence on how strongly it implies {topic} will happen.

Sentence: "{text}"

Score from 1-10:
1 = No implication at all
5 = Neutral/ambiguous
10 = Strongly implies it will happen

Reply with ONLY a single number (1-10), nothing else."""

    # Input: CSV with columns [title, body_text, source, date, url]
    input_csv: Optional[str] = None

    # Output directory for all results
    output_dir: str = "./output"

    # Models - default triple-LLM setup
    models: list = field(default_factory=lambda: [
        ModelConfig(name="GPT", model_id="openai/gpt-5-mini"),
        ModelConfig(name="Claude", model_id="anthropic/claude-sonnet-4-5"),
        ModelConfig(name="Gemini", model_id="gemini/gemini-2.5-flash-preview-09-2025"),
    ])

    # Sentence extraction
    min_sentence_length: int = 30
    max_sentences: int = 200
    require_topic_mention: bool = True  # Only sentences mentioning topic keyword

    # Rating settings
    max_concurrent_per_model: int = 5
    batch_size: int = 10

    # Visualization settings
    create_umap: bool = True
    create_haam: bool = True
    create_figures: bool = True

    # API keys - if None, uses environment variables
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    def __post_init__(self):
        """Set up API keys and output directory."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
        if self.anthropic_api_key:
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
        if self.google_api_key:
            os.environ["GOOGLE_API_KEY"] = self.google_api_key

    def get_prompt(self, text: str) -> str:
        """Format prompt with topic and text."""
        return self.prompt_template.format(topic=self.topic, text=text)

    @property
    def topic_keyword(self) -> str:
        """Extract main keyword from topic for sentence filtering."""
        # "Draymond Green trade" -> "Draymond"
        return self.topic.split()[0]


# Preset configs for common use cases
def trade_analysis(player: str) -> Config:
    """Preset for player trade analysis."""
    return Config(
        topic=f"{player} trade",
        prompt_template=f"""Rate this sentence on how strongly it implies {player} will be traded.

Sentence: "{{text}}"

Score from 1-10:
1 = No trade implication at all
5 = Neutral/ambiguous
10 = Strongly implies trade will happen

Reply with ONLY a single number (1-10), nothing else."""
    )


def sentiment_analysis(subject: str) -> Config:
    """Preset for sentiment analysis."""
    return Config(
        topic=f"{subject} sentiment",
        prompt_template=f"""Rate the sentiment about {subject} in this sentence.

Sentence: "{{text}}"

Score from 1-10:
1 = Very negative
5 = Neutral
10 = Very positive

Reply with ONLY a single number (1-10), nothing else."""
    )
