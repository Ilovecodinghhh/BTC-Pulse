"""
LLM Sentiment Engine — multi-provider support.
Supports DeepSeek (cheap), Ollama (free/local), and OpenAI (paid).
All use the OpenAI-compatible SDK interface.
"""

import json
from datetime import datetime, timezone
from loguru import logger

from database.init_db import get_connection
from utils.config import load_config, get_api_key


class LLMSentiment:
    """
    Scores text sentiment using LLM API.
    Provider-agnostic: works with DeepSeek, Ollama, or OpenAI.
    All three support the OpenAI SDK interface.
    """

    SYSTEM_PROMPT = """You are a cryptocurrency market sentiment analyst.
Analyze the given text about Bitcoin and return a JSON object with:
- sentiment_score: float from -1 (extremely bearish) to +1 (extremely bullish)
- narrative_tags: list of key narratives detected (e.g., "halving", "ETF", "regulation", "macro_fear")
- confidence: float from 0 to 1
- summary: one-sentence summary of the sentiment

Only return valid JSON, nothing else."""

    # Model mapping per provider
    PROVIDER_MODELS = {
        "deepseek": "deepseek-chat",
        "ollama": "llama3.1",          # Or whatever model is installed locally
        "openai": "gpt-4o-mini",
    }

    def __init__(self):
        cfg = load_config()
        ai_cfg = cfg.get("api_keys", {})

        self.provider = ai_cfg.get("ai_provider", "deepseek")
        self.api_key = ai_cfg.get("ai_api_key", "")
        self.base_url = ai_cfg.get("ai_base_url", "")
        self.model = self.PROVIDER_MODELS.get(self.provider, "deepseek-chat")

        # Ollama doesn't need an API key
        if self.provider == "ollama" and not self.api_key:
            self.api_key = "ollama"  # Placeholder — Ollama ignores this
        if self.provider == "ollama" and not self.base_url:
            self.base_url = "http://localhost:11434/v1"

        # DeepSeek default URL
        if self.provider == "deepseek" and not self.base_url:
            self.base_url = "https://api.deepseek.com"

        # Legacy: fall back to old openai key if ai_api_key is empty
        if not self.api_key and self.provider == "openai":
            self.api_key = get_api_key("openai")

    def _get_client(self):
        """Create OpenAI-compatible client for any provider."""
        import openai

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        return openai.OpenAI(**kwargs)

    def analyze(self, text: str, source: str = "manual") -> dict:
        """
        Analyze text sentiment using the configured LLM provider.
        Returns sentiment score and narrative tags.
        """
        if not self.api_key:
            logger.warning(f"No API key for {self.provider} — skipping LLM sentiment. "
                          f"Set ai_api_key in config.yaml (or use Ollama for free local inference)")
            return {
                "sentiment_score": 0,
                "narrative_tags": [],
                "confidence": 0,
                "summary": f"LLM analysis unavailable (no {self.provider} API key)",
            }

        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this text:\n\n{text[:2000]}"},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()

            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[-1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            result = json.loads(content)

            # Store in database
            self._store(result, text[:500], source)

            logger.info(f"LLM sentiment ({self.provider}/{self.model}): "
                       f"score={result.get('sentiment_score', 0):.2f}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"sentiment_score": 0, "narrative_tags": [], "confidence": 0, "error": str(e)}
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed ({self.provider}): {e}")
            return {"sentiment_score": 0, "narrative_tags": [], "confidence": 0, "error": str(e)}

    def _store(self, result: dict, text_snippet: str, source: str):
        """Store LLM sentiment result in database."""
        conn = get_connection()
        try:
            conn.execute(
                """INSERT INTO table_ai_sentiment
                   (timestamp, source, text_snippet, sentiment_score, narrative_tags, model_used)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                    source,
                    text_snippet,
                    result.get("sentiment_score", 0),
                    json.dumps(result.get("narrative_tags", [])),
                    f"{self.provider}/{self.model}",
                ),
            )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to store LLM sentiment: {e}")
        finally:
            conn.close()

    def get_latest_score(self) -> float:
        """Get the most recent AI sentiment score from database."""
        conn = get_connection()
        try:
            cursor = conn.execute(
                "SELECT sentiment_score FROM table_ai_sentiment ORDER BY timestamp DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return float(row[0]) if row else 0.0
        finally:
            conn.close()
