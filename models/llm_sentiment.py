"""
LLM Sentiment Engine — uses OpenAI API to score analyst text.
Identifies narratives that technical indicators can't capture.
"""

import json
from datetime import datetime, timezone
from loguru import logger

from database.init_db import get_connection
from utils.config import get_api_key


class LLMSentiment:
    """Scores text sentiment using LLM API (OpenAI-compatible)."""

    SYSTEM_PROMPT = """You are a cryptocurrency market sentiment analyst.
Analyze the given text about Bitcoin and return a JSON object with:
- sentiment_score: float from -1 (extremely bearish) to +1 (extremely bullish)
- narrative_tags: list of key narratives detected (e.g., "halving", "ETF", "regulation", "macro_fear")
- confidence: float from 0 to 1
- summary: one-sentence summary of the sentiment

Only return valid JSON, nothing else."""

    def __init__(self):
        self.api_key = get_api_key("openai")

    def analyze(self, text: str, source: str = "manual") -> dict:
        """
        Analyze text sentiment using LLM.
        Returns sentiment score and narrative tags.
        """
        if not self.api_key:
            logger.warning("No OpenAI API key configured — skipping LLM sentiment")
            return {
                "sentiment_score": 0,
                "narrative_tags": [],
                "confidence": 0,
                "summary": "LLM analysis unavailable (no API key)",
            }

        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this text:\n\n{text[:2000]}"},
                ],
                temperature=0.1,
                max_tokens=300,
            )

            content = response.choices[0].message.content.strip()
            # Try to parse JSON
            result = json.loads(content)

            # Store in database
            self._store(result, text[:500], source)

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return {"sentiment_score": 0, "narrative_tags": [], "confidence": 0, "error": str(e)}
        except Exception as e:
            logger.error(f"LLM sentiment analysis failed: {e}")
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
                    "gpt-4o-mini",
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
