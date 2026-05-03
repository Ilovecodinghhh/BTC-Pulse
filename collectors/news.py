"""
News collector — fetches Bitcoin-related headlines for LLM sentiment analysis.
Uses free RSS feeds from CoinDesk and CoinTelegraph.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from loguru import logger

from utils.retry import safe_api_call


class NewsCollector:
    """
    Collects Bitcoin news headlines from free RSS feeds.
    Provides text input for LLM sentiment analysis.
    """

    RSS_FEEDS = {
        "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "cointelegraph": "https://cointelegraph.com/rss",
    }

    BTC_KEYWORDS = {"bitcoin", "btc", "crypto", "cryptocurrency", "halving", "etf", "mining"}

    @safe_api_call
    def _fetch_feed(self, url: str, timeout: int = 30) -> list[dict]:
        """Fetch and parse an RSS feed."""
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "BTC-Pulse/2.0 (news-collector)"
        })
        resp.raise_for_status()

        root = ET.fromstring(resp.content)
        items = []

        for item in root.iter("item"):
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            pub_date = item.findtext("pubDate", "")
            link = item.findtext("link", "")

            items.append({
                "title": title.strip(),
                "description": description.strip()[:500],
                "pub_date": pub_date,
                "link": link,
            })

        return items

    def collect(self, max_items: int = 20) -> list[dict]:
        """
        Fetch recent Bitcoin-related headlines from all RSS feeds.
        Filters for BTC-relevant articles.
        """
        all_items = []

        for source, url in self.RSS_FEEDS.items():
            try:
                items = self._fetch_feed(url)
                if items:
                    for item in items:
                        item["source"] = source
                    all_items.extend(items)
                    logger.info(f"Fetched {len(items)} items from {source}")
            except Exception as e:
                logger.warning(f"Failed to fetch {source}: {e}")

        # Filter for BTC-relevant headlines
        btc_items = []
        for item in all_items:
            text = (item["title"] + " " + item.get("description", "")).lower()
            if any(kw in text for kw in self.BTC_KEYWORDS):
                btc_items.append(item)

        btc_items = btc_items[:max_items]
        logger.info(f"Collected {len(btc_items)} BTC-related headlines")
        return btc_items

    def collect_as_text(self, max_items: int = 20) -> str:
        """
        Collect headlines and format as a single text block
        suitable for LLM sentiment analysis.
        """
        items = self.collect(max_items=max_items)
        if not items:
            return ""

        lines = [f"Bitcoin News Headlines ({datetime.now(timezone.utc).strftime('%Y-%m-%d')}):\n"]
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. [{item['source']}] {item['title']}")
            if item.get("description"):
                lines.append(f"   {item['description'][:200]}")

        return "\n".join(lines)
