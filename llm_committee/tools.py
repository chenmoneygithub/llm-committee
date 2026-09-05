"""
Keyless web tools for committee members (no API keys required).

Provides ``web_search`` (DuckDuckGo via the ``ddgs`` package) and ``fetch_page``
(URL -> readable text). These are designed to be passed to ``dspy.ReAct`` so committee
members can gather evidence before forming opinions.

Both tools are intentionally defensive: network calls are wrapped so a transient
failure returns a short error string rather than raising, which keeps a ReAct loop
from crashing mid-trajectory.
"""

from __future__ import annotations

import re


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web and return the top results.

    Args:
        query: The search query.
        max_results: Number of results to return (default 5).

    Returns:
        A formatted string with title, URL, and snippet for each result, or an error
        message if the search fails.
    """
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=max_results))
    except Exception as e:  # noqa: BLE001 - tools must never crash the ReAct loop
        return f"[web_search error: {type(e).__name__}: {e}]"

    if not results:
        return "No results found."

    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        href = r.get("href", "")
        body = r.get("body", "")
        blocks.append(f"[{i}] {title}\n{href}\n{body}")
    return "\n\n".join(blocks)


def fetch_page(url: str, max_chars: int = 3000) -> str:
    """Fetch a web page and return its readable text content.

    Args:
        url: The URL to fetch.
        max_chars: Max characters of text to return (default 3000, to bound context).

    Returns:
        Cleaned text content of the page, truncated to max_chars, or an error message.
    """
    try:
        import requests

        resp = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (research committee bot)"},
        )
        resp.raise_for_status()
    except Exception as e:  # noqa: BLE001
        return f"[fetch_page error: {type(e).__name__}: {e}]"

    text = resp.text
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.decompose()
        text = soup.get_text(separator=" ")
    except Exception:  # noqa: BLE001 - fall back to raw text if bs4 unavailable
        pass

    # Collapse whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + " ...[truncated]"
    return text


def default_tools() -> list:
    """Return the default tool set for committee members."""
    return [web_search, fetch_page]
