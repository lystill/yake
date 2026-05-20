"""
Research report fetcher.
Priority: local research_reports/[code].txt → web search fallback.
"""

import os
import json

RESEARCH_DIR = os.path.join(os.path.dirname(__file__), "..", "research_reports")


def _web_search_summary(stock_name: str, stock_code: str) -> str:
    """Search the web for recent industry research on the stock."""
    queries = [
        f"{stock_name} {stock_code} 券商研报 2025 2026 行业分析",
        f"{stock_name} 行业竞争格局 产能 ASP 趋势",
    ]

    results = []
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            for query in queries:
                for r in ddgs.text(query, max_results=5):
                    snippet = r.get("body", "") or r.get("snippet", "")
                    title = r.get("title", "")
                    if snippet:
                        results.append(f"[{title}] {snippet}")
                    if len(results) >= 8:
                        break
                if len(results) >= 8:
                    break
    except ImportError:
        return "（Web Search 不可用：duckduckgo_search 未安装）"
    except Exception as e:
        return f"（Web Search 失败：{e}）"

    if not results:
        return "（未找到相关研报摘要）"

    summary = "\n".join(results[:8])
    return f"【Web Search 行研摘要 — {stock_name} ({stock_code})】\n{summary}"


def fetch_research_report(
    stock_code: str,
    stock_name: str = "",
) -> str:
    """
    Fetch research context text for the given stock.

    Returns empty string if no content found.
    """
    # ——— 1. Check local file ———
    local_path = os.path.join(RESEARCH_DIR, f"{stock_code}.txt")
    if os.path.exists(local_path):
        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
        # File exists but is empty → fall through to web search

    # ——— 2. Web search fallback ———
    return _web_search_summary(stock_name, stock_code)
