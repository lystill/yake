"""
SOTP Parallel Dispatch Node — Segment Valuation Concurrent Execution
=====================================================================
Runs N segment agent LLM calls concurrently within a single node
using ThreadPoolExecutor. All segment valuations are collected and
written to state['segment_valuations'].
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import litellm
from state import PipelineState, BusinessSegment
from prompts.segment_agent import SEGMENT_AGENT_PROMPT
from utils.config import get_model


def _call_segment(state: PipelineState, seg: BusinessSegment, idx: int) -> dict:
    """Execute one segment agent LLM call."""
    industry = state.get("industry", "")
    research = state.get("research_context", "")
    industry_context = f"公司所在行业: {industry}"
    if research:
        industry_context += f"\n行业研报摘要: {research[:800]}"
    industry_context += f"\n板块所属SOP分支: {seg.branch or '未标注'}"

    prompt = SEGMENT_AGENT_PROMPT.format(
        segment_name=seg.name,
        segment_branch=seg.branch or "待判断",
        profit_contribution=seg.profit_contribution,
        revenue_share=seg.revenue_share,
        gross_margin=seg.gross_margin,
        yoy_growth=seg.yoy_growth,
        profit_driver=state.get("profit_driver", ""),
        penetration_stage=state.get("penetration_stage", ""),
        asp_trend=state.get("competition_asp_trend", ""),
        market_share_trend=state.get("market_share_trend", ""),
        competition_verdict=state.get("competition_verdict", ""),
        industry_context=industry_context,
    )

    try:
        response = litellm.completion(
            model=get_model(), messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        result = json.loads(content)
        result["segment_name"] = seg.name
        result["segment_profit"] = seg.profit_contribution
        result["segment_branch"] = seg.branch
        return result
    except Exception as e:
        return {
            "segment_name": seg.name,
            "segment_profit": seg.profit_contribution,
            "segment_branch": seg.branch,
            "error": str(e),
            "comparable_pe": 15.0,
            "valuation": seg.profit_contribution * 15.0,
            "reasoning": f"LLM调用失败，使用默认PE=15x: {e}",
        }


def dispatch_parallel_sotp(state: PipelineState) -> PipelineState:
    """
    Run N segment agents concurrently using ThreadPoolExecutor.

    Each segment gets its own LLM call. Total wall-clock time ≈ max(segment_times).
    Results are collected into state['segment_valuations'].
    """
    segments = state.get("segments", [])
    if len(segments) < 2:
        state["segment_valuations"] = []
        return state

    print(f"  ⚡ [SOTP Parallel] 启动 {len(segments)} 个板块并行估值...")

    with ThreadPoolExecutor(max_workers=len(segments)) as executor:
        futures = {
            executor.submit(_call_segment, state, seg, i): (i, seg.name)
            for i, seg in enumerate(segments)
        }

        results = [None] * len(segments)
        for future in as_completed(futures):
            idx, name = futures[future]
            try:
                result = future.result()
                results[idx] = result
                pe = result.get("comparable_pe", 0)
                val = result.get("valuation", 0)
                print(f"  ✓ [Seg {idx}] {name}: PE={pe:.0f}x, 估值={val:.0f}亿")
            except Exception as e:
                results[idx] = {
                    "segment_name": name, "error": str(e),
                    "comparable_pe": 15.0,
                    "valuation": segments[idx].profit_contribution * 15.0,
                }
                print(f"  ✗ [Seg {idx}] {name}: 执行失败 - {e}")

    state["segment_valuations"] = [r for r in results if r is not None]
    return state
