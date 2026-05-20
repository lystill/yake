"""
Segment Agent Node — 单业务板块估值
=====================================
Independent LLM call: values ONE business segment.
Used in SOTP fan-out: N segments = N parallel Segment Agent calls.
"""

import json
import litellm
from state import PipelineState, BusinessSegment
from prompts.segment_agent import SEGMENT_AGENT_PROMPT
from utils.config import get_model


def _build_industry_context(state: PipelineState, seg: BusinessSegment) -> str:
    """Build minimal industry context for this specific segment."""
    industry = state.get("industry", "")
    research = state.get("research_context", "")

    parts = [f"公司所在行业: {industry}"]
    if research:
        parts.append(f"行业研报摘要: {research[:800]}")
    parts.append(f"板块所属SOP分支: {seg.branch or '未标注'}")
    return "\n".join(parts)


def agent_segment(state: PipelineState) -> PipelineState:
    """
    Segment Agent: value a single business segment.

    Expects a state override key 'segment_index' to know which segment to value.
    This is set via LangGraph Send with partial state updates.
    """
    segments = state.get("segments", [])
    idx = state.get("_segment_index", 0)

    if idx >= len(segments):
        vals = state.get("segment_valuations", [])
        vals.append({
            "segment_name": f"unknown_{idx}",
            "error": "segment index out of range",
            "comparable_pe": 0,
            "valuation": 0,
        })
        return {"segment_valuations": vals}

    seg: BusinessSegment = segments[idx]
    industry_context = _build_industry_context(state, seg)

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
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=400,
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
    except Exception as e:
        result = {
            "segment_name": seg.name,
            "segment_profit": seg.profit_contribution,
            "segment_branch": seg.branch,
            "error": str(e),
            "comparable_pe": 15.0,
            "valuation": seg.profit_contribution * 15.0,
            "reasoning": f"LLM调用失败，使用默认PE=15x: {e}",
        }

    valuations = state.get("segment_valuations", [])
    valuations.append(result)

    return {"segment_valuations": valuations}
