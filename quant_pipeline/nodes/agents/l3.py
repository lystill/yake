"""
L3 Agent Node — 竞争格局 & 卷度诊断
=====================================
Independent LLM call: analyzes competition intensity, ASP trends,
market share dynamics, and governance red flags.
Runs in parallel with L1 and L2 agents via LangGraph Send fan-out.
"""

import json
import litellm
from state import PipelineState
from prompts.l3_agent import L3_AGENT_PROMPT
from nodes.node3_reflection import REFLECTION_PROMPT_INJECT
from utils.config import get_model


def _build_l3_snapshot(state: PipelineState) -> str:
    """Build snapshot focused on competitive dynamics."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"行业分类: {industry}",
        f"当前股价: {state.get('current_price', 'N/A')} 元",
        "",
        "=== 财务基本面 ===",
        f"营业收入 (近3年): {fin.revenue}",
        f"归母净利润 (近3年): {fin.net_profit_parent}",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        f"固定资产/总资产: {fin.fa_to_ta:.1%}",
        "",
        "=== 业务板块竞争数据 ===",
    ]

    for seg in segments:
        lines.append(
            f"- {seg.name}: "
            f"收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, "
            f"增速 {seg.yoy_growth:+.1f}%, "
            f"利润 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"分支: {seg.branch or '未标注'}"
        )

    lines.append("")
    lines.append("=== ASP 与份额判断线索 ===")
    if len(fin.revenue) >= 3 and len(segments) >= 1:
        # Revenue per unit analysis — use segment growth vs industry
        for seg in segments:
            margin_clue = ""
            if hasattr(seg, 'gross_margin') and seg.gross_margin > 0:
                if seg.gross_margin > 50:
                    margin_clue = "高毛利率 → 产品有技术壁垒/定价权，ASP 抗跌"
                elif seg.gross_margin < 15:
                    margin_clue = "低毛利率 → 大宗/周期品特征，ASP 随行就市"
                elif seg.gross_margin < 30:
                    margin_clue = "中等毛利率 → 工业品，面临温和竞争"
            lines.append(f"  {seg.name}: 毛利率{seg.gross_margin:.1f}% → {margin_clue}")

    # Research context for competitive intelligence
    research = state.get("research_context", "")
    if research:
        lines.append("")
        lines.append("=== 行业研报/竞争情报 ===")
        lines.append(research[:2000])

    return "\n".join(lines)


def agent_l3(state: PipelineState) -> PipelineState:
    """L3 Agent: competitive analysis independently."""
    snapshot = _build_l3_snapshot(state)

    # Read upstream agent outputs for context
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})

    profit_driver = l1.get("profit_driver", state.get("profit_driver", "涨价驱动"))
    l1_reasoning = l1.get("reasoning", "未获取L1分析")
    penetration_stage = l2.get("penetration_stage", state.get("penetration_stage", ""))
    heavy_asset = l2.get("heavy_asset", state.get("fixed_asset_heavy", False))
    suggested_framework = l2.get("suggested_framework", "")

    prompt = L3_AGENT_PROMPT.format(
        company_data=snapshot,
        profit_driver=profit_driver,
        l1_reasoning=l1_reasoning,
        penetration_stage=penetration_stage or "不适用（非放量驱动）",
        heavy_asset=str(heavy_asset),
        suggested_framework=suggested_framework or "待定",
    )

    # Inject reflection directive on loop-back
    if state.get("reflection_round", 0) > 0:
        prev = state.get("l3_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视竞争格局】\n"
            f"上一轮你的判断是: ASP={prev.get('asp_trend', '?')}, "
            f"份额={prev.get('market_share_trend', '?')}, PEG调整={prev.get('peg_adjustment', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的竞争因子。\n\n"
        )
        prompt = REFLECTION_PROMPT_INJECT + reexamine + prompt

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        state["l3_output"] = json.loads(content)
    except Exception as e:
        state["error"] = f"L3 Agent failed: {e}"
        state["l3_output"] = {
            "asp_trend": "稳定",
            "market_share_trend": "稳定",
            "peg_adjustment": 1.0,
            "competition_verdict": f"LLM调用失败: {e}",
            "key_competitors": [],
            "supply_side_note": "",
            "governance_flags": [],
        }

    return {"l3_output": state.get("l3_output", {})}
