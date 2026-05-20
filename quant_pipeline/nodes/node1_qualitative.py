"""
Node 1 — 行研定性 (L1/L2/L3 Classification)
================================================
Calls an LLM via LiteLLM to analyze financial data and assign SOP labels.
Forces structured JSON output only.
"""

import json
import litellm
from state import (
    PipelineState,
    FinancialMetrics,
    BusinessSegment,
    ProfitDriver,
    PenetrationStage,
)
from prompts.qualitative import L1_TO_L3_COMBINED_PROMPT
from nodes.node3_reflection import REFLECTION_PROMPT_INJECT
from utils.config import get_model


def _build_company_snapshot(state: PipelineState) -> str:
    """Serialize key financial and segment data into a compact text block for the LLM."""
    fin: FinancialMetrics = state["financials"]
    segments: list[BusinessSegment] = state.get("segments", [])

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"当前股价: {state.get('current_price', 'N/A')} 元",
        "",
        "=== 近三年财务数据 ===",
        f"营业收入 (亿): {fin.revenue}",
        f"归母净利润 (亿): {fin.net_profit_parent}",
        f"经营现金流 (亿): {fin.op_cash_flow}",
        f"净现比3Y均值: {fin.cash_ratio_3y_avg:.2f}",
        f"固定资产/总资产: {fin.fa_to_ta:.1%}",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        "",
        "=== 业务板块构成 ===",
    ]

    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%"
        )

    # Append research context if available
    research = state.get("research_context", "")
    if research:
        lines.append("")
        lines.append("=== 行业研报/情报摘要 ===")
        lines.append(research[:2000])  # Truncate to keep prompt manageable

    return "\n".join(lines)


def node1_qualitative(state: PipelineState) -> PipelineState:
    """
    Call LLM to classify:
      L1 — profit_driver (涨价驱动 / 放量驱动 / 稳定事件)
      L2 — penetration_stage (<5% / 5-30% / >30%)
      L3 — competition: ASP trend, market share, PEG haircut
    """
    snapshot = _build_company_snapshot(state)
    prompt = L1_TO_L3_COMBINED_PROMPT.format(company_data=snapshot)

    # Inject reflection prompt on loop-back rounds
    if state.get("reflection_round", 0) > 0:
        prompt = REFLECTION_PROMPT_INJECT + prompt

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()

        # Strip code fences if present
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        result = json.loads(content)
    except Exception as e:
        state["error"] = f"Node 1 LLM call failed: {e}"
        return state

    # ——— L1 ———
    l1 = result.get("L1", {})
    state["profit_driver"] = l1.get("profit_driver", "涨价驱动")

    # ——— L2 ———
    l2 = result.get("L2", {})
    state["penetration_stage"] = l2.get("penetration_stage", "")

    # ——— L3 ———
    l3 = result.get("L3", {})
    state["competition_asp_trend"] = l3.get("asp_trend", "稳定")
    state["market_share_trend"] = l3.get("market_share_trend", "稳定")
    state["peg_adjustment"] = float(l3.get("peg_haircut", 1.0))
    state["competition_verdict"] = l3.get("competition_verdict", "")

    # ——— Fixed-asset heaviness check ———
    state["fixed_asset_heavy"] = state["financials"].fa_to_ta > 0.40

    return state
