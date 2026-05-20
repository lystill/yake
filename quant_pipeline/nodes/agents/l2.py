"""
L2 Agent Node — 生命周期 & 资产属性判定
==========================================
Independent LLM call: classifies penetration stage and asset intensity.
Runs in parallel with L1 and L3 agents via LangGraph Send fan-out.
Reads L1 output for context (profit driver determines what lifecycle metrics matter).
"""

import json
import litellm
from state import PipelineState
from prompts.l2_agent import L2_AGENT_PROMPT
from nodes.node3_reflection import REFLECTION_PROMPT_INJECT
from utils.config import get_model


def _build_l2_snapshot(state: PipelineState) -> str:
    """Build financial snapshot focused on lifecycle and asset signals."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"行业分类: {industry}",
        "",
        "=== 资产结构 ===",
        f"固定资产: {fin.fixed_assets:.1f} 亿元",
        f"总资产: {fin.total_assets:.1f} 亿元",
        f"固定资产/总资产: {fin.fa_to_ta:.1%}",
        f"[判定标准: >40% → 重资产 → PB-ROE; ≤40% → 轻资产 → PE/PEG]",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        "",
        "=== 增长指标 ===",
    ]

    if len(fin.revenue) >= 3:
        rev_cagr = (fin.revenue[2] / fin.revenue[0]) ** (1/2) - 1 if fin.revenue[0] > 0 else 0
        lines.append(f"3年收入 CAGR: {rev_cagr:+.1%}")
    if len(fin.net_profit_parent) >= 3:
        profit_cagr = (fin.net_profit_parent[2] / fin.net_profit_parent[0]) ** (1/2) - 1 if fin.net_profit_parent[0] > 0 else 0
        lines.append(f"3年利润 CAGR: {profit_cagr:+.1%}")

    lines.extend([
        f"营业收入 (近3年): {fin.revenue}",
        f"归母净利润 (近3年): {fin.net_profit_parent}",
        "",
        "=== 业务板块详情 ===",
    ])

    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%, "
            f"利润贡献 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"ROE: {seg.roe:.1f}%"
        )

    # Include industry research for penetration rate context
    research = state.get("research_context", "")
    if research:
        lines.append("")
        lines.append("=== 行业研报情报（可用于判断渗透率）===")
        lines.append(research[:1500])

    return "\n".join(lines)


def agent_l2(state: PipelineState) -> PipelineState:
    """L2 Agent: classify lifecycle stage and asset heaviness independently."""
    snapshot = _build_l2_snapshot(state)

    # Read L1 context if available
    l1 = state.get("l1_output", {})
    profit_driver = l1.get("profit_driver", state.get("profit_driver", "涨价驱动"))
    l1_confidence = l1.get("confidence", 0.5)
    l1_reasoning = l1.get("reasoning", "未获取L1分析")

    prompt = L2_AGENT_PROMPT.format(
        company_data=snapshot,
        profit_driver=profit_driver,
        confidence=l1_confidence,
        l1_reasoning=l1_reasoning,
    )

    # Inject reflection directive on loop-back
    if state.get("reflection_round", 0) > 0:
        prev = state.get("l2_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视生命周期和资产属性】\n"
            f"上一轮你的判断是: 渗透率={prev.get('penetration_stage', '?')}, "
            f"重资产={prev.get('heavy_asset', '?')}, 建议框架={prev.get('suggested_framework', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的因子。\n\n"
        )
        prompt = REFLECTION_PROMPT_INJECT + reexamine + prompt

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        state["l2_output"] = json.loads(content)
    except Exception as e:
        state["error"] = f"L2 Agent failed: {e}"
        state["l2_output"] = {
            "penetration_stage": "",
            "current_rate_estimate": "",
            "heavy_asset": state["financials"].fa_to_ta > 0.40,
            "fa_to_ta_check": f"固/总={state['financials'].fa_to_ta:.1%}",
            "suggested_framework": "PE/PEG弹性锚",
            "reasoning": f"LLM调用失败，使用默认值: {e}",
        }

    return {"l2_output": state.get("l2_output", {})}
