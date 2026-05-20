"""
L1 Agent Node — 利润驱动力诊断
=================================
Independent LLM call: classifies the ROOT profit driver.
Runs in parallel with L2 and L3 agents via LangGraph Send fan-out.
"""

import json
import litellm
from state import PipelineState
from prompts.l1_agent import L1_AGENT_PROMPT
from nodes.node3_reflection import REFLECTION_PROMPT_INJECT
from utils.config import get_model


def _build_l1_snapshot(state: PipelineState) -> str:
    """Build financial snapshot focused on profit driver signals."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")
    macro_beta = state.get("macro_beta", "")

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"行业分类: {industry}",
        f"宏观环境: {macro_beta}",
        "",
        "=== 近三年核心财务（亿元）===",
        f"营业收入: {fin.revenue}",
        f"归母净利润: {fin.net_profit_parent}",
        f"经营现金流: {fin.op_cash_flow}",
        f"净现比3Y均值: {fin.cash_ratio_3y_avg:.2f}",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        "",
        "=== 利润弹性分析 ===",
    ]

    if len(fin.revenue) >= 2 and len(fin.net_profit_parent) >= 2:
        rev_changes = [fin.revenue[i] / fin.revenue[i-1] - 1 for i in range(1, len(fin.revenue))]
        profit_changes = [fin.net_profit_parent[i] / fin.net_profit_parent[i-1] - 1 for i in range(1, len(fin.net_profit_parent))]
        lines.append(f"收入年化变动: {[f'{c:.1%}' for c in rev_changes]}")
        lines.append(f"利润年化变动: {[f'{c:.1%}' for c in profit_changes]}")
        lines.append(f"收入-利润变动同步性: {'同步' if max(rev_changes)-min(rev_changes) < 0.3 else '明显背离（典型周期/价格驱动特征）'}")

    if len(fin.net_profit_parent) >= 2:
        margin_changes = []
        for i in range(len(fin.revenue)):
            if fin.revenue[i] > 0:
                m = fin.net_profit_parent[i] / fin.revenue[i] * 100
                margin_changes.append(m)
        if margin_changes:
            lines.append(f"净利润率逐年: {[f'{m:.1f}%' for m in margin_changes]}")
            if max(margin_changes) - min(margin_changes) > 10:
                lines.append("⚠ 净利率波动 >10个百分点 → 典型周期品/涨价驱动信号")

    lines.append("")
    lines.append("=== 业务板块构成 ===")
    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%, "
            f"利润贡献 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%)"
        )

    research = state.get("research_context", "")
    if research:
        lines.append("")
        lines.append("=== 行业研报情报 ===")
        lines.append(research[:1500])

    return "\n".join(lines)


def agent_l1(state: PipelineState) -> PipelineState:
    """L1 Agent: classify profit driver independently."""
    snapshot = _build_l1_snapshot(state)
    prompt = L1_AGENT_PROMPT.format(company_data=snapshot)

    # Inject reflection directive on loop-back
    if state.get("reflection_round", 0) > 0:
        prev = state.get("l1_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视利润驱动力】\n"
            f"上一轮你的判断是: {prev.get('profit_driver', '?')} (置信度 {prev.get('confidence', 0):.0%})\n"
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
        state["l1_output"] = json.loads(content)
    except Exception as e:
        state["error"] = f"L1 Agent failed: {e}"
        state["l1_output"] = {
            "profit_driver": "涨价驱动",
            "confidence": 0.0,
            "reasoning": f"LLM调用失败: {e}",
            "primary_segment": "",
            "price_vs_volume_clue": "",
        }

    # Also write directly for backward compatibility with downstream nodes
    state["profit_driver"] = state["l1_output"].get("profit_driver", "涨价驱动")

    return {"l1_output": state.get("l1_output", {}), "profit_driver": state.get("profit_driver", "涨价驱动")}
