"""
Parallel Qualitative Dispatch Node — L1/L2/L3 Concurrent Execution
====================================================================
Runs L1, L2, L3 LLM calls concurrently within a single node using
ThreadPoolExecutor. All three outputs are collected and written to state.

This approach avoids LangGraph Send-based fan-out complexity while still
achieving true parallel LLM execution (3 calls in the time of 1).

Each agent call is timed and logged to state['agent_audit'] for full
audit trail reproducibility.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import time
import json
import litellm
from state import PipelineState, AgentAuditEntry
from prompts.l1_agent import L1_AGENT_PROMPT
from prompts.l2_agent import L2_AGENT_PROMPT
from prompts.l3_agent import L3_AGENT_PROMPT
from nodes.node3_reflection import REFLECTION_PROMPT_INJECT, build_red_team_injection
from utils.config import get_model


def _build_agent_snapshot(state: PipelineState, agent: str) -> str:
    """Build the relevant snapshot for each agent."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")
    macro_beta = state.get("macro_beta", "")
    research = state.get("research_context", "")

    if agent == "l1":
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
            lines.append(f"收入-利润同步性: {'同步' if max(rev_changes)-min(rev_changes) < 0.3 else '明显背离（典型周期/价格驱动特征）'}")
        if len(fin.revenue) >= 2:
            margin_changes = []
            for i in range(len(fin.revenue)):
                if fin.revenue[i] > 0:
                    margin_changes.append(fin.net_profit_parent[i] / fin.revenue[i] * 100)
            if margin_changes:
                lines.append(f"净利润率逐年: {[f'{m:.1f}%' for m in margin_changes]}")
                if max(margin_changes) - min(margin_changes) > 10:
                    lines.append("⚠ 净利率波动 >10个百分点 → 典型周期品/涨价驱动信号")

    elif agent == "l2":
        lines = [
            f"股票代码: {state['stock_code']}",
            f"股票名称: {state['stock_name']}",
            f"行业分类: {industry}",
            "",
            "=== 资产结构（补丁1: 含在建工程影子指标）===",
            f"固定资产: {fin.fixed_assets:.1f} 亿元",
            f"在建工程: {fin.construction_in_progress:.1f} 亿元",
            f"总资产: {fin.total_assets:.1f} 亿元",
            f"固定资产/总资产 (原始): {fin.fa_to_ta:.1%}",
            f"(固定资产+在建工程)/总资产 (调整后): {fin.adjusted_fa_to_ta:.1%}",
            f"[判定标准: 调整后比值>40% → 重资产 → PB-ROE; ≤40% → 轻资产 → PE/PEG]",
            f"⚠ 影子指标 — 若原始≤40%但调整后>40%，说明大量产能尚未转固，应视为重资产",
            f"资产负债率: {fin.debt_ratio:.1f}%",
            "",
            "=== 资产周转率（ATO 影子验证）===",
            f"资产周转率 (最新收入/总资产): {fin.ato:.3f}",
            f"ATO 趋势: {fin.ato_trend}",
            f"[若ATO持续恶化且调整后FA/TA>40%: 重资产置信度提升，降低PEG弹性]",
            "",
            "=== 增长指标 ===",
        ]
        if len(fin.revenue) >= 3:
            rev_cagr = (fin.revenue[2] / fin.revenue[0]) ** (1/2) - 1 if fin.revenue[0] > 0 else 0
            lines.append(f"3年收入 CAGR: {rev_cagr:+.1%}")
        if len(fin.net_profit_parent) >= 3:
            profit_cagr = (fin.net_profit_parent[2] / fin.net_profit_parent[0]) ** (1/2) - 1 if (fin.net_profit_parent[0] > 0 and fin.net_profit_parent[2] > 0) else 0
            lines.append(f"3年利润 CAGR: {profit_cagr:+.1%}")
        lines.extend([
            f"营业收入 (近3年): {fin.revenue}",
            f"归母净利润 (近3年): {fin.net_profit_parent}",
        ])

    elif agent == "l3":
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
        ]

    # Common: segment breakdown
    lines.append("")
    lines.append("=== 业务板块构成 ===")
    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%, "
            f"利润贡献 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"分支: {seg.branch or '未标注'}"
        )

    # Research context
    if research:
        lines.append("")
        lines.append("=== 行业研报/情报 ===")
        lines.append(research[:1500])

    return "\n".join(lines)


def _call_l1(state: PipelineState) -> dict:
    """Execute L1 agent LLM call."""
    snapshot = _build_agent_snapshot(state, "l1")
    prompt = L1_AGENT_PROMPT.format(company_data=snapshot)

    if state.get("reflection_round", 0) > 0:
        prev = state.get("l1_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视利润驱动力】\n"
            f"上一轮你的判断是: {prev.get('profit_driver', '?')} (置信度 {prev.get('confidence', 0):.0%})\n"
            f"请考虑市场定价是否揭示了模型忽略的因子。\n\n"
        )
        red_inject = build_red_team_injection(state)
        prompt = REFLECTION_PROMPT_INJECT + red_inject + reexamine + prompt

    try:
        response = litellm.completion(
            model=get_model(), messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
    except Exception as e:
        return {"profit_driver": "涨价驱动", "confidence": 0.0, "reasoning": f"LLM失败: {e}", "primary_segment": "", "price_vs_volume_clue": ""}


def _call_l2(state: PipelineState) -> dict:
    """Execute L2 agent LLM call."""
    snapshot = _build_agent_snapshot(state, "l2")
    l1 = state.get("l1_output", {})
    pd = l1.get("profit_driver", state.get("profit_driver", "涨价驱动"))
    l1_conf = l1.get("confidence", 0.5)
    l1_reasoning = l1.get("reasoning", "未获取L1分析")

    prompt = L2_AGENT_PROMPT.format(
        company_data=snapshot, profit_driver=pd,
        confidence=l1_conf, l1_reasoning=l1_reasoning,
    )

    if state.get("reflection_round", 0) > 0:
        prev = state.get("l2_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视生命周期和资产属性】\n"
            f"上一轮你的判断是: 渗透率={prev.get('penetration_stage', '?')}, "
            f"重资产={prev.get('heavy_asset', '?')}, 建议框架={prev.get('suggested_framework', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的因子。\n\n"
        )
        red_inject = build_red_team_injection(state)
        prompt = REFLECTION_PROMPT_INJECT + red_inject + reexamine + prompt

    try:
        response = litellm.completion(
            model=get_model(), messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
    except Exception as e:
        return {
            "penetration_stage": "", "current_rate_estimate": "",
            "heavy_asset": state["financials"].adjusted_fa_to_ta > 0.40,
            "fa_to_ta_check": f"调整后(固+在)/总={state['financials'].adjusted_fa_to_ta:.1%}",
            "ato_shadow": f"ATO={state['financials'].ato:.3f}, 趋势={state['financials'].ato_trend}",
            "suggested_framework": "PE/PEG弹性锚" if state["financials"].adjusted_fa_to_ta <= 0.40 else "PB-ROE底部锚",
            "reasoning": f"LLM失败，使用默认值: {e}",
        }


def _call_l3(state: PipelineState) -> dict:
    """Execute L3 agent LLM call."""
    snapshot = _build_agent_snapshot(state, "l3")
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})

    prompt = L3_AGENT_PROMPT.format(
        company_data=snapshot,
        profit_driver=l1.get("profit_driver", state.get("profit_driver", "涨价驱动")),
        l1_reasoning=l1.get("reasoning", "未获取L1分析"),
        penetration_stage=l2.get("penetration_stage", state.get("penetration_stage", "")) or "不适用（非放量驱动）",
        heavy_asset=str(l2.get("heavy_asset", state.get("fixed_asset_heavy", False))),
        suggested_framework=l2.get("suggested_framework", "") or "待定",
    )

    if state.get("reflection_round", 0) > 0:
        prev = state.get("l3_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视竞争格局】\n"
            f"上一轮你的判断是: ASP={prev.get('asp_trend', '?')}, "
            f"份额={prev.get('market_share_trend', '?')}, PEG调整={prev.get('peg_adjustment', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的竞争因子。\n\n"
        )
        red_inject = build_red_team_injection(state)
        prompt = REFLECTION_PROMPT_INJECT + red_inject + reexamine + prompt

    try:
        response = litellm.completion(
            model=get_model(), messages=[{"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        return json.loads(content)
    except Exception as e:
        return {
            "asp_trend": "稳定", "market_share_trend": "稳定", "peg_adjustment": 1.0,
            "competition_verdict": f"LLM失败: {e}", "key_competitors": [],
            "supply_side_note": "", "governance_flags": [],
        }


def dispatch_parallel_qualitative(state: PipelineState) -> PipelineState:
    """
    Run L1, L2, L3 agents concurrently using ThreadPoolExecutor.

    All three LLM calls fire at the same time. Total wall-clock time ≈ max(L1_time, L2_time, L3_time).
    Results are written to l1_output, l2_output, l3_output in state.
    Audit trail entries are captured per agent.
    """
    model = get_model()
    ts_start = datetime.datetime.now()
    print(f"  ⚡ [Parallel] 启动 L1/L2/L3 三 Agent 并行分析... (模型: {model})")

    audit = {}
    # Track per-agent start times
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_call_l1, state): "L1",
            executor.submit(_call_l2, state): "L2",
            executor.submit(_call_l3, state): "L3",
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            t_agent_start = time.perf_counter()
            fallback = False
            error_msg = ""

            try:
                result = future.result()
                latency = (time.perf_counter() - t_agent_start) * 1000
                # Check if result indicates fallback
                if "LLM失败" in str(result.get("reasoning", "")) or "LLM调用失败" in str(result.get("competition_verdict", "")) or "LLM失败" in str(result.get("competition_verdict", "")):
                    fallback = True
                    error_msg = str(result.get("reasoning", result.get("competition_verdict", "")))

                if agent_name == "L1":
                    state["l1_output"] = result
                    state["profit_driver"] = result.get("profit_driver", "涨价驱动")
                    print(f"  ✓ [L1 Agent] 利润驱动力: {result.get('profit_driver', '?')} (置信度 {result.get('confidence', 0):.0%}) | {latency:.0f}ms")
                elif agent_name == "L2":
                    state["l2_output"] = result
                    print(f"  ✓ [L2 Agent] 渗透率: {result.get('penetration_stage', '?') or 'N/A'} | 重资产: {result.get('heavy_asset', '?')} | {latency:.0f}ms")
                elif agent_name == "L3":
                    state["l3_output"] = result
                    print(f"  ✓ [L3 Agent] ASP: {result.get('asp_trend', '?')} | 份额: {result.get('market_share_trend', '?')} | PEG: {result.get('peg_adjustment', '?')} | {latency:.0f}ms")

                audit[agent_name] = {
                    "agent": agent_name,
                    "model": model,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "latency_ms": round(latency, 1),
                    "fallback": fallback,
                    "error": error_msg[:200] if error_msg else "",
                }

            except Exception as e:
                latency = (time.perf_counter() - t_agent_start) * 1000
                print(f"  ✗ [{agent_name} Agent] 执行失败 ({latency:.0f}ms): {e}")
                audit[agent_name] = {
                    "agent": agent_name,
                    "model": model,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "latency_ms": round(latency, 1),
                    "fallback": True,
                    "error": str(e)[:200],
                }
                # Set fallback outputs
                if agent_name == "L1":
                    state["l1_output"] = {"profit_driver": "涨价驱动", "confidence": 0.0, "reasoning": f"执行失败: {e}", "primary_segment": "", "price_vs_volume_clue": ""}
                elif agent_name == "L2":
                    adj = state["financials"].adjusted_fa_to_ta if state.get("financials") else 0.0
                    state["l2_output"] = {"penetration_stage": "", "current_rate_estimate": "", "heavy_asset": adj > 0.40, "fa_to_ta_check": f"调整后(固+在)/总={adj:.1%}", "ato_shadow": "", "suggested_framework": "PB-ROE底部锚" if adj > 0.40 else "PE/PEG弹性锚", "reasoning": f"执行失败: {e}"}
                elif agent_name == "L3":
                    state["l3_output"] = {"asp_trend": "稳定", "market_share_trend": "稳定", "peg_adjustment": 1.0, "competition_verdict": f"执行失败: {e}", "key_competitors": [], "supply_side_note": "", "governance_flags": []}

    total_latency = (time.perf_counter() - t0) * 1000
    audit["_total_wall_ms"] = round(total_latency, 1)
    audit["_parallel_efficiency"] = f"3 agents in {total_latency:.0f}ms (serial would be ~{sum(a.get('latency_ms', 0) for a in audit.values() if isinstance(a, dict) and 'latency_ms' in a):.0f}ms)"

    state["agent_audit"] = audit
    return state
