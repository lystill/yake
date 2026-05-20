"""
Red Team Agent Node — 空头杠精对抗性分析
==========================================
Runs BEFORE the L1/L2/L3 re-analysis in the reflection loop.
Outputs bearish signals that get injected into the reflection prompt,
forcing forward agents to confront worst-case scenarios.

This prevents the "consensus drift" where multiple reflection rounds
cause agents to converge on compromise valuations.
"""

import re
import json
import litellm
from state import PipelineState
from prompts.red_team import (
    RED_TEAM_PROMPT,
    RED_TEAM_CAPEX_PROMPT,
    RED_TEAM_COMPETITION_PROMPT,
    RED_TEAM_STRATEGIC_GROWTH_PROMPT,
    RED_TEAM_PHARMA_PROMPT,
)
from utils.config import get_model


# ═══════════════════════════════════════════════════════════
#  Robust JSON cleaning (handles LLM malformed output)
# ═══════════════════════════════════════════════════════════

def clean_and_parse_json(raw_text: str) -> dict:
    """Extract and parse JSON from LLM output with aggressive repair."""
    if not raw_text or not raw_text.strip():
        return _red_team_fallback("空响应")

    # Strip markdown code fences
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3].strip()

    # Extract JSON block — find the outermost { ... }
    match = re.search(r'\{.*\}', text, re.DOTALL)
    json_str = match.group(0) if match else text

    # Fix unescaped newlines inside string values
    json_str = json_str.replace('\r', '')

    # Try raw parse first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Repair 1: collapse multi-line string values (common LLM failure)
    repaired = _repair_multiline_strings(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair 2: truncate trailing garbage
    repaired = _repair_trailing_garbage(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Repair 3: brute-force brace/suffix completion
    repaired = _repair_brace_completion(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Ultimate fallback
    print(f"  ⚠ [Red Team] JSON 全部修复策略失败，使用兜底")
    return _red_team_fallback(f"JSON解析失败，原始片段: {json_str[:200]}")


def _repair_multiline_strings(json_str: str) -> str:
    """Collapse literal newlines inside JSON string values (not structural)."""
    # Replace newlines within quoted strings — naive but effective for LLM output
    result = []
    in_string = False
    escape_next = False
    for ch in json_str:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        result.append(ch)
    return ''.join(result)


def _repair_trailing_garbage(json_str: str) -> str:
    """Find the last valid closing brace/bracket and truncate."""
    # Walk backwards to find balanced closing
    depth = 0
    last_valid = len(json_str)
    for i in range(len(json_str) - 1, -1, -1):
        ch = json_str[i]
        if ch == '}':
            depth += 1
        elif ch == '{':
            depth -= 1
            if depth == 0:
                last_valid = i
    # Try truncating at last_valid
    candidate = json_str[:last_valid] + '}'
    return candidate


def _repair_brace_completion(json_str: str) -> str:
    """Brute-force: pad missing closing braces/ brackets/ quotes."""
    s = json_str.strip()
    # Count unclosed braces/brackets
    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')

    # Remove trailing comma before closing (common LLM artifact)
    s = re.sub(r',\s*$', '', s)

    # Close open structures
    if open_brackets > 0:
        s += ']' * open_brackets
    if open_braces > 0:
        s += '}' * open_braces

    # Fix unclosed string values (odd number of quotes)
    if s.count('"') % 2 != 0:
        # Check if last value is truncated — add closing quote
        last_quote = s.rfind('"')
        last_colon = s.rfind(':')
        if last_colon > last_quote:
            s += '"'

    return s


def _red_team_fallback(reason: str) -> dict:
    """Deterministic fallback when all JSON repair strategies fail."""
    return {
        "bearish_signals": [f"红队JSON解析失败: {reason}"],
        "severity": "high",
        "reasoning": "红队文本解析全面崩溃，强制转入人工审计",
        "most_dangerous_signal": "无法自动分析——模型输出不可解析",
        "suggested_adjustments": {
            "peg_haircut_recommendation": "PEG建议强制打7折以计入不确定性",
            "valuation_floor_risk": "极端悲观估值下限: 净资产打8折",
        },
    }


# ═══════════════════════════════════════════════════════════
#  Company data snapshot
# ═══════════════════════════════════════════════════════════


def _build_red_team_snapshot(state: PipelineState) -> str:
    """Build a data-rich snapshot for the red team to attack."""
    fin = state["financials"]
    segments = state.get("segments", [])
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})
    l3 = state.get("l3_output", {})
    valuation = state.get("valuation")
    governance = state.get("governance_flags", [])

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"行业: {state.get('industry', '')}",
        f"当前股价: {state.get('current_price', 0):.2f} 元",
        f"总股本: {state.get('total_shares', 0):.2f} 亿股",
        "",
        "=== 前向 Agent 的判断（请无情攻击）===",
        f"L1 利润驱动力: {l1.get('profit_driver', '?')} (置信度 {l1.get('confidence', 0):.0%})",
        f"L2 渗透率: {l2.get('penetration_stage', '?')}, 重资产: {l2.get('heavy_asset', '?')}",
        f"L3 ASP趋势: {l3.get('asp_trend', '?')}, 份额: {l3.get('market_share_trend', '?')}, PEG调整: {l3.get('peg_adjustment', '?')}",
        "",
        "=== 财务数据 ===",
        f"近3年营收 (亿): {fin.revenue}",
        f"近3年归母净利润 (亿): {fin.net_profit_parent}",
        f"近3年经营现金流 (亿): {fin.op_cash_flow}",
        f"净现比3Y均值: {fin.cash_ratio_3y_avg:.2f}",
        f"固定资产: {fin.fixed_assets:.1f}亿",
        f"在建工程: {fin.construction_in_progress:.1f}亿",
        f"总资产: {fin.total_assets:.1f}亿",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        f"固定资产/总资产: {fin.fa_to_ta:.1%}",
        f"调整后(固+在)/总资产: {fin.adjusted_fa_to_ta:.1%}",
        f"资产周转率 (ATO): {fin.ato:.3f}, 趋势: {fin.ato_trend}",
        "",
        "=== 业务板块 ===",
    ]
    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入{seg.revenue:.1f}亿({seg.revenue_share:.1f}%), "
            f"毛利率{seg.gross_margin:.1f}%, 增速{seg.yoy_growth:+.1f}%, "
            f"利润{seg.profit_contribution:.1f}亿({seg.profit_share:.1f}%), "
            f"分支: {seg.branch or '未标注'}"
        )

    if valuation:
        lines.extend([
            "",
            "=== 当前估值结论（请挑战）===",
            f"估值框架: {valuation.framework}",
            f"基准股价: {valuation.base_case_price:.2f} 元",
            f"乐观/悲观: {valuation.bull_case_price:.2f} / {valuation.bear_case_price:.2f} 元",
        ])

    if governance:
        lines.append("")
        lines.append("=== 治理红旗 ===")
        for g in governance:
            lines.append(f"- {g}")

    return "\n".join(lines)


def node_red_team(state: PipelineState) -> PipelineState:
    """
    Run the red team (short-seller) agent to find bearish signals.

    Only runs during reflection loops (reflection_round > 0).
    Output is stored in state['red_team_output'] and injected into
    the next round of L1/L2/L3 analysis via REFLECTION_PROMPT_INJECT.
    """
    round_num = state.get("reflection_round", 0)

    # Only run on reflection loop-back (round >= 1 means we're re-analyzing)
    if round_num < 1:
        state["red_team_output"] = {}
        return state

    snapshot = _build_red_team_snapshot(state)
    prompt = RED_TEAM_PROMPT.format(company_data=snapshot)

    print(f"  🩸 [Red Team] 启动空头杠精对抗性分析 (反思轮次 {round_num})...")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        result = clean_and_parse_json(content)
        state["red_team_output"] = result

        severity = result.get("severity", "?")
        signals = result.get("bearish_signals", [])
        print(f"  ✓ [Red Team] 严重级别: {severity} | 发现 {len(signals)} 个负面信号")
        if signals:
            print(f"    最危险信号: {result.get('most_dangerous_signal', 'N/A')[:100]}")

    except Exception as e:
        print(f"  ✗ [Red Team] LLM调用失败: {e}")
        state["red_team_output"] = _red_team_fallback(f"LLM调用异常: {e}")

    return state


# ═══════════════════════════════════════════════════════════════
#  v3.2 专项红队节点 — CAPEX 供给侧定向审查
# ═══════════════════════════════════════════════════════════════

def node_red_team_capex(state: PipelineState) -> PipelineState:
    """
    CAPEX-focused red team. Triggered when intelligent_deviation_router
    detects Value Trap from supply-side overcapacity risk.

    Reads state['red_team_focus'] for the specific accusation to investigate.
    Output stored in state['red_team_output'].
    """
    round_num = state.get("reflection_round", 0)
    focus = state.get("red_team_focus", "Value Trap: 供给侧产能风险")

    snapshot = _build_red_team_snapshot(state)
    prompt = RED_TEAM_CAPEX_PROMPT.format(
        red_team_focus=focus,
        company_data=snapshot,
    )

    print(f"  🏭 [Red Team CAPEX] 供给侧定向审查 (轮次 {round_num})")
    print(f"    焦点: {focus[:120]}")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        result = clean_and_parse_json(content)
        state["red_team_output"] = result

        severity = result.get("severity", "?")
        signals = result.get("bearish_signals", [])
        print(f"  ✓ [Red Team CAPEX] 严重级别: {severity} | 发现 {len(signals)} 个CAPEX负面信号")
        if signals:
            print(f"    最危险信号: {result.get('most_dangerous_signal', 'N/A')[:100]}")
        cycle_pos = result.get("capex_cycle_position", "?")
        depr = result.get("depreciation_impact_3y", "?")
        print(f"    周期位置: {cycle_pos} | 折旧冲击: {depr}")

    except Exception as e:
        print(f"  ✗ [Red Team CAPEX] LLM调用失败: {e}")
        state["red_team_output"] = _red_team_fallback(f"CAPEX专项审查异常: {e}")

    return state


# ═══════════════════════════════════════════════════════════════
#  v3.2 专项红队节点 — 竞争格局/ASP 定向审查
# ═══════════════════════════════════════════════════════════════

def node_red_team_competition(state: PipelineState) -> PipelineState:
    """
    Competition/ASP-focused red team. Triggered when intelligent_deviation_router
    detects Value Trap from price war / market share erosion.

    Reads state['red_team_focus'] for the specific accusation to investigate.
    Output stored in state['red_team_output'].
    """
    round_num = state.get("reflection_round", 0)
    focus = state.get("red_team_focus", "Value Trap: 恶性价格战风险")

    snapshot = _build_red_team_snapshot(state)
    prompt = RED_TEAM_COMPETITION_PROMPT.format(
        red_team_focus=focus,
        company_data=snapshot,
    )

    print(f"  ⚔  [Red Team COMP] 竞争格局定向审查 (轮次 {round_num})")
    print(f"    焦点: {focus[:120]}")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        result = clean_and_parse_json(content)
        state["red_team_output"] = result

        severity = result.get("severity", "?")
        signals = result.get("bearish_signals", [])
        print(f"  ✓ [Red Team COMP] 严重级别: {severity} | 发现 {len(signals)} 个竞争负面信号")
        if signals:
            print(f"    最危险信号: {result.get('most_dangerous_signal', 'N/A')[:100]}")
        price_war = result.get("price_war_assessment", "?")
        moat = result.get("moat_durability", "?")
        print(f"    价格战性质: {price_war} | 护城河: {moat}")

    except Exception as e:
        print(f"  ✗ [Red Team COMP] LLM调用失败: {e}")
        state["red_team_output"] = _red_team_fallback(f"竞争专项审查异常: {e}")

    return state


# ═══════════════════════════════════════════════════════════════
#  v3.2 专项红队节点 — 战略成长股深水区审查
# ═══════════════════════════════════════════════════════════════

def node_red_team_strategic_growth(state: PipelineState) -> PipelineState:
    """
    Strategic growth-focused red team. Targets deep tech risks (chip fabrication,
    giant competition, order visibility, tech route substitution) instead of
    superficial cash-flow attacks.

    ⛔ Permanently banned attack vectors:
      - "No profit / negative net income"
      - "Negative operating cash flow"
      - "High PS multiple (>30x)"
      - "Low gross margin during ramp-up"
      - "Widening losses"

    Read into state['red_team_focus'] for the specific accusation.
    Output stored in state['red_team_output'].
    """
    round_num = state.get("reflection_round", 0)
    focus = state.get("red_team_focus", "战略成长股深水区审查：技术卡脖子/巨头降维/订单能见度")

    snapshot = _build_red_team_snapshot(state)
    prompt = RED_TEAM_STRATEGIC_GROWTH_PROMPT.format(
        red_team_focus=focus,
        company_data=snapshot,
    )

    print(f"  🚀 [Red Team STRATEGIC] 战略成长股深水区审查 (轮次 {round_num})")
    print(f"    焦点: {focus[:120]}")
    print(f"    ⛔ 禁攻维度: 无利润/负现金流/高PS/低毛利/亏损扩大")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        result = clean_and_parse_json(content)
        state["red_team_output"] = result

        severity = result.get("severity", "?")
        signals = result.get("bearish_signals", [])
        print(f"  ✓ [Red Team STRATEGIC] 严重级别: {severity} | 发现 {len(signals)} 个深水区风险")
        if signals:
            print(f"    最危险信号: {result.get('most_dangerous_signal', 'N/A')[:100]}")
        tech_risk = result.get("tech_bottleneck_risk", "?")
        giant_risk = result.get("giant_competition_risk", "?")
        order_vis = result.get("order_visibility", "?")
        print(f"    卡脖子: {tech_risk} | 巨头威胁: {giant_risk} | 订单能见度: {order_vis}")

    except Exception as e:
        print(f"  ✗ [Red Team STRATEGIC] LLM调用失败: {e}")
        state["red_team_output"] = _red_team_fallback(f"战略成长股深水区审查异常: {e}")

    return state


# ═══════════════════════════════════════════════════════════════
#  v3.3 专项红队节点 — 创新药/Biotech 深水区审查
# ═══════════════════════════════════════════════════════════════

def node_red_team_pharma(state: PipelineState) -> PipelineState:
    """
    Pharma/biotech-focused red team. Targets clinical trial failure, FDA rejection,
    NRDL pricing cuts, patent litigation, and geopolitical risks instead of
    superficial financial attacks (R&D spend, losses, negative cash flow).

    ⛔ Permanently banned attack vectors:
      - High R&D expense / R&D ratio
      - Negative net income / widening losses
      - Negative operating cash flow
      - Negative PE / uncomputable PE
      - Gross margin volatility
      - High selling expense ratio

    Reads state['red_team_focus'] for the specific accusation.
    Output stored in state['red_team_output'].
    """
    round_num = state.get("reflection_round", 0)
    focus = state.get("red_team_focus", "创新药深水区审查：临床Ⅲ期失败/FDA拒批/医保砍单/专利诉讼")

    snapshot = _build_red_team_snapshot(state)
    prompt = RED_TEAM_PHARMA_PROMPT.format(
        red_team_focus=focus,
        company_data=snapshot,
    )

    print(f"  💊 [Red Team PHARMA] 创新药/Biotech 深水区审查 (轮次 {round_num})")
    print(f"    焦点: {focus[:120]}")
    print(f"    ⛔ 禁攻维度: 研发费用/亏损/负现金流/PE不可计算/毛利率/销售费用率")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=600,
        )
        content = response.choices[0].message.content.strip()
        result = clean_and_parse_json(content)
        state["red_team_output"] = result

        severity = result.get("severity", "?")
        signals = result.get("bearish_signals", [])
        print(f"  ✓ [Red Team PHARMA] 严重级别: {severity} | 发现 {len(signals)} 个深水区风险")
        if signals:
            print(f"    最危险信号: {result.get('most_dangerous_signal', 'N/A')[:100]}")
        pipeline_risk = result.get("core_pipeline_risk", "?")
        regulatory_risk = result.get("regulatory_risk", "?")
        nrdl_risk = result.get("nrdl_pricing_risk", "?")
        patent_risk = result.get("patent_geopolitical_risk", "?")
        runway = result.get("cash_runway_months", "?")
        print(f"    管线风险: {pipeline_risk} | 监管: {regulatory_risk} | NRDL: {nrdl_risk} | 专利/地缘: {patent_risk} | 现金跑道: {runway}")

    except Exception as e:
        print(f"  ✗ [Red Team PHARMA] LLM调用失败: {e}")
        state["red_team_output"] = _red_team_fallback(f"创新药深水区审查异常: {e}")

    return state
