"""
Serial Qualitative Chain Node — L1 → L2 → L3 Strict Sequential Execution
==========================================================================
Replaces dispatch_parallel.py (blind 3-agent parallel fan-out).

The new chain enforces the INVEST_SOP 3-layer decision tree as a strict
dependency graph:

  L1 (利润驱动力归因)
    │  写入: profit_driver, l1_output
    │
    ▼  L1 结论强制注入 L2 Prompt
  L2 (商业模式定向诊断)
    │  分支 A → 重资产属性 + CAPEX 状态
    │  分支 B → 渗透率阶段 + 在建工程影子指标
    │  分支 C1 → 现金流可预测性 + 分红可持续性
    │  分支 C2 → 催化剂时间线 + 概率评估
    │  写入: l2_output, heavy_asset, penetration_stage, capex_note, suggested_framework
    │
    ▼  L1+L2 完备标签注入 L3 Prompt
  L3 (竞争格局 + 最终框架落锚)
    │  7 框架选 1 — 不允许自由发散
    │  写入: l3_output, asp_trend, market_share_trend, peg_adjustment,
    │        capex_status, confirmed_framework, governance_flags

Each agent call is timed and logged to agent_audit for full reproducibility.
LLM failures trigger per-agent fallbacks that respect the serial dependency.
"""

import datetime
import time
import json
import litellm
from state import PipelineState
from prompts.l1_agent import L1_AGENT_PROMPT
from prompts.l2_agent import L2_AGENT_PROMPT
from prompts.l3_agent import L3_AGENT_PROMPT
from nodes.node3_reflection import build_red_team_injection
from utils.config import get_model

# ═══════════════════════════════════════════════════════════════
#  Reflection injection template
# ═══════════════════════════════════════════════════════════════

REFLECTION_PROMPT_INJECT = (
    "【⚠ 投资总监反思指令 — 上一轮估值与市场定价出现严重偏离】\n"
    "请以更高的怀疑精神重新审视你的分析。重点关注以下问题：\n"
    "1. 你的核心假设是否有被市场证伪的证据？\n"
    "2. 是否存在你忽略的负面因子（竞争恶化/技术替代/政策风险）？\n"
    "3. 如果你的判断是错的，最可能的错误来源是什么？\n\n"
)

# ═══════════════════════════════════════════════════════════════
#  Per-Agent Snapshot Builders
# ═══════════════════════════════════════════════════════════════

def _build_l1_snapshot(state: PipelineState) -> str:
    """Financial snapshot focused on profit-driver signals (revenue-profit divergence)."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")
    macro_beta = state.get("macro_beta", "")
    research = state.get("research_context", "")

    lines = [
        f"股票代码: {state['stock_code']}",
        f"股票名称: {state['stock_name']}",
        f"行业分类: {industry}",
        f"宏观防线: {macro_beta}",
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
        rev_changes = [fin.revenue[i] / fin.revenue[i-1] - 1
                       for i in range(1, len(fin.revenue))]
        profit_changes = [fin.net_profit_parent[i] / fin.net_profit_parent[i-1] - 1
                          for i in range(1, len(fin.net_profit_parent))]
        lines.append(f"收入年化变动: {[f'{c:.1%}' for c in rev_changes]}")
        lines.append(f"利润年化变动: {[f'{c:.1%}' for c in profit_changes]}")
        sync = "同步"
        if max(rev_changes) - min(rev_changes) >= 0.3:
            sync = "明显背离（典型周期/价格驱动特征）"
        lines.append(f"收入-利润变动同步性: {sync}")

    if len(fin.revenue) >= 2:
        margin_changes = []
        for i in range(len(fin.revenue)):
            if fin.revenue[i] > 0:
                margin_changes.append(fin.net_profit_parent[i] / fin.revenue[i] * 100)
        if margin_changes:
            lines.append(f"净利润率逐年: {[f'{m:.1f}%' for m in margin_changes]}")
            if max(margin_changes) - min(margin_changes) > 10:
                lines.append(
                    "⚠ 净利率波动 >10个百分点 → 典型周期品/涨价驱动信号")

    lines.append("")
    lines.append("=== 业务板块构成 ===")
    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%, "
            f"利润贡献 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"ROE: {seg.roe:.1f}%"
        )

    if research:
        lines.append("")
        lines.append("=== 行业研报/情报 ===")
        lines.append(research[:1500])

    return "\n".join(lines)


def _build_l2_snapshot(state: PipelineState) -> str:
    """Snapshot focused on asset structure, ATO, and growth metrics."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")
    research = state.get("research_context", "")

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
        f"⚠ 影子指标: 若原始≤40%但调整后>40%，大量产能尚未转固，应视为重资产",
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
        rev_cagr = (fin.revenue[2] / fin.revenue[0]) ** (1/2) - 1 \
            if fin.revenue[0] > 0 else 0
        lines.append(f"3年收入 CAGR: {rev_cagr:+.1%}")
    if len(fin.net_profit_parent) >= 3:
        profit_cagr = (fin.net_profit_parent[2] / fin.net_profit_parent[0]) ** (1/2) - 1 \
            if fin.net_profit_parent[0] > 0 else 0
        lines.append(f"3年利润 CAGR: {profit_cagr:+.1%}")
    lines.extend([
        f"营业收入 (近3年): {fin.revenue}",
        f"归母净利润 (近3年): {fin.net_profit_parent}",
    ])

    lines.append("")
    lines.append("=== 业务板块详情 ===")
    for seg in segments:
        lines.append(
            f"- {seg.name}: 收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}%, 增速 {seg.yoy_growth:+.1f}%, "
            f"利润贡献 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"ROE: {seg.roe:.1f}%"
        )

    if research:
        lines.append("")
        lines.append("=== 行业研报情报 ===")
        lines.append(research[:1500])

    return "\n".join(lines)


def _build_l3_snapshot(state: PipelineState) -> str:
    """Snapshot focused on competitive dynamics and ASP clues."""
    fin = state["financials"]
    segments = state.get("segments", [])
    industry = state.get("industry", "")
    research = state.get("research_context", "")

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
        margin_clue = ""
        if hasattr(seg, 'gross_margin') and seg.gross_margin > 0:
            if seg.gross_margin > 50:
                margin_clue = "高毛利率 → 技术壁垒/定价权，ASP 抗跌"
            elif seg.gross_margin < 15:
                margin_clue = "低毛利率 → 大宗/周期品特征，ASP 随行就市"
            elif seg.gross_margin < 30:
                margin_clue = "中等毛利率 → 工业品，面临温和竞争"
            else:
                margin_clue = "中高毛利率 → 有一定差异化"
        lines.append(
            f"- {seg.name}: "
            f"收入 {seg.revenue:.1f}亿 ({seg.revenue_share:.1f}%), "
            f"毛利率 {seg.gross_margin:.1f}% → {margin_clue}, "
            f"增速 {seg.yoy_growth:+.1f}%, "
            f"利润 {seg.profit_contribution:.1f}亿 ({seg.profit_share:.1f}%), "
            f"分支: {seg.branch or '未标注'}"
        )

    if research:
        lines.append("")
        lines.append("=== 行业研报/竞争情报 ===")
        lines.append(research[:2000])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Serial Agent Call Helpers
# ═══════════════════════════════════════════════════════════════

def _clean_llm_json(content: str) -> str:
    """Strip markdown fences from LLM output."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Drop opening fence (may have language tag)
        content = "\n".join(lines[1:])
        if content.endswith("```"):
            content = content[:-3]
    return content.strip()


def _inject_reflection(state: PipelineState, agent: str, prompt: str) -> str:
    """If this is a reflection round, prepend the reflection directive."""
    if state.get("reflection_round", 0) <= 0:
        return prompt

    if agent == "L1":
        prev = state.get("l1_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视利润驱动力】\n"
            f"上一轮你的判断是: {prev.get('profit_driver', '?')} "
            f"(置信度 {prev.get('confidence', 0):.0%})\n"
            f"请考虑市场定价是否揭示了模型忽略的因子。\n\n"
        )
    elif agent == "L2":
        prev = state.get("l2_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视商业模式诊断】\n"
            f"上一轮你的判断是: 渗透率={prev.get('penetration_stage', '?')}, "
            f"重资产={prev.get('heavy_asset', '?')}, "
            f"建议框架={prev.get('suggested_framework', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的因子。\n\n"
        )
    else:  # L3
        prev = state.get("l3_output", {})
        reexamine = (
            f"【⚠ 反思轮次 {state['reflection_round']} — 请重新审视竞争格局】\n"
            f"上一轮你的判断是: ASP={prev.get('asp_trend', '?')}, "
            f"份额={prev.get('market_share_trend', '?')}, "
            f"PEG={prev.get('peg_adjustment', '?')}, "
            f"CAPEX={prev.get('capex_status', '?')}\n"
            f"请考虑市场定价是否揭示了模型忽略的竞争因子。\n\n"
        )

    red_inject = build_red_team_injection(state)
    return REFLECTION_PROMPT_INJECT + red_inject + reexamine + prompt


# ═══════════════════════════════════════════════════════════════
#  Step 1 — L1: 利润驱动力分类
# ═══════════════════════════════════════════════════════════════

_VALID_PROFIT_DRIVERS = {"涨价驱动", "放量驱动", "稳定分红(C1)", "离散事件(C2)"}
# Backward-compat mapping for old LLM outputs
_PROFIT_DRIVER_ALIAS = {
    "稳定/事件": "稳定分红(C1)",
    "稳定": "稳定分红(C1)",
    "事件驱动": "离散事件(C2)",
}


def _normalize_profit_driver(raw: str) -> str:
    """Map any legacy or fuzzy LLM output to canonical ProfitDriver value."""
    raw = raw.strip()
    if raw in _VALID_PROFIT_DRIVERS:
        return raw
    if raw in _PROFIT_DRIVER_ALIAS:
        return _PROFIT_DRIVER_ALIAS[raw]
    # Best-effort fuzzy match
    for canonical in _VALID_PROFIT_DRIVERS:
        if canonical[:3] in raw or raw[:3] in canonical:
            return canonical
    return "涨价驱动"  # ultimate fallback


def _run_l1(state: PipelineState, model: str) -> dict:
    """
    Step 1: L1 — classify profit driver.

    Writes l1_output and profit_driver into state.
    Returns audit entry dict.
    """
    snapshot = _build_l1_snapshot(state)
    prompt = L1_AGENT_PROMPT.format(company_data=snapshot)
    prompt = _inject_reflection(state, "L1", prompt)

    t_start = time.perf_counter()
    fallback = False
    error_msg = ""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        content = _clean_llm_json(response.choices[0].message.content)
        result = json.loads(content)
    except Exception as e:
        fallback = True
        error_msg = str(e)[:200]
        result = {
            "profit_driver": "涨价驱动",
            "confidence": 0.0,
            "reasoning": f"LLM调用失败: {error_msg}",
            "primary_segment": "",
            "price_vs_volume_clue": "",
        }

    pd = _normalize_profit_driver(result.get("profit_driver", "涨价驱动"))
    result["profit_driver"] = pd

    state["l1_output"] = result
    state["profit_driver"] = pd

    latency = (time.perf_counter() - t_start) * 1000
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "")[:80]
    print(f"  ✓ [L1] 利润驱动力: {pd} (置信度 {confidence:.0%}) | {latency:.0f}ms")
    if reasoning:
        print(f"    依据: {reasoning}")

    return {
        "agent": "L1", "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
        "latency_ms": round(latency, 1),
        "fallback": fallback, "error": error_msg,
    }


# ═══════════════════════════════════════════════════════════════
#  Step 2 — L2: 商业模式定向诊断 (branch-aware)
# ═══════════════════════════════════════════════════════════════

_VALID_L2_FRAMEWORKS = {
    "PB-ROE底部锚", "PE/PEG弹性锚", "PS+管线估值",
    "PEG+S曲线+SOTP", "成熟PE+股息率", "DCF/DDM类债券", "概率加权/rNPV",
}


def _run_l2(state: PipelineState, model: str) -> dict:
    """
    Step 2: L2 — business model diagnosis with L1 context forced.

    L2's prompt dynamically switches diagnostic focus:
      - 涨价驱动 → 重资产属性 + CAPEX 状态
      - 放量驱动 → 渗透率阶段 + 在建工程影子指标
      - 稳定分红(C1) → 现金流可预测性 + 分红可持续性
      - 离散事件(C2) → 催化剂时间线 + 概率评估

    Writes l2_output into state.
    Returns audit entry dict.
    """
    snapshot = _build_l2_snapshot(state)
    l1 = state.get("l1_output", {})
    pd = l1.get("profit_driver", state.get("profit_driver", "涨价驱动"))
    l1_conf = l1.get("confidence", 0.5)
    l1_reasoning = l1.get("reasoning", "未获取L1分析")

    prompt = L2_AGENT_PROMPT.format(
        company_data=snapshot,
        profit_driver=pd,
        confidence=l1_conf,
        l1_reasoning=l1_reasoning,
    )
    prompt = _inject_reflection(state, "L2", prompt)

    t_start = time.perf_counter()
    fallback = False
    error_msg = ""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500,
        )
        content = _clean_llm_json(response.choices[0].message.content)
        result = json.loads(content)
    except Exception as e:
        fallback = True
        error_msg = str(e)[:200]
        result = _l2_fallback(state, pd, error_msg)

    # Validate suggested_framework
    fw = result.get("suggested_framework", "")
    if fw not in _VALID_L2_FRAMEWORKS:
        result["suggested_framework"] = _l2_default_framework(pd, result)

    state["l2_output"] = result

    latency = (time.perf_counter() - t_start) * 1000
    ps = result.get("penetration_stage", "") or "N/A"
    ha = result.get("heavy_asset", False)
    fw = result.get("suggested_framework", "?")
    capex = result.get("capex_note", "")
    print(f"  ✓ [L2] 渗透率: {ps} | 重资产: {ha} | 框架: {fw} | {latency:.0f}ms")
    if capex and capex != "N/A":
        print(f"    CAPEX: {capex[:120]}")

    return {
        "agent": "L2", "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
        "latency_ms": round(latency, 1),
        "fallback": fallback, "error": error_msg,
    }


def _l2_fallback(state: PipelineState, profit_driver: str, error_msg: str) -> dict:
    """Branch-aware L2 fallback when LLM call fails."""
    adj = state["financials"].adjusted_fa_to_ta
    is_heavy = adj > 0.40
    ato = state["financials"].ato
    ato_trend = state["financials"].ato_trend

    base = {
        "fa_to_ta_check": f"调整后(固+在)/总={adj:.1%}",
        "ato_shadow": f"ATO={ato:.3f}, 趋势={ato_trend}",
        "reasoning": f"LLM调用失败，使用分支规则默认值: {error_msg}",
    }

    if profit_driver == "涨价驱动":
        return {
            **base,
            "penetration_stage": "",
            "current_rate_estimate": "",
            "heavy_asset": is_heavy,
            "capex_note": "N/A（LLM失败，未获取CAPEX数据）",
            "suggested_framework": "PB-ROE底部锚" if is_heavy else "PE/PEG弹性锚",
        }
    elif profit_driver == "放量驱动":
        return {
            **base,
            "penetration_stage": "5-30%爆发期",
            "current_rate_estimate": "LLM失败，默认爆发期",
            "heavy_asset": is_heavy,
            "capex_note": "N/A",
            "suggested_framework": "PEG+S曲线+SOTP",
        }
    elif profit_driver == "稳定分红(C1)":
        return {
            **base,
            "penetration_stage": "",
            "current_rate_estimate": "",
            "heavy_asset": False,
            "capex_note": "N/A",
            "suggested_framework": "DCF/DDM类债券",
        }
    else:  # 离散事件(C2)
        return {
            **base,
            "penetration_stage": "",
            "current_rate_estimate": "",
            "heavy_asset": False,
            "capex_note": "N/A",
            "suggested_framework": "概率加权/rNPV",
        }


def _l2_default_framework(profit_driver: str, result: dict) -> str:
    """Return default framework when LLM output is invalid."""
    if profit_driver == "涨价驱动":
        return "PB-ROE底部锚" if result.get("heavy_asset", False) else "PE/PEG弹性锚"
    elif profit_driver == "放量驱动":
        ps = result.get("penetration_stage", "")
        if "<5%" in ps:
            return "PS+管线估值"
        elif ">30%" in ps:
            return "成熟PE+股息率"
        return "PEG+S曲线+SOTP"
    elif profit_driver == "稳定分红(C1)":
        return "DCF/DDM类债券"
    else:
        return "概率加权/rNPV"


# ═══════════════════════════════════════════════════════════════
#  Step 3 — L3: 竞争格局 + 最终框架落锚
# ═══════════════════════════════════════════════════════════════

_VALID_ASP_TRENDS = {"恶性崩塌(>15%)", "正常温和(<10%)", "稳定/涨价", "N/A"}
_VALID_CAPEX_STATUS = {"CAPEX高企/产能投放", "CAPEX萎缩/供给刚性", "N/A"}


def _run_l3(state: PipelineState, model: str) -> dict:
    """
    Step 3: L3 — competition analysis with full L1+L2 context.

    L3 receives:
      - L1: profit_driver, l1_reasoning
      - L2: penetration_stage, heavy_asset, suggested_framework, capex_note

    Its job: ASP trend, market share, PEG adjustment, supply-side CAPEX,
    governance flags, and FINAL valuation framework confirmation.

    Must pick from exactly 7 frameworks — free-form divergence = error.
    Writes l3_output into state.
    Returns audit entry dict.
    """
    snapshot = _build_l3_snapshot(state)
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})

    pd = l1.get("profit_driver", state.get("profit_driver", "涨价驱动"))
    l1_reasoning = l1.get("reasoning", "未获取L1分析")
    ps = l2.get("penetration_stage", state.get("penetration_stage", ""))
    ha = str(l2.get("heavy_asset", state.get("fixed_asset_heavy", False)))
    fw = l2.get("suggested_framework", "")
    capex_note = l2.get("capex_note", "")

    prompt = L3_AGENT_PROMPT.format(
        company_data=snapshot,
        profit_driver=pd,
        l1_reasoning=l1_reasoning,
        penetration_stage=ps or "不适用（非放量驱动）",
        heavy_asset=ha,
        suggested_framework=fw or "待定",
        capex_note=capex_note or "未获取",
    )
    prompt = _inject_reflection(state, "L3", prompt)

    t_start = time.perf_counter()
    fallback = False
    error_msg = ""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        content = _clean_llm_json(response.choices[0].message.content)
        result = json.loads(content)
    except Exception as e:
        fallback = True
        error_msg = str(e)[:200]
        result = _l3_fallback(state, pd, error_msg)

    # Validate and normalize critical fields
    asp = result.get("asp_trend", "N/A")
    if asp not in _VALID_ASP_TRENDS:
        asp = "N/A"
    result["asp_trend"] = asp

    capex = result.get("capex_status", "N/A")
    if capex not in _VALID_CAPEX_STATUS:
        capex = "N/A"
    result["capex_status"] = capex

    cfw = result.get("confirmed_framework", "")
    if cfw not in _VALID_L2_FRAMEWORKS:
        # Fallback to L2's suggested framework if L3's is invalid
        cfw = fw if fw in _VALID_L2_FRAMEWORKS else _l2_default_framework(pd, l2)
        result["confirmed_framework"] = cfw

    state["l3_output"] = result

    latency = (time.perf_counter() - t_start) * 1000
    share = result.get("market_share_trend", "?")
    peg = result.get("peg_adjustment", "?")
    print(f"  ✓ [L3] ASP: {asp} | 份额: {share} | PEG: {peg} | "
          f"CAPEX: {capex} | 框架: {cfw} | {latency:.0f}ms")
    gov = result.get("governance_flags", [])
    if gov:
        print(f"    ⛔ 治理红旗: {len(gov)} 条 — {gov[0][:100]}")

    return {
        "agent": "L3", "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
        "latency_ms": round(latency, 1),
        "fallback": fallback, "error": error_msg,
    }


def _l3_fallback(state: PipelineState, profit_driver: str, error_msg: str) -> dict:
    """Branch-aware L3 fallback when LLM call fails."""
    base = {
        "market_share_trend": "稳定",
        "peg_adjustment": 1.0,
        "competition_verdict": f"LLM调用失败: {error_msg}",
        "key_competitors": [],
        "supply_side_note": "",
        "governance_flags": [],
    }

    if profit_driver == "涨价驱动":
        return {**base, "asp_trend": "N/A", "capex_status": "N/A",
                "confirmed_framework": "PE/PEG弹性锚"}
    elif profit_driver == "放量驱动":
        return {**base, "asp_trend": "正常温和(<10%)", "capex_status": "N/A",
                "confirmed_framework": "PEG+S曲线+SOTP"}
    elif profit_driver == "稳定分红(C1)":
        return {**base, "asp_trend": "N/A", "capex_status": "N/A",
                "confirmed_framework": "DCF/DDM类债券"}
    else:
        return {**base, "asp_trend": "N/A", "capex_status": "N/A",
                "confirmed_framework": "概率加权/rNPV"}


# ═══════════════════════════════════════════════════════════════
#  In-Chain Early Validation
# ═══════════════════════════════════════════════════════════════

def _validate_in_chain(state: PipelineState) -> list[str]:
    """Lightweight early-check. Full validation runs in merge_qualitative."""
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})
    l3 = state.get("l3_output", {})

    conflicts = []
    pd = l1.get("profit_driver", "")
    ps = l2.get("penetration_stage", "")
    asp = l3.get("asp_trend", "")
    share = l3.get("market_share_trend", "")
    peg = l3.get("peg_adjustment", 1.0)
    fw = l3.get("confirmed_framework", l2.get("suggested_framework", ""))

    # Non-branch-B should not have penetration stage
    if pd not in ("放量驱动",) and ps not in ("",):
        conflicts.append(
            f"[Early] L1={pd} 但 L2渗透率={ps} — "
            f"渗透率仅适用于放量驱动，建议清空"
        )

    # ASP red line (branch B hard constraint)
    if asp == "恶性崩塌(>15%)" and share == "下滑" and peg > 0.7:
        conflicts.append(
            "[Early] ASP恶性崩塌+份额下滑 → PEG应为0.7 "
            f"(当前: {peg})"
        )

    # 涨价驱动 + penetration stage = contradiction
    if pd == "涨价驱动" and ps not in ("",):
        conflicts.append(
            f"[Early] 涨价驱动不应有渗透率({ps}) — "
            f"周期品估值锚是供需平衡表，非渗透率S曲线"
        )

    # Framework consistency: check that L3 didn't drift
    if fw and fw not in _VALID_L2_FRAMEWORKS:
        conflicts.append(f"[Early] L3最终框架 '{fw}' 不在7框架白名单中")

    return conflicts


# ═══════════════════════════════════════════════════════════════
#  Main Node: node_qualitative_chain
# ═══════════════════════════════════════════════════════════════

def node_qualitative_chain(state: PipelineState) -> PipelineState:
    """
    Serial qualitative chain — replaces dispatch_parallel_qualitative.

    Strictly enforces the INVEST_SOP 3-layer decision tree:
      L1 → profit_driver → L2 (branch-aware diagnosis) → L3 (framework anchor)

    All three agent outputs are written to state:
      - l1_output, profit_driver
      - l2_output (penetration_stage, heavy_asset, capex_note, suggested_framework)
      - l3_output (asp_trend, market_share_trend, peg_adjustment,
                   capex_status, confirmed_framework, governance_flags)

    After all three complete, runs in-chain early validation.
    The merge_qualitative node still executes downstream for full
    cross-validation and hard-coded red-line enforcement.
    """
    model = get_model()
    t0 = time.perf_counter()

    print(f"  ⛓  [Chain] L1 → L2 → L3 串行定性分析 (模型: {model})")

    audit = {}

    # —— Step 1: L1 — 利润驱动力分类 ——
    audit["L1"] = _run_l1(state, model)

    # —— Step 2: L2 — 商业模式定向诊断 (强制接收L1结论) ——
    audit["L2"] = _run_l2(state, model)

    # —— Step 3: L3 — 竞争格局 + 最终框架落锚 (接收L1+L2完备标签) ——
    audit["L3"] = _run_l3(state, model)

    # —— Early validation ——
    conflicts = _validate_in_chain(state)
    if conflicts:
        print(f"  ⚠ [Chain] 早期冲突检测 {len(conflicts)} 个:")
        for c in conflicts:
            print(f"    - {c[:150]}")

    total_latency = (time.perf_counter() - t0) * 1000
    sum_latency = sum(
        a.get("latency_ms", 0) for a in audit.values()
        if isinstance(a, dict) and "latency_ms" in a
    )
    audit["_total_wall_ms"] = round(total_latency, 1)
    audit["_mode"] = f"串行L1→L2→L3, wall={total_latency:.0f}ms, sum={sum_latency:.0f}ms"
    audit["_early_conflicts"] = len(conflicts)

    state["agent_audit"] = audit

    print(f"  ⛓  [Chain] 完成: wall={total_latency:.0f}ms, sum={sum_latency:.0f}ms")
    return state
