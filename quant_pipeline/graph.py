"""
INVEST SOP — LangGraph 并行 DAG 拓扑
=======================================
Parallel pipeline implementing the INVEST_SOP.md closed-loop framework.

Parallelism strategy:
  - L1/L2/L3 agents: serial chain (L1→L2→L3) within node_qualitative_chain node
  - SOTP segment valuation: Python ThreadPoolExecutor within dispatch_parallel_sotp node
  - LangGraph manages the overall DAG topology, conditional routing, and state flow

Graph structure:

                          START
                            │
                     [node0_fetch] ────(error)──► END
                            │ (ok)
                    [node0_quick_screen] ──(branch_D/abandon)──► END/branch_D
                            │ (pass)
                            │
                  [dispatch_chain]     ← Serial: L1 → L2 → L3 (strict dependency)
                            │
                  [merge_qualitative]     ← cross-validation & conflict resolution
                            │
                    <SOTP trigger check>
                      ┌────┴────┐
                    YES        NO
                      │          │
            [dispatch_sotp]   [node2_valuation]  ← ThreadPool: N segments concurrently
                      │          │
               [merge_sotp]     │
                      │          │
                      └────┬─────┘
                           │
                   [node3_reflection]     ← CIO reconciliation + deviation classifier
                           │
                    ┌──────┴──────┐
                    │ deviation    │
                    │ > 30%?       │
                    └──────┬──────┘
               ┌─────────┼─────────┐
         ┌─────┴────┐   YES   ┌────┴─────┐
         │ 泡沫容忍  │         │ 5-way路由 │
         └─────┬────┘         └────┬─────┘
               │          ┌───────┼───────┐
               │    ┌─────┴──┐ ┌──┴──┐ ┌──┴────┐
               │    │CAPEX红队│ │竞争红队│ │质变/未知│
               │    └─────┬──┘ └──┬──┘ └──┬────┘
               │          │       │       │
               │          └──┬────┘  ┌────┘
               │             │       │
               │      loop → dispatch / re-screen
               │                      │
               └──────────┬───────────┘
                          │
                     [node_report] → END

LangGraph features demonstrated:
  - Conditional routing: after quick screen, after SOTP check, after reflection
  - State management: all labels flow through TypedDict state
  - Cross-validation: merge node detects inconsistencies between parallel agent outputs
  - Reflection loop: conditional edge back to dispatch for re-examination
  - The DAG structure is genuine — not reducible to a simple while loop
"""

from typing import Literal
from langgraph.graph import StateGraph, END
from state import PipelineState, ScreeningVerdict

from nodes.node0_fetch import node0_fetch
from nodes.node0_quick_screen import node0_quick_screen, route_after_screening
from nodes.dispatch_chain import node_qualitative_chain
from nodes.merge_qualitative import merge_qualitative
from nodes.dispatch_sotp import dispatch_parallel_sotp
from nodes.merge_sotp import merge_sotp
from nodes.node2_valuation import node2_valuation, _check_sotp_trigger
from nodes.node3_reflection import node3_reflection, MAX_REFLECTION_ROUNDS
from nodes.red_team_agent import node_red_team, node_red_team_capex, node_red_team_competition, node_red_team_strategic_growth, node_red_team_pharma


# ============================================================
#  Branch-D node: 高风险博弈处理
# ============================================================

def node_branch_D(state: PipelineState) -> PipelineState:
    """
    When the stock fails cash-flow screening, enter Branch D.
    Valuation framework: 清算价值法 / 资产打折法.
    """
    state["profit_driver"] = "有毒的增长"
    state["penetration_stage"] = ""
    state["sotp_triggered"] = False
    state["competition_verdict"] = "未进入竞争分析（熔断降维）"
    return state


# ============================================================
#  Branch-E node: v3.2 战略成长股豁免
# ============================================================

def node_branch_E(state: PipelineState) -> PipelineState:
    """
    v3.2: When a high-barrier hard-tech stock triggers strategic growth exemption.
    These are AI chips / advanced packaging / IC substrate companies with:
      - Strategic tech keywords in name/industry/research
      - YoY revenue growth >= 30%
      - Negative/low cash flow (would fail normal screening)

    Valuation framework: 动态行业PS（市销率）— 35x base, 40x ultra-high-growth.
    Red team focuses on deep tech risks (chip fabrication, order visibility) not cash flow.
    """
    state["profit_driver"] = "战略成长股（烧钱期高壁垒硬科技）"
    state["penetration_stage"] = "5-30%爆发期"
    state["fixed_asset_heavy"] = False
    state["sotp_triggered"] = False
    state["competition_verdict"] = "战略成长豁免：不适用传统净现比快筛，改用动态行业PS估值"
    return state


# ============================================================
#  Routing Functions
# ============================================================

def route_after_fetch(state: PipelineState) -> Literal["node0_quick_screen", "__end__"]:
    if state.get("error"):
        print(f"  ✗ [Node 0 Fetch] 数据抓取失败，终止流水线: {state['error']}")
        return "__end__"
    return "node0_quick_screen"


def route_to_valuation(state: PipelineState) -> Literal["dispatch_parallel_sotp", "node2_valuation"]:
    """
    After qualitative merge, decide valuation path:
      - SOTP triggered (segments > 1) → dispatch parallel segment valuation
      - SOTP not triggered or single segment → node2_valuation

    硬编码红线① 已移入 node2_valuation._hard_sotp_check(),
    确保状态修改被 LangGraph 持久化（条件边不持久化状态）。
    """
    segments = state.get("segments", [])
    triggered, reasons = _check_sotp_trigger(segments)

    if triggered and len(segments) > 1:
        state["sotp_triggered"] = True
        state["sotp_trigger_reasons"] = reasons
        print(f"  → [Route] SOTP 触发 ({len(segments)} 个板块): {'; '.join(reasons[:2])}")
        return "dispatch_parallel_sotp"
    else:
        # 单板块或未触发 → 统一走 node2_valuation
        # node2_valuation 内部执行硬编码红线① 二次校验
        state["sotp_triggered"] = triggered
        state["sotp_trigger_reasons"] = reasons
        return "node2_valuation"


def intelligent_deviation_router(state: PipelineState) -> str:
    """
    v3.2 5-way intelligent deviation router — replaces binary should_reflect.

    Routes based on red_team_focus set by node3_reflection._classify_deviation():
      - "基本面质变"                  → node_qualitative_chain (re-analyse L1→L2→L3)
      - "Value Trap: 供给侧产能风险"   → node_red_team_capex
      - "Value Trap: 恶性价格战风险"   → node_red_team_competition
      - "泡沫定价容忍"                 → node_report (accept narrative premium)
      - "未知严重偏离"                 → node0_quick_screen (re-screen from scratch)

    Falls back to node_report when reflection is not triggered or max rounds reached.
    """
    triggered = state.get("reflection_triggered", False)
    round_num = state.get("reflection_round", 0)

    if not triggered or round_num >= MAX_REFLECTION_ROUNDS:
        return "node_report"

    focus = state.get("red_team_focus", "")

    if focus == "基本面质变":
        print(f"  → [Router] 基本面质变 → 回溯至 L1/L2/L3 串行重分析")
        return "node_qualitative_chain"
    elif focus.startswith("Value Trap: 供给侧"):
        print(f"  → [Router] CAPEX供给侧风险 → 专项CAPEX红队")
        return "node_red_team_capex"
    elif focus.startswith("Value Trap: 恶性价格战"):
        print(f"  → [Router] 竞争格局风险 → 专项竞争红队")
        return "node_red_team_competition"
    elif focus.startswith("Value Trap: 管线/监管"):
        print(f"  → [Router] 创新药管线/监管风险 → 专项医药红队")
        return "node_red_team_pharma"
    elif focus == "泡沫定价容忍":
        print(f"  → [Router] 泡沫定价容忍 → 接受叙事溢价，直接报告")
        return "node_report"
    else:
        print(f"  → [Router] 未知偏离 → 回溯至前置快筛节点")
        return "node0_quick_screen"


# ============================================================
#  Report node: 格式化输出 + 落盘 Markdown
# ============================================================

def _build_markdown(state: PipelineState) -> str:
    """Build an investment-committee-grade markdown report."""
    verdict = state.get("screening_verdict", "")
    valuation = state.get("valuation")
    stock_name = state.get("stock_name", "")
    stock_code = state.get("stock_code", "")
    deviation = state.get("valuation_deviation", 0.0)
    reflection_round = state.get("reflection_round", 0)
    governance_flags: list = state.get("governance_flags", [])
    error = state.get("error", "")
    current_price = state.get("current_price", 0)
    industry = state.get("industry", "")
    macro_beta = state.get("macro_beta", "")

    lines = []

    # ═══ CONFIDENTIAL HEADER ═══
    lines.append(f"# INVEST SOP 投研报告")
    lines.append(f"## {stock_name} · {stock_code}")
    lines.append("")
    lines.append(f"> **生成日期**: 2026-05-19 | **框架**: SOP v3.2 双轨制 (LangGraph 串行 DAG + 5路智能纠偏) | **分类**: 内部投研 · 禁止外传")
    lines.append(f"> **行业**: {industry or '—'} | **宏观防线**: {macro_beta or '—'}")

    # Show agent consensus
    consensus = state.get("label_consensus")
    conflicts = state.get("label_conflicts", [])
    if consensus is True:
        lines.append(f"> **Agent 一致性**: ✓ 三 Agent 交叉验证通过")
    elif consensus is False:
        lines.append(f"> **Agent 一致性**: ⚠ {len(conflicts)} 个冲突已标记")
        for c in conflicts[:3]:
            lines.append(f">   - {c[:100]}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # ═══ GOVERNANCE WARNING ═══
    if governance_flags:
        lines.append("## ⛔ 治理风控警告面板")
        lines.append("")
        lines.append("| 红旗类型 | 风险详情 | 风控措施 |")
        lines.append("|:--------:|----------|----------|")
        for flag in governance_flags:
            if any(kw in flag for kw in ["减持", "套现"]):
                tag, action = "大股东减持", "仓位 ≤ 2%，硬止损 -15%"
            elif any(kw in flag for kw in ["造假", "立案", "调查"]):
                tag, action = "财务诚信", "一票否决，禁止建仓"
            elif any(kw in flag for kw in ["审计", "非标"]):
                tag, action = "审计风险", "一票否决，禁止建仓"
            elif any(kw in flag for kw in ["质押", "爆仓"]):
                tag, action = "股权质押", "仓位 ≤ 1%，无融资"
            elif any(kw in flag for kw in ["关联", "担保"]):
                tag, action = "关联交易", "需法务尽调"
            else:
                tag, action = "其他警示", "需投委会审议"
            lines.append(f"| **{tag}** | {flag} | {action} |")
        lines.append("")
        lines.append("> ⚠ **风控红线**: 上述治理问题构成实质性障碍，上述风控措施为硬约束。")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ═══ EXECUTIVE SUMMARY — v3.1 双轨制 ═══
    lines.append("## Ⅰ. 投决摘要（双轨制决策面板）")
    lines.append("")

    cio = state.get("cio_reconciliation", {})
    classical_price = cio.get("classical_price", 0.0)
    narrative_price = cio.get("narrative_price", 0.0)
    premium_rate = cio.get("narrative_premium_rate", 0.0)
    classical_dev = cio.get("classical_deviation", 0.0)
    narrative_dev = cio.get("narrative_deviation", 0.0)

    # Fallback: if no CIO reconciliation yet, compute from valuation
    if not cio:
        sotp = state.get("sotp")
        if sotp and sotp.triggered and sotp.discounted_price > 0:
            classical_price = sotp.discounted_price
        elif sotp and sotp.triggered:
            classical_price = sotp.implied_price
        elif valuation:
            classical_price = valuation.base_case_price
        narrative_price = classical_price
        if current_price > 0:
            classical_dev = (classical_price - current_price) / current_price
            narrative_dev = classical_dev

    lines.append("| 决策维度 | 经典古典价值（清算底裤） | 动态科技叙事（多头弹性） | 现价与盘面观察 |")
    lines.append("|:---------|:-------------------------|:-------------------------|:--------------:|")

    # Row: 理论股价
    classical_label = "SOTP脱水底线" if state.get("sotp_triggered") else "纯框架估值"
    narrative_label = "总监综合裁决均衡价" if cio else "（同古典价）"
    lines.append(
        f"| **理论股价** | "
        f"{classical_price:.2f} 元 ({classical_label}) | "
        f"{narrative_price:.2f} 元 {narrative_label} | "
        f"**{current_price:.2f} 元** |"
    )

    # Row: 偏离度
    lines.append(
        f"| **偏离度** | "
        f"{_deviation_text(classical_dev)} | "
        f"{_deviation_text(narrative_dev)} | "
        f"— |"
    )

    # Row: 溢价率/折扣率
    if premium_rate != 0:
        direction = "🟢 叙事溢价" if premium_rate > 0 else "🔴 风险折扣"
        lines.append(
            f"| **CIO 调校** | "
            f"— | "
            f"{direction} **{premium_rate:+.1%}** | "
            f"— |"
        )

    # Row: 快筛 + 总股本
    lines.append(
        f"| **快筛结论** | "
        f"{_verdict_badge(verdict)} | "
        f"总股本: {state.get('total_shares', 0):.2f} 亿股 | "
        f"反思: {reflection_round} 轮 |"
    )
    lines.append("")

    # CIO 前瞻性冲突调和批注
    if cio and cio.get("cio_verdict"):
        lines.append("### CIO 前瞻性冲突调和批注")
        lines.append("")
        lines.append(f"> **裁决**: {cio.get('cio_verdict', '')}")
        lines.append(f"> **核心理由**: {cio.get('rationale', '')}")
        key_judgment = cio.get('key_judgment', '')
        if key_judgment:
            lines.append(f"> **关键判断**: {key_judgment}")
        lines.append("")
        lines.append("---")
        lines.append("")

    # ═══ KEY METRICS TABLE ═══
    lines.append("### 关键指标")
    lines.append("")
    lines.append("| 项目 | 内容 |")
    lines.append("|:-----|:-----|")

    if valuation:
        lines.append(f"| **基准市值** | {valuation.base_case_value:.0f} 亿 |")
        if valuation.bull_case_price and valuation.bear_case_price:
            val_range = f"{valuation.bear_case_price:.2f} ~ {valuation.bull_case_price:.2f} 元"
            lines.append(f"| **估值区间** | {val_range} |")

    # Show classical valuation for reference
    if valuation:
        lines.append(f"| **原始框架估值** | {valuation.base_case_price:.2f} 元 ({valuation.framework}) |")
    lines.append("")

    # ═══ QUALITATIVE LABELS ═══
    lines.append("## Ⅱ. 定性标签")
    lines.append("")
    lines.append("| 维度 | 判断 |")
    lines.append("|:-----|:-----|")
    lines.append(f"| L1 利润驱动力 | **{state.get('profit_driver', '—')}** |")
    lines.append(f"| L2 渗透率阶段 | **{state.get('penetration_stage', '—')}** |")
    lines.append(f"| 重资产判定 | {'是 (固/总 > 40%)' if state.get('fixed_asset_heavy') else '否'} |")
    lines.append(f"| ASP 趋势 | {state.get('competition_asp_trend', '—')} |")
    lines.append(f"| 市场份额趋势 | {state.get('market_share_trend', '—')} |")
    lines.append(f"| PEG 调整 | {state.get('peg_adjustment', 1.0):.2f} |")
    lines.append("")

    comp_verdict = state.get("competition_verdict", "")
    if comp_verdict:
        lines.append("**竞争格局研判**")
        lines.append(f"> {comp_verdict}")
        lines.append("")

    # ═══ VALUATION ═══
    if valuation:
        lines.append("---")
        lines.append("## Ⅲ. 估值分析")
        lines.append("")

        framework = valuation.framework or "（SOTP 分部加总）"
        lines.append(f"**估值框架**: {framework}")
        lines.append("")

        # SOTP breakdown
        sotp = state.get("sotp")
        if sotp and sotp.triggered:
            lines.append("### SOTP 分部估值明细")
            lines.append("")
            lines.append("| 业务板块 | 分支 | 2026E 净利 (亿) | 可比 PE | 估值 (亿) | 权重 |")
            lines.append("|:---------|:----:|:---------------:|:------:|:--------:|:----:|")
            total_val = sotp.total_valuation or 1
            for seg in sotp.segments:
                seg_val = (seg.profit_contribution * seg.comparable_pe) if seg.comparable_pe else 0
                weight = seg_val / total_val * 100 if total_val > 0 else 0
                lines.append(
                    f"| {seg.name} | {seg.branch or '—'} | "
                    f"{seg.profit_contribution:.1f} | "
                    f"{seg.comparable_pe:.0f}x | {seg_val:.0f} | {weight:.1f}% |"
                )
            lines.append(f"| **合计** | | | | **{total_val:.0f}** | **100%** |")
            lines.append("")
            lines.append(f"- **综合 PE (2026E)**: {sotp.composite_pe:.1f}x")
            lines.append(f"- **隐含股价**: {sotp.implied_price:.2f} 元")
            if sotp.trigger_reasons:
                lines.append(f"- **触发依据**: {'; '.join(sotp.trigger_reasons[:3])}")
            # 补丁2: 集团折价展示
            if sotp.conglomerate_discount > 0:
                lines.append(f"- 🔻 **集团折价**: {sotp.conglomerate_discount:.0%}（多元化业务缺乏协同）")
                lines.append(f"- **折价前估值**: {sotp.total_valuation:.0f} 亿 / {sotp.implied_price:.2f} 元")
                lines.append(f"- **折价后估值**: {sotp.discounted_valuation:.0f} 亿 / {sotp.discounted_price:.2f} 元")
            lines.append("")

        # Deviation
        if deviation > 0 and current_price > 0:
            theory = valuation.base_case_price
            upside = (valuation.bull_case_price / current_price - 1) * 100 if valuation.bull_case_price else 0
            downside = (valuation.bear_case_price / current_price - 1) * 100 if valuation.bear_case_price else 0

            lines.append("### 偏离度与风险收益比")
            lines.append("")
            if deviation > 0.30:
                lines.append(f"> 🔴 **严重偏离 ({deviation:.1%})** — "
                             f"理论价 {theory:.2f} vs 市场价 {current_price:.2f}")
            elif deviation > 0.15:
                lines.append(f"> 🟡 **显著偏离 ({deviation:.1%})** — "
                             f"理论价 {theory:.2f} vs 市场价 {current_price:.2f}")
            else:
                lines.append(f"> 🟢 **偏离可控 ({deviation:.1%})** — "
                             f"理论价 {theory:.2f} vs 市场价 {current_price:.2f}")
            lines.append("")

            if upside or downside:
                lines.append(f"- 上行空间: **{upside:+.1f}%** | 下行风险: **{downside:+.1f}%**")
                lines.append(f"- 风险收益比: **1:{abs(upside/downside) if downside else 999:.1f}**")
                lines.append("")

        # Risks
        if valuation.key_risks:
            lines.append("### 关键风险矩阵")
            lines.append("")
            for i, r in enumerate(valuation.key_risks[:5], 1):
                lines.append(f"{i}. ⚠ {r}")
            lines.append("")

        # Recommendation
        if valuation.recommendation:
            lines.append("### 买方建议")
            lines.append("")
            lines.append(f"> {valuation.recommendation}")
            lines.append("")

    # ═══ ERROR ═══
    if error:
        lines.append("---")
        lines.append("## ⚡ 异常记录")
        lines.append("")
        lines.append(f"```\n{error}\n```")
        lines.append("")

    # ═══ RED TEAM FINDINGS (补丁3) ═══
    red = state.get("red_team_output", {})
    if red and red.get("bearish_signals"):
        lines.append("---")
        lines.append("## Ⅳ-A. 红队（空头杠精）对抗性审查")
        lines.append("")
        lines.append(f"> **严重级别**: {red.get('severity', 'N/A')}")
        lines.append(f"> **核心逻辑**: {red.get('reasoning', 'N/A')}")
        lines.append("")
        lines.append("| # | 负面信号 |")
        lines.append("|:-:|:---------|")
        for i, s in enumerate(red.get("bearish_signals", []), 1):
            lines.append(f"| {i} | {s[:150]} |")
        most_dangerous = red.get("most_dangerous_signal", "")
        if most_dangerous:
            lines.append(f"\n> ⚠ **最有杀伤力的信号**: {most_dangerous}")
        adj = red.get("suggested_adjustments", {})
        if adj:
            if adj.get("peg_haircut_recommendation"):
                lines.append(f"\n- PEG建议调整: {adj['peg_haircut_recommendation']}")
            if adj.get("valuation_floor_risk"):
                lines.append(f"- 极端悲观下限: {adj['valuation_floor_risk']}")
        lines.append("")

    # ═══ AGENT AUDIT TRAIL ═══
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})
    l3 = state.get("l3_output", {})
    if l1 or l2 or l3:
        lines.append("---")
        lines.append("## Ⅳ. Agent 审计轨迹")
        lines.append("")
        lines.append("| Agent | 关键判断 | 置信度 |")
        lines.append("|:------|:---------|:------:|")
        if l1:
            lines.append(f"| **L1 利润驱动力** | {l1.get('profit_driver', '—')} | {l1.get('confidence', 0):.0%} |")
        if l2:
            lines.append(f"| **L2 生命周期** | {l2.get('penetration_stage', '—') or '不适用'} / {'重资产' if l2.get('heavy_asset') else '轻资产'} | — |")
        if l3:
            lines.append(f"| **L3 竞争格局** | ASP {l3.get('asp_trend', '—')} / 份额{l3.get('market_share_trend', '—')} | — |")
        lines.append("")

    # ═══ DISCLAIMER ═══
    lines.append("---")
    lines.append("## 免责声明")
    lines.append("")
    lines.append("> 本报告由 INVEST SOP 量化投研流水线自动生成，仅供内部研究参考，")
    lines.append("> 不构成任何形式的投资建议。报告中的估值基于公开财务数据和 LLM 分析，")
    lines.append("> 可能与实际情况存在偏差。投资决策请以人工尽调为准。")
    lines.append("")

    topology = "串行 DAG (L1→L2→L3 + SOTP并行 + CIO双轨调和 + 5路智能纠偏)"
    footer = f"*{stock_name} ({stock_code}) · SOP v3.2 · {topology} · 2026-05-19*"
    if reflection_round > 0:
        footer += f" · *经 {reflection_round} 轮投资总监反思修正*"
    lines.append(footer)

    return "\n".join(lines)


def _verdict_badge(verdict: str) -> str:
    if "放弃" in verdict:
        return "🔴 放弃"
    if "分支D" in verdict or "熔断" in verdict:
        return "🟡 熔断→分支D"
    return "🟢 通过"


def _deviation_badge(deviation: float, current_price: float, valuation) -> str:
    if not valuation or current_price <= 0:
        return "N/A"
    if deviation > 0.30:
        return f"🔴 {deviation:.1%} (严重偏离)"
    elif deviation > 0.15:
        return f"🟡 {deviation:.1%} (显著偏离)"
    return f"🟢 {deviation:.1%} (可控)"


def _deviation_text(deviation: float) -> str:
    """Simple deviation badge for dual-track panel (no valuation obj needed)."""
    if abs(deviation) > 0.30:
        direction = "极度高估" if deviation < 0 else "极度低估"
        return f"🔴 {deviation:+.1%} ({direction})"
    elif abs(deviation) > 0.15:
        direction = "高估" if deviation < 0 else "低估"
        return f"🟡 {deviation:+.1%} ({direction})"
    elif abs(deviation) > 0.05:
        return f"🟢 {deviation:+.1%} (合理)"
    return f"🟢 {deviation:+.1%} (基本吻合)"


# ═══════════════════════════════════════════════════════════
#  PDF 生成 (Markdown → HTML → PDF via weasyprint)
# ═══════════════════════════════════════════════════════════

PDF_CSS = """
  @page { size: A4; margin: 2cm 2.2cm; }
  body { font-family: 'Noto Sans CJK SC', 'SimSun', 'Microsoft YaHei', sans-serif;
         font-size: 11pt; line-height: 1.7; color: #1a1a1a; }
  h1 { font-size: 18pt; border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; margin-top: 0; }
  h2 { font-size: 14pt; color: #1a3a5c; margin-top: 28px; }
  h3 { font-size: 12pt; color: #2c5f8a; margin-top: 20px; }
  table { width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 9.5pt; }
  th { background: #1a3a5c; color: white; padding: 6px 8px; text-align: left; }
  td { border: 1px solid #ddd; padding: 5px 8px; }
  tr:nth-child(even) { background: #f7f9fb; }
  blockquote { border-left: 4px solid #1a3a5c; padding: 8px 14px;
               margin: 12px 0; background: #f0f4f8; color: #333; }
  code { background: #f0f0f0; padding: 1px 4px; font-size: 9pt; }
  hr { border: none; border-top: 1px solid #ddd; margin: 24px 0; }
  strong { color: #1a3a5c; }
  .footer { font-size: 8pt; color: #999; margin-top: 32px; border-top: 1px solid #ccc; padding-top: 8px; }
"""


def _markdown_to_html(md_text: str) -> str:
    """Convert INVEST SOP markdown to styled HTML for PDF rendering."""
    from markdown_it import MarkdownIt
    md = MarkdownIt().enable(["table", "strikethrough"])

    body = md.render(md_text)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><style>{PDF_CSS}</style></head>
<body>
{body}
</body>
</html>"""
    return html


def _write_pdf(stock_code: str, md_text: str) -> str:
    """Generate PDF from markdown. Returns file path, or empty string on failure."""
    try:
        from weasyprint import HTML
        html = _markdown_to_html(md_text)
        pdf_path = f"{stock_code}_REPORT.pdf"
        HTML(string=html).write_pdf(pdf_path)
        return pdf_path
    except Exception as e:
        print(f"  ⚠ PDF 生成失败: {e}")
        return ""


def node_report(state: PipelineState) -> PipelineState:
    """Print a formatted summary and write the full report to .md file."""
    verdict = state.get("screening_verdict", "")
    valuation = state.get("valuation")
    deviation = state.get("valuation_deviation", 0.0)
    reflection_round = state.get("reflection_round", 0)
    consensus = state.get("label_consensus")

    print("\n" + "=" * 64)
    print(f"  INVEST SOP 分析报告 — {state.get('stock_name', '')} ({state.get('stock_code', '')})")
    print("=" * 64)
    print(f"  快筛结论: {verdict}")
    print(f"  Agent 一致性: {'✓ 三 Agent 一致' if consensus else '⚠ 有冲突' if consensus is False else 'N/A'}")
    print(f"  L1 利润驱动力: {state.get('profit_driver', 'N/A')}")
    print(f"  L2 渗透率阶段: {state.get('penetration_stage', 'N/A')}")
    print(f"  SOTP 触发: {state.get('sotp_triggered', False)}")
    print(f"  ASP 趋势: {state.get('competition_asp_trend', 'N/A')}")
    print(f"  份额趋势: {state.get('market_share_trend', 'N/A')}")
    print(f"  PEG 调整: {state.get('peg_adjustment', 1.0)}")
    print(f"  反思轮次: {reflection_round}")
    print(f"  估值偏离度: {deviation:.1%}")

    red = state.get("red_team_output", {})
    if red and red.get("bearish_signals"):
        print(f"  🩸 红队严重级别: {red.get('severity', 'N/A')} | 发现 {len(red.get('bearish_signals', []))} 个负面信号")

    cio = state.get("cio_reconciliation", {})
    if cio:
        classical = cio.get("classical_price", 0)
        narrative = cio.get("narrative_price", 0)
        premium = cio.get("narrative_premium_rate", 0)
        print(f"  🏛  CIO 双轨裁决:")
        print(f"    古典价(清算底裤): {classical:.2f} 元 | 叙事价(多头弹性): {narrative:.2f} 元 | 调校: {premium:+.1%}")
        verdict = cio.get("cio_verdict", "")
        if verdict:
            print(f"    裁决: {verdict[:120]}")

    if valuation:
        print(f"\n  估值框架: {valuation.framework}")
        print(f"  基准市值: {valuation.base_case_value:.0f} 亿")
        print(f"  基准股价: {valuation.base_case_price:.2f} 元")
        if valuation.bull_case_price:
            print(f"  乐观股价: {valuation.bull_case_price:.2f} 元")
        if valuation.bear_case_price:
            print(f"  悲观股价: {valuation.bear_case_price:.2f} 元")

        sotp = state.get("sotp")
        if sotp and sotp.triggered:
            print(f"\n  [SOTP 分部估值]")
            for seg in sotp.segments:
                print(f"    {seg.name}: PE {seg.comparable_pe:.0f}x")
            if sotp.conglomerate_discount > 0:
                print(f"  🔻 集团折价: {sotp.conglomerate_discount:.0%} → 折价后 {sotp.discounted_price:.2f} 元")

        if valuation.key_risks:
            print(f"\n  关键风险:")
            for r in valuation.key_risks[:5]:
                print(f"    ⚠ {r}")

        if valuation.recommendation:
            print(f"\n  买方立场:\n    {valuation.recommendation}")

    if state.get("error"):
        print(f"\n  ❌ 错误: {state['error']}")

    conflicts = state.get("label_conflicts", [])
    if conflicts:
        print(f"\n  ⚠ Agent 交叉验证冲突:")
        for c in conflicts:
            print(f"    - {c[:120]}")

    print("\n" + "=" * 64 + "\n")

    # Write markdown file
    stock_code = state.get("stock_code", "UNKNOWN")
    md_path = f"{stock_code}_REPORT.md"
    markdown = _build_markdown(state)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print(f"  📄 Markdown: {md_path}")

    # Generate PDF (only when explicitly requested)
    if state.get("generate_pdf"):
        pdf_path = _write_pdf(stock_code, markdown)
        if pdf_path:
            print(f"  📕 PDF:      {pdf_path}")

    print()

    return state


# ============================================================
#  Graph Construction
# ============================================================

def build_graph() -> StateGraph:
    """
    Construct and compile the INVEST SOP parallel LangGraph DAG.

    Topology:

      START → node0_fetch → node0_quick_screen
        (pass) → node_qualitative_chain (3 LLMs concurrent)
          → merge_qualitative (cross-validation)
            → (SOTP check)
              (yes) → dispatch_parallel_sotp (N LLMs concurrent) → merge_sotp
              (no)  → node2_valuation
                → node3_reflection
                  → (>30%) loop → node_qualitative_chain
                  → (≤30%) node_report → END
    """
    graph = StateGraph(PipelineState)

    # ——— Add all nodes ———
    graph.add_node("node0_fetch", node0_fetch)
    graph.add_node("node0_quick_screen", node0_quick_screen)
    graph.add_node("node_branch_D", node_branch_D)
    graph.add_node("node_branch_E", node_branch_E)  # v3.2: 战略成长股豁免

    # Serial qualitative chain (L1 → L2 → L3 in strict dependency order)
    graph.add_node("node_qualitative_chain", node_qualitative_chain)
    graph.add_node("merge_qualitative", merge_qualitative)

    # Single-framework valuation (non-SOTP path)
    graph.add_node("node2_valuation", node2_valuation)

    # SOTP parallel dispatch (runs N segment agents concurrently)
    graph.add_node("dispatch_parallel_sotp", dispatch_parallel_sotp)
    graph.add_node("merge_sotp", merge_sotp)

    # Reflection and report
    graph.add_node("node3_reflection", node3_reflection)
    graph.add_node("node_red_team", node_red_team)  # 补丁3: 空头杠精（保留向后兼容）
    graph.add_node("node_red_team_capex", node_red_team_capex)  # v3.2: CAPEX供给侧专项红队
    graph.add_node("node_red_team_competition", node_red_team_competition)  # v3.2: 竞争格局专项红队
    graph.add_node("node_red_team_strategic_growth", node_red_team_strategic_growth)  # v3.2: 战略成长股深水区红队
    graph.add_node("node_red_team_pharma", node_red_team_pharma)  # v3.3: 创新药/Biotech 深水区红队
    graph.add_node("node_report", node_report)

    # ——— Entry point ———
    graph.set_entry_point("node0_fetch")

    # ——— Edges ———

    # After data fetch: route to quick screen or abort
    graph.add_conditional_edges(
        "node0_fetch",
        route_after_fetch,
        {"node0_quick_screen": "node0_quick_screen", "__end__": END},
    )

    # After quick screen: pass, branch-D, branch-E, or abandon
    graph.add_conditional_edges(
        "node0_quick_screen",
        route_after_screening,
        {
            "node_qualitative_chain": "node_qualitative_chain",
            "node_branch_D": "node_branch_D",
            "node_branch_E": "node_branch_E",
            "__end__": END,
        },
    )

    # Branch D → single-framework valuation
    graph.add_edge("node_branch_D", "node2_valuation")
    # Branch E → same valuation node (picks Branch-E PS logic internally)
    graph.add_edge("node_branch_E", "node_qualitative_chain")

    # ═══════════════════════════════════════════════════════════
    #  PARALLEL BLOCK #1: L1/L2/L3 → Cross-Validation
    # ═══════════════════════════════════════════════════════════
    graph.add_edge("node_qualitative_chain", "merge_qualitative")

    # After merge: route to SOTP parallel or single-framework valuation
    graph.add_conditional_edges(
        "merge_qualitative",
        route_to_valuation,
        {"dispatch_parallel_sotp": "dispatch_parallel_sotp", "node2_valuation": "node2_valuation"},
    )

    # ═══════════════════════════════════════════════════════════
    #  PARALLEL BLOCK #2: SOTP Segment Valuation
    # ═══════════════════════════════════════════════════════════
    graph.add_edge("dispatch_parallel_sotp", "merge_sotp")

    # Both valuation paths converge to reflection
    graph.add_edge("merge_sotp", "node3_reflection")
    graph.add_edge("node2_valuation", "node3_reflection")

    # ═══════════════════════════════════════════════════════════
    #  REFLECTION LOOP (v3.2: 5-way intelligent router + specialized red teams)
    # ═══════════════════════════════════════════════════════════
    graph.add_conditional_edges(
        "node3_reflection",
        intelligent_deviation_router,
        {
            "node_report": "node_report",
            "node_qualitative_chain": "node_qualitative_chain",
            "node0_quick_screen": "node0_quick_screen",
            "node_red_team_capex": "node_red_team_capex",
            "node_red_team_competition": "node_red_team_competition",
            "node_red_team_pharma": "node_red_team_pharma",
        },
    )
    # Specialized red team findings → inject into L1/L2/L3 re-analysis
    graph.add_edge("node_red_team_capex", "node_qualitative_chain")
    graph.add_edge("node_red_team_competition", "node_qualitative_chain")
    graph.add_edge("node_red_team_strategic_growth", "node_qualitative_chain")
    graph.add_edge("node_red_team_pharma", "node_qualitative_chain")  # v3.3: 创新药/Biotech 深水区红队
    # Legacy red team → qualitative chain (retained for backward compatibility)
    graph.add_edge("node_red_team", "node_qualitative_chain")

    # Report → END
    graph.add_edge("node_report", END)

    # ——— Compile ———
    return graph.compile()


# ============================================================
#  Convenience runner
# ============================================================

def run_pipeline(state: PipelineState, generate_pdf: bool = False) -> PipelineState:
    """Build the graph and invoke it once. Minimal input: stock_code only."""
    state.setdefault("reflection_round", 0)
    state.setdefault("generate_pdf", generate_pdf)
    state.setdefault("valuation_deviation", 0.0)
    state.setdefault("reflection_triggered", False)
    state.setdefault("governance_flags", [])
    state.setdefault("research_context", "")
    state.setdefault("industry", "")
    state.setdefault("macro_beta", "进攻防线")
    state.setdefault("segment_valuations", [])
    state.setdefault("label_conflicts", [])
    state.setdefault("sotp_trigger_reasons", [])
    state.setdefault("exemption_reasons", [])  # v3.2: 战略成长股豁免
    state.setdefault("pharma_valuation_details", {})  # v3.3: 医药PS估值明细
    state.setdefault("red_team_output", {})  # 补丁3
    state.setdefault("red_team_focus", "")  # v3.2: 智能路由注入的红队专项指令
    state.setdefault("latest_quarter_signals", {})  # v3.2: 最新季度信号
    state.setdefault("cio_reconciliation", {})  # v3.1 CIO双轨调和
    state.setdefault("asp_trend", "N/A")  # v3.2 ASP趋势（分支B专属）
    state.setdefault("capex_status", "N/A")  # v3.2 CAPEX状态（分支A专属）
    app = build_graph()
    final_state = app.invoke(state)
    return final_state


# ============================================================
#  Demo: 驰宏锌锗 quick test
# ============================================================
if __name__ == "__main__":
    from dotenv import load_dotenv; load_dotenv()
    from utils import config as _
    from state import FinancialMetrics, BusinessSegment

    demo_state: PipelineState = {
        "stock_code": "600497",
        "stock_name": "驰宏锌锗",
        "current_price": 9.95,
        "total_shares": 50.40,
        "industry": "铅锌",
        "financials": FinancialMetrics(
            revenue=[220.69, 188.03, 240.59],
            net_profit_parent=[15.07, 12.93, 10.35],
            op_cash_flow=[36.06, 23.66, 37.63],
            fixed_assets=77.71,
            total_assets=258.99,
            debt_ratio=27.7,
        ),
        "segments": [
            BusinessSegment(
                name="锌产品",
                revenue=140.42, revenue_share=58.4,
                profit_contribution=4.7, profit_share=35.0,
                gross_margin=9.48, yoy_growth=23.0,
                branch="分支A",
            ),
            BusinessSegment(
                name="银产品",
                revenue=32.18, revenue_share=13.4,
                profit_contribution=2.8, profit_share=20.7,
                gross_margin=24.41, yoy_growth=48.8,
                branch="分支A",
            ),
            BusinessSegment(
                name="锗产品(提取端)",
                revenue=9.75, revenue_share=4.1,
                profit_contribution=3.5, profit_share=16.0,
                gross_margin=65.0, yoy_growth=30.0,
                branch="分支B",
            ),
        ],
        "governance_flags": [],
    }

    print("\n▶ 启动 INVEST SOP 并行流水线 (LangGraph DAG)")
    print(f"  标的: {demo_state['stock_name']} ({demo_state['stock_code']})")
    print(f"  行业: {demo_state.get('industry', 'N/A')}")
    print(f"  拓扑: L1→L2→L3 串行链 → 交叉验证 → SOTP 并行 → CIO双轨 → 5路智能纠偏")

    result = run_pipeline(demo_state)

    print("▶ 流水线完成。最终状态键值:")
    for k in sorted(result.keys()):
        v = result[k]
        if hasattr(v, 'model_dump'):
            v = str(v.model_dump())[:120]
        elif isinstance(v, (list, dict)):
            v = str(v)[:120]
        elif isinstance(v, float):
            v = f"{v:.2f}"
        print(f"    {k}: {v}")
