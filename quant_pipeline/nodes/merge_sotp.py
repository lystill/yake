"""
INVEST_SOP v3.1 — SOTP 聚合层 + 核心主业叙事溢价
==================================================
Fan-in module that applies narrative premium coefficients to
high-barrier, high-growth core segments to prevent traditional
business segments from dragging the valuation down to pure
commodity/chemical levels.
"""

from state import PipelineState, BusinessSegment, SOTPResult, ValuationConclusion


# ——— Segments eligible for narrative premium ———
NARRATIVE_PREMIUM_KEYWORDS = [
    "前驱体", "HBM", "high-k", "光刻胶", "半导体", "KrF", "ArF",
    "锗", "镓", "铟", "稀土", "碳化硅", "氮化镓",
]

NARRATIVE_PREMIUM_MULTIPLIER = 1.20  # +20% for core tech moat segments


def _is_narrative_eligible(seg: dict | BusinessSegment) -> bool:
    """Check if a segment qualifies for the narrative premium."""
    if isinstance(seg, dict):
        name = seg.get("segment_name", seg.get("name", ""))
        branch = seg.get("segment_branch", seg.get("branch", ""))
    else:
        name = seg.name
        branch = seg.branch
    return (
        branch == "分支B"
        and any(kw in name for kw in NARRATIVE_PREMIUM_KEYWORDS)
    )


def _apply_narrative_premium(
    state: PipelineState,
    segments: list[dict | BusinessSegment],
) -> float:
    """
    Apply narrative premium to core high-barrier, high-growth segments.

    Rules:
      - 分支B (放量驱动) + 5-30%爆发期 + key semiconductor materials → 1.20x
      - All other segments → 1.00x (no adjustment)

    Returns the adjusted total valuation.
    """
    l1_driver = state.get("profit_driver", "")
    l2_stage = state.get("penetration_stage", "")

    # Only activate in 放量驱动 + 爆发期 context
    narrative_active = (
        l1_driver == "放量驱动"
        and l2_stage == "5-30%爆发期"
    )

    total_adjusted = 0.0

    for seg in segments:
        if isinstance(seg, dict):
            seg_value = seg.get("valuation", seg.get("segment_value", 0.0))
            seg_name = seg.get("segment_name", seg.get("name", ""))
            seg_branch = seg.get("segment_branch", seg.get("branch", ""))
        else:
            seg_value = seg.profit_contribution * seg.comparable_pe if seg.comparable_pe else 0
            seg_name = seg.name
            seg_branch = seg.branch

        if narrative_active and _is_narrative_eligible(seg):
            adjusted = seg_value * NARRATIVE_PREMIUM_MULTIPLIER
            if isinstance(seg, dict):
                seg["premium_applied"] = (
                    "触发 +20% 叙事溢价 (AI算力/高认证壁垒/供应链卡脖子)"
                )
                seg["valuation"] = round(adjusted, 1)
            total_adjusted += adjusted
        else:
            total_adjusted += seg_value

    return round(total_adjusted, 1)


def _compute_conglomerate_discount(
    total_value: float,
    num_segments: int,
) -> float:
    """
    Apply conglomerate discount for multi-segment companies.

    Rationale: multi-business conglomerates often trade at a discount
    to sum-of-parts due to complexity and capital allocation inefficiency.

    Scale:
      - 2 segments: 5% discount
      - 3 segments: 8% discount
      - 4+ segments: 12% discount
    """
    if num_segments <= 1:
        return total_value
    elif num_segments == 2:
        discount = 0.05
    elif num_segments == 3:
        discount = 0.08
    else:
        discount = 0.12

    return round(total_value * (1 - discount), 1)


# ═══════════════════════════════════════════════════════════════
#  Main merge_sotp node — fan-in after parallel segment agents
# ═══════════════════════════════════════════════════════════════

def merge_sotp(state: PipelineState) -> PipelineState:
    """
    Collect segment valuations from parallel dispatch, apply narrative
    premium and conglomerate discount, then produce SOTPResult and
    ValuationConclusion.

    This is the fan-in counterpart to dispatch_parallel_sotp.
    On the SOTP path, this replaces node2_valuation's LLM call —
    segment values come from individual LLM agents in the dispatch node.
    """
    segments = state.get("segments", [])
    segment_valuations = state.get("segment_valuations", [])

    # Ensure sotp_triggered is set (conditional edge sets it, but may not persist)
    state["sotp_triggered"] = True

    if not segment_valuations:
        # No parallel results — fall back to raw segments with narrative premium
        print("  ⚠ [merge_sotp] 无并行板块估值结果，使用回退方案")
        return _merge_sotp_fallback(state, segments)

    print(f"  ✓ [merge_sotp] 聚合 {len(segment_valuations)} 个板块估值...")

    # Build SOTP segments from agent outputs
    sotp_segments = []
    total_val = 0.0
    total_profit = 0.0

    for sv in segment_valuations:
        # Map from dispatch_sotp output keys:
        #   segment_name / segment_profit / segment_branch / comparable_pe / valuation
        seg_name = sv.get("segment_name", sv.get("name", ""))
        pe = sv.get("comparable_pe", sv.get("pe_assigned", 0))
        profit = sv.get("segment_profit", sv.get("net_profit_2026e", 0))
        branch = sv.get("segment_branch", sv.get("branch", ""))
        seg_value = sv.get("valuation", sv.get("segment_value", profit * pe if profit and pe else 0))

        total_val += seg_value
        total_profit += profit

        sotp_segments.append(BusinessSegment(
            name=seg_name,
            revenue=0,
            revenue_share=0,
            profit_contribution=profit,
            profit_share=0,
            gross_margin=0,
            yoy_growth=0,
            branch=branch,
            comparable_pe=pe,
        ))

    # Apply narrative premium to eligible segments
    narrative_total = _apply_narrative_premium(state, segment_valuations)

    # Use narrative-adjusted values for the SOTP total
    if narrative_total > total_val:
        print(f"    📈 叙事溢价激活: {total_val:.0f}亿 → {narrative_total:.0f}亿 (+{(narrative_total/total_val - 1)*100:.0f}%)")
        total_val = narrative_total

    # Compute conglomerate discount
    num_segments = len(sotp_segments)
    discount = 0.0
    if num_segments == 2:
        discount = 0.05
    elif num_segments == 3:
        discount = 0.08
    elif num_segments >= 4:
        discount = 0.12

    discounted_val = _compute_conglomerate_discount(total_val, num_segments)
    if discount > 0:
        print(f"    🔻 集团折价: {discount:.0%} → 折价后 {discounted_val:.0f}亿")

    total_shares = state.get("total_shares", 1.0)
    if total_shares <= 0:
        total_shares = 1.0

    implied_price = total_val / total_shares
    discounted_price = discounted_val / total_shares

    # Composite PE
    composite_pe = total_val / total_profit if total_profit > 0 else 0

    # Build SOTPResult
    sotp = SOTPResult(
        triggered=True,
        trigger_reasons=state.get("sotp_trigger_reasons", []),
        segments=sotp_segments,
        total_valuation=round(total_val, 1),
        implied_price=round(implied_price, 2),
        composite_pe=round(composite_pe, 1),
        conglomerate_discount=discount,
        discounted_valuation=round(discounted_val, 1),
        discounted_price=round(discounted_price, 2),
    )
    state["sotp"] = sotp

    # Build ValuationConclusion
    # Bull/bear: ±20% around base
    bull_price = round(discounted_price * 1.20, 2)
    bear_price = round(discounted_price * 0.80, 2)

    state["valuation"] = ValuationConclusion(
        framework="SOTP 分部加总 (v3.1 叙事溢价+集团折价)",
        sotp=sotp,
        base_case_value=round(discounted_val, 1),
        base_case_price=round(discounted_price, 2),
        bull_case_price=bull_price,
        bear_case_price=bear_price,
        key_risks=_collect_risks(state),
        recommendation=_build_recommendation(state, discounted_price),
    )

    print(f"    ✓ SOTP 估值: {discounted_val:.0f}亿 / {discounted_price:.2f}元 (PE {composite_pe:.1f}x)")

    return state


def _merge_sotp_fallback(state: PipelineState, segments: list) -> PipelineState:
    """Fallback when no segment agent results are available."""
    total_val = 0.0
    total_profit = 0.0
    sotp_segments = []

    for seg in segments:
        profit = seg.profit_contribution
        pe = seg.comparable_pe if seg.comparable_pe > 0 else 15.0
        seg_val = profit * pe
        total_val += seg_val
        total_profit += profit
        sotp_segments.append(BusinessSegment(
            name=seg.name,
            revenue=seg.revenue,
            revenue_share=seg.revenue_share,
            profit_contribution=profit,
            profit_share=seg.profit_share,
            gross_margin=seg.gross_margin,
            yoy_growth=seg.yoy_growth,
            branch=seg.branch,
            comparable_pe=pe,
        ))

    total_shares = state.get("total_shares", 1.0)
    if total_shares <= 0:
        total_shares = 1.0

    implied_price = total_val / total_shares
    composite_pe = total_val / total_profit if total_profit > 0 else 0
    discount = 0.08 if len(segments) >= 3 else (0.05 if len(segments) == 2 else 0)
    discounted_val = total_val * (1 - discount)
    discounted_price = discounted_val / total_shares

    sotp = SOTPResult(
        triggered=True,
        trigger_reasons=state.get("sotp_trigger_reasons", []),
        segments=sotp_segments,
        total_valuation=round(total_val, 1),
        implied_price=round(implied_price, 2),
        composite_pe=round(composite_pe, 1),
        conglomerate_discount=discount,
        discounted_valuation=round(discounted_val, 1),
        discounted_price=round(discounted_price, 2),
    )
    state["sotp"] = sotp

    state["valuation"] = ValuationConclusion(
        framework="SOTP 分部加总 (回退模式·无LLM板块分析)",
        sotp=sotp,
        base_case_value=round(discounted_val, 1),
        base_case_price=round(discounted_price, 2),
        bull_case_price=round(discounted_price * 1.20, 2),
        bear_case_price=round(discounted_price * 0.80, 2),
        key_risks=["板块估值使用默认PE(无LLM深度分析)"],
        recommendation="SOTP 回退模式估值，建议重新运行获取 LLM 板块分析",
    )

    return state


def _collect_risks(state: PipelineState) -> list[str]:
    """Collect cross-cutting risks from various state fields."""
    risks = []
    asp = state.get("competition_asp_trend", "")
    share = state.get("market_share_trend", "")
    gov = state.get("governance_flags", [])

    if "年降>15%" in asp:
        risks.append("ASP年降>15%：核心产品面临严重价格压力")
    if "下滑" in share:
        risks.append("市场份额下滑：竞争地位正在恶化")
    if gov:
        risks.append(f"治理警示: {len(gov)} 条红旗")

    fin = state.get("financials")
    if fin and fin.debt_ratio > 60:
        risks.append(f"高负债率: {fin.debt_ratio:.1f}%")

    return risks


def _build_recommendation(state: PipelineState, discounted_price: float) -> str:
    """Build a brief buy-side recommendation."""
    current_price = state.get("current_price", 0.0)
    name = state.get("stock_name", "")
    macro = state.get("macro_beta", "")

    if current_price <= 0:
        return f"{name} SOTP 估值 {discounted_price:.2f} 元，待确认市场价格后判断"

    upside = (discounted_price / current_price - 1) * 100

    if upside > 30:
        action = "显著低估，建议重点研究"
    elif upside > 10:
        action = "存在安全边际，可考虑分批建仓"
    elif upside > -10:
        action = "估值合理，观望为主"
    elif upside > -30:
        action = "估值偏高，仅适合在进攻防线中作为卫星仓位"
    else:
        action = "严重高估，建议回避"

    return f"{name} SOTP 公允价 {discounted_price:.2f} 元 vs 现价 {current_price:.2f} 元 ({upside:+.1f}%)。{macro}环境下，{action}。"
