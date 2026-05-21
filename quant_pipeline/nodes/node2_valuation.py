"""
Node 2 — 量化估值 (SOTP Trigger + Valuation)
================================================
Step A: Pure Python SOTP trigger check (no LLM).
Step B: Route to branch-specific LLM valuation prompt.
"""

import json
import litellm
from utils.xml_extractor import extract_json
from utils.strategic_context import get_strategic_context
from state import (
    PipelineState,
    BusinessSegment,
    SOTPResult,
    ValuationConclusion,
    ProfitDriver,
    ScreeningVerdict,
)
from prompts.valuation import (
    SOTP_VALUATION_PROMPT,
    SINGLE_FRAMEWORK_VALUATION_PROMPT,
    BRANCH_D_VALUATION_PROMPT,
    BRANCH_E_VALUATION_PROMPT,
    BRANCH_E_PHARMA_VALUATION_PROMPT,
)
from utils.config import get_model


# ============================================================
#  Step A — SOTP Trigger Math (Hard-coded Python)
# ============================================================

SOTP_REVENUE_THRESHOLD = 20.0   # 第二业务收入占比 ≥ 20%
SOTP_PROFIT_THRESHOLD = 15.0    # 第二业务利润贡献 ≥ 15%
SOTP_GROWTH_DIFF = 20.0         # 两业务增速差 ≥ 20%
SOTP_ROE_DIFF = 10.0            # 两业务 ROE 差 ≥ 10%


def _check_sotp_trigger(
    segments: list[BusinessSegment],
) -> tuple[bool, list[str]]:
    """
    Return (triggered, list of trigger reasons).
    SOP 三 SOTP 量化触发条件:
      条件1: 第二业务收入占比 ≥ 20%
      条件2: 第二业务利润贡献 ≥ 15%
      条件3: 两业务增速差 ≥ 20%
      条件4: 两业务 ROE 差 ≥ 10%
    满足任一即触发。
    """
    if len(segments) < 2:
        return False, ["仅一个业务板块，不适用SOTP"]

    # Sort by revenue share descending — top 2 are the two largest
    sorted_by_rev = sorted(segments, key=lambda s: s.revenue_share, reverse=True)
    a, b = sorted_by_rev[0], sorted_by_rev[1]

    reasons = []

    # Condition 1: revenue share
    if b.revenue_share >= SOTP_REVENUE_THRESHOLD:
        reasons.append(
            f"条件1触发: {b.name}收入占比{b.revenue_share:.1f}% ≥ {SOTP_REVENUE_THRESHOLD}%"
        )

    # Condition 2: profit share
    if b.profit_share >= SOTP_PROFIT_THRESHOLD:
        reasons.append(
            f"条件2触发: {b.name}利润贡献{b.profit_share:.1f}% ≥ {SOTP_PROFIT_THRESHOLD}%"
        )

    # Condition 3: growth rate delta
    growth_delta = abs(a.yoy_growth - b.yoy_growth)
    if growth_delta >= SOTP_GROWTH_DIFF:
        reasons.append(
            f"条件3触发: {a.name}({a.yoy_growth:+.1f}%) vs "
            f"{b.name}({b.yoy_growth:+.1f}%), 增速差{growth_delta:.1f}% ≥ {SOTP_GROWTH_DIFF}%"
        )

    # Condition 4: ROE delta
    if a.roe > 0 and b.roe > 0:
        roe_delta = abs(a.roe - b.roe)
        if roe_delta >= SOTP_ROE_DIFF:
            reasons.append(
                f"条件4触发: {a.name}(ROE {a.roe:.1f}%) vs "
                f"{b.name}(ROE {b.roe:.1f}%), ROE差{roe_delta:.1f}% ≥ {SOTP_ROE_DIFF}%"
            )

    if not reasons:
        # Check if any segment has a fundamentally different valuation paradigm
        # (heuristic: flag if segments span different SOP branches)
        branches = {s.branch for s in segments if s.branch}
        if len(branches) >= 2:
            reasons.append(
                f"定性触发: 业务板块跨越多个SOP分支 ({', '.join(branches)})，"
                "估值体系脱节，建议SOTP"
            )

    return len(reasons) > 0, reasons


# ============================================================
#  Macro Beta Context + PE 乘数
# ============================================================

MACRO_MULTIPLIER = {
    "进攻防线": 1.12,
    "防御防线": 0.92,
}

MACRO_CONTEXT_TEMPLATE = """【宏观经济防线 — {macro_beta}】

{macro_guidance}
请在你的估值倍数选择中体现该宏观环境的影响。"""

MACRO_GUIDANCE = {
    "进攻防线": (
        "当前处于进攻防线——流动性充裕，市场风险偏好上行。\n"
        "对于放量驱动型（分支B）高壁垒新材料公司：\n"
        "  - 市场在进攻防线中给予「技术独占期」成长股 PE 溢价是常态\n"
        "  - PEG 1.2-1.8 在渗透率爆发期属于合理范围，不自动视为高估\n"
        "  - 国产替代/供应链安全逻辑在进攻防线中获得更高的估值容忍度\n"
        "请在估值时考虑进攻防线下的合理溢价，不要系统性低估成长性。"
    ),
    "防御防线": (
        "当前处于防御防线——流动性收紧，市场风险偏好下行。\n"
        "估值应以安全边际为核心：\n"
        "  - 更加重视净现比和资产负债率等硬指标\n"
        "  - PE 倍数应偏向保守端，PEG 建议 ≤ 1.0\n"
        "  - 类债券（分支C1）和 PB 底部锚（分支A-PB）框架更具防御价值\n"
        "请在估值时优先考虑下行保护。"
    ),
}


def _build_macro_context(state: PipelineState) -> str:
    """Build macro-environment guidance for the valuation prompt."""
    macro_beta = state.get("macro_beta", "进攻防线")
    guidance = MACRO_GUIDANCE.get(macro_beta, MACRO_GUIDANCE["进攻防线"])
    return MACRO_CONTEXT_TEMPLATE.format(
        macro_beta=macro_beta,
        macro_guidance=guidance,
    )


def _get_macro_multiplier(state: PipelineState) -> float:
    """Return the calibration multiplier for the current macro environment."""
    macro_beta = state.get("macro_beta", "进攻防线")
    return MACRO_MULTIPLIER.get(macro_beta, 1.0)


def _hard_sotp_check(state: PipelineState, segments: list) -> list[str]:
    """
    硬编码红线①: 多主业 / LNG / 半导体材料 → 强制 SOTP

    直接在 node 内执行（非条件边），确保状态修改被 LangGraph 持久化。
    """
    reasons = []
    research_ctx = state.get("research_context", "")
    industry = state.get("industry", "")
    stock_name = state.get("stock_name", "")

    # Rule A: 板块数量 > 1
    if len(segments) > 1:
        reasons.append(
            f"硬编码红线A: {len(segments)} 个独立业务板块，禁止整体法估值"
        )

    # Rule B: LNG / 传统业务关键词
    lng_keywords = ["LNG", "液化天然气", "保温绝热", "造船", "运输",
                    "化工", "工程", "环保", "地产", "贸易"]
    for seg in segments:
        seg_name = seg.name if hasattr(seg, 'name') else str(seg)
        for kw in lng_keywords:
            if kw in seg_name and seg.branch not in ("分支B", ""):
                reasons.append(
                    f"硬编码红线B: 板块「{seg_name}」含非科技关键词「{kw}」，"
                    "传统业务与高科技板块估值体系脱节，强制SOTP"
                )
                break
        if reasons:
            break

    # Rule C: 科技+传统双域信号
    if not reasons:
        all_text = f"{research_ctx} {industry} {stock_name}"
        tech_domain = ["前驱体", "光刻胶", "半导体材料", "电子特气", "HBM",
                      "KrF", "ArF", "芯片", "AI", "算力", "国产替代"]
        traditional_domain = ["LNG", "液化天然气", "保温", "绝热", "造船",
                              "化工", "工程", "环保", "贸易", "地产", "水泥"]

        tech_hits = [kw for kw in tech_domain if kw in all_text]
        trad_hits = [kw for kw in traditional_domain if kw in all_text]

        if tech_hits and trad_hits:
            reasons.append(
                f"硬编码红线C: 科技域{tech_hits[:2]}+传统域{trad_hits[:2]}并存，"
                "估值体系脱节，强制SOTP"
            )
        elif len(tech_hits) >= 2 and len(segments) == 1:
            reasons.append(
                f"硬编码红线C2: 合并口径下检测多科技信号{tech_hits[:3]}，"
                "保守强制SOTP防止单一板块遮蔽多元化结构"
            )

    # Rule D: 半导体材料行业 + 合并口径 → 必为 akshare 未拆分
    if not reasons and len(segments) == 1:
        seg_name = segments[0].name if hasattr(segments[0], 'name') else str(segments[0])
        if "合并" in seg_name and industry in (
            "半导体材料", "半导体", "电子材料", "新材料",
            "军工配套", "高端制造", "医药", "医疗",
        ):
            reasons.append(
                f"硬编码红线D: 行业「{industry}」仅合并口径单板块，"
                "半导体/高端制造企业必然多产品线，强制LLM分解SOTP"
            )

    if reasons:
        print(f"  🔒 [硬编码红线①] {'; '.join(reasons[:2])}")

    return reasons


# ============================================================
#  v3.3: Pharma/Biotech detection for Branch E PS fallback
# ============================================================

PHARMA_INDUSTRY_KEYWORDS = [
    "医药", "医疗", "生物制药", "创新药", "biotech", "Biotech",
    "生物科技", "制药", "中药", "化学药",
]

PHARMA_RESEARCH_KEYWORDS = [
    "创新药", "生物制药", "biotech", "Biotech", "单抗", "双抗", "ADC药物",
    "基因治疗", "靶向药", "CAR-T", "mRNA", "细胞治疗", "基因编辑",
    "PD-1", "PD-L1", "BTK抑制剂", "PROTAC", "临床Ⅲ期",
    "FDA突破性疗法", "孤儿药", "NDA", "BLA", "IND",
]

PHARMA_KNOWN_COMPANIES = {
    "百济神州", "信达生物", "荣昌生物", "君实生物", "康希诺",
    "科伦博泰", "传奇生物", "和黄医药", "再鼎医药", "康方生物",
    "诺诚健华", "亚盛医药", "恒瑞医药", "翰森制药", "石药集团",
    "中国生物制药", "药明康德", "药明生物", "泰格医药", "凯莱英",
    "康龙化成", "昭衍新药",
}


def _is_pharma_biotech(state: PipelineState) -> bool:
    """Detect if the stock is an innovative pharma/biotech requiring PS valuation."""
    industry = state.get("industry", "")
    stock_name = state.get("stock_name", "")
    research_ctx = state.get("research_context", "")
    exemption_reasons = state.get("exemption_reasons", [])

    # Check exemption reasons for pharma-specific indicators (from node0_quick_screen)
    # "v3.3豁免" is intentionally excluded — the YAML Tier 0 registry applies it to
    # ALL strategic assets (semiconductor + pharma), so it cannot discriminate.
    for reason in exemption_reasons:
        if "创新药" in reason or "管线驱动" in reason or "Biotech" in reason:
            return True

    # Check known pharma companies
    if any(c in stock_name for c in PHARMA_KNOWN_COMPANIES):
        return True

    # Check industry
    if any(kw in industry for kw in PHARMA_INDUSTRY_KEYWORDS):
        return True

    # Check research context for pharma keywords
    corpus = f"{stock_name} {research_ctx}".lower()
    if any(kw.lower() in corpus for kw in PHARMA_RESEARCH_KEYWORDS):
        return True

    return False


# ============================================================
#  Step B — Valuation LLM Call
# ============================================================

def _build_labels_summary(state: PipelineState) -> str:
    return (
        f"利润驱动力(L1): {state.get('profit_driver', '')}\n"
        f"渗透率阶段(L2): {state.get('penetration_stage', '')}\n"
        f"固/总: {state['financials'].fa_to_ta:.1%} (重资产={state.get('fixed_asset_heavy', False)})\n"
        f"ASP趋势: {state.get('competition_asp_trend', '')}\n"
        f"份额趋势: {state.get('market_share_trend', '')}\n"
        f"PEG调整系数: {state.get('peg_adjustment', 1.0)}\n"
        f"卷度判断: {state.get('competition_verdict', '')}"
    )


def _build_company_data(state: PipelineState) -> str:
    fin = state["financials"]
    segments = state.get("segments", [])
    lines = [
        f"股票: {state['stock_name']} ({state['stock_code']})",
        f"股价: {state.get('current_price', 'N/A')} 元",
        f"总股本: {state.get('total_shares', 'N/A')} 亿股",
        f"近三年营收: {fin.revenue}",
        f"近三年归母净利润: {fin.net_profit_parent}",
        f"净现比3Y均值: {fin.cash_ratio_3y_avg:.2f}",
        f"资产负债率: {fin.debt_ratio:.1f}%",
        "",
    ]
    for s in segments:
        lines.append(
            f"板块-{s.name}: 收入{s.revenue:.1f}亿({s.revenue_share:.0f}%), "
            f"利润贡献{s.profit_contribution:.1f}亿({s.profit_share:.0f}%), "
            f"毛利率{s.gross_margin:.1f}%, 增速{s.yoy_growth:+.0f}%, "
            f"SOP分支={s.branch or '待定'}"
        )
    return "\n".join(lines)


def node2_valuation(state: PipelineState) -> PipelineState:
    """
    1. Run SOTP trigger math
    2. Select the right valuation prompt
    3. Call LLM to produce structured valuation output
    """
    segments = state.get("segments", [])

    # —— Step A: SOTP trigger check ——
    triggered, reasons = _check_sotp_trigger(segments)

    # ═══════════════════════════════════════════════════════════
    #  硬编码红线①: 多主业 / LNG / 半导体材料 → 强制SOTP
    #  (在 node 内执行而非条件边，确保状态持久化)
    # ═══════════════════════════════════════════════════════════
    if not triggered:
        hard_reasons = _hard_sotp_check(state, segments)
        if hard_reasons:
            triggered = True
            reasons.extend(hard_reasons)

    state["sotp_triggered"] = triggered
    # 单板块时标记走 LLM 分解 SOTP 路径
    if triggered and len(segments) <= 1:
        state["sotp_force_single_path"] = True

    # —— Step B: Select prompt ——
    verdict = state.get("screening_verdict", "")
    # v3.3: Pharma/Biotech detection for Branch E PS fallback
    is_pharma = _is_pharma_biotech(state) if verdict == ScreeningVerdict.BRANCH_E.value else False
    if is_pharma:
        prompt_template = BRANCH_E_PHARMA_VALUATION_PROMPT
    elif verdict == ScreeningVerdict.BRANCH_E.value:
        prompt_template = BRANCH_E_VALUATION_PROMPT
    elif verdict == "熔断→分支D":
        prompt_template = BRANCH_D_VALUATION_PROMPT
    elif triggered or state.get("sotp_force_single_path"):
        prompt_template = SOTP_VALUATION_PROMPT
        if state.get("sotp_force_single_path"):
            state["sotp_triggered"] = True
    else:
        prompt_template = SINGLE_FRAMEWORK_VALUATION_PROMPT

    company_data = _build_company_data(state)
    labels = _build_labels_summary(state)
    macro_context = _build_macro_context(state)

    # Branch D has no {macro_context} or {sotp_reasons} placeholder
    is_branch_d = verdict == "熔断→分支D"
    # Branch E (both hard-tech and pharma) has {company_data} and {labels} only
    is_branch_e = verdict == ScreeningVerdict.BRANCH_E.value
    if is_branch_d:
        prompt = prompt_template.format(company_data=company_data)
    elif is_branch_e:
        prompt = prompt_template.format(company_data=company_data, labels=labels)
    else:
        prompt = prompt_template.format(
            company_data=company_data,
            labels=labels,
            macro_context=macro_context,
            sotp_reasons="\n".join(reasons) if triggered else "SOTP未触发",
        )

    # ═══════════════════════════════════════════════════════════
    #  v3.3: 动态注入战略资产豁免上下文
    # ═══════════════════════════════════════════════════════════
    strategic_ctx = get_strategic_context(state.get("stock_code", ""))
    if strategic_ctx:
        prompt = strategic_ctx + "\n\n" + prompt
        print(f"  🔥 [战略上下文] 已向估值Agent注入战略豁免通知（{state['stock_code']}）")

    if is_pharma:
        print(f"  💊 [v3.3] 医药/Biotech PS估值Fallback已激活 — 禁止PE/PEG，强制动态PS+管线溢价")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200,
        )
        content = response.choices[0].message.content.strip()
        result = extract_json(content)
    except Exception as e:
        state["error"] = f"Node 2 LLM call failed: {e}"
        return state

    # —— Step C: Populate valuation conclusion ——
    # Branch E (both variants) and Branch D are macro-insensitive
    is_macro_sensitive = not (is_branch_d or is_branch_e)
    multiplier = _get_macro_multiplier(state) if is_macro_sensitive else 1.0
    if triggered and not is_pharma:
        sotp_segments = []
        total_val = 0.0
        for seg in result.get("segments", []):
            seg_obj = BusinessSegment(
                name=seg.get("name", ""),
                revenue=0,
                revenue_share=0,
                profit_contribution=seg.get("net_profit_2026e", 0),
                profit_share=0,
                gross_margin=0,
                yoy_growth=0,
                branch=seg.get("branch", ""),
                comparable_pe=seg.get("pe_assigned", 0),
            )
            sotp_segments.append(seg_obj)
            total_val += seg.get("segment_value", 0)

        state["sotp"] = SOTPResult(
            triggered=True,
            trigger_reasons=reasons,
            segments=sotp_segments,
            total_valuation=round(result.get("total_value", total_val) * multiplier, 1),
            implied_price=round(result.get("implied_price", 0) * multiplier, 2),
            composite_pe=result.get("composite_pe_2026e", 0),
        )
    elif is_pharma:
        # v3.3: Pharma SOTP is not applicable — single PS+pipeline framework
        state["sotp"] = SOTPResult(triggered=False, trigger_reasons=[])
        # Store pharma-specific details for audit
        state["pharma_valuation_details"] = {
            "ps_baseline": result.get("ps_baseline", 0),
            "pipeline_premium_pct": result.get("pipeline_premium_pct", 0),
            "ps_final": result.get("ps_final", 0),
            "pipeline_valuation_addon": result.get("pipeline_valuation_addon", 0),
            "comparable_companies": result.get("comparable_companies", []),
        }
    else:
        state["sotp"] = SOTPResult(triggered=False, trigger_reasons=reasons)

    # v3.3: Pharma risks are dicts — convert to strings for ValuationConclusion
    raw_risks = result.get("key_risks", result.get("key_pharma_risks", []))
    if is_pharma and raw_risks and isinstance(raw_risks[0], dict):
        key_risks = [
            f"[{r.get('probability', '?')}] {r.get('risk', '')} (PS影响: {r.get('ps_impact', '?')})"
            for r in raw_risks
        ]
    else:
        key_risks = raw_risks

    # Pharma uses implied_valuation/implied_price; others use base_case_value/base_case_price
    if is_pharma:
        base_val = result.get("implied_valuation", 0)
        base_price = result.get("implied_price", 0)
    else:
        base_val = result.get(
            "base_case_value",
            result.get("total_value", result.get("distressed_value", 0)),
        )
        base_price = result.get(
            "base_case_price",
            result.get("implied_price", result.get("distressed_price", 0)),
        )

    state["valuation"] = ValuationConclusion(
        framework=result.get("framework", ""),
        sotp=state.get("sotp"),
        base_case_value=round(base_val * multiplier, 1),
        base_case_price=round(base_price * multiplier, 2),
        bull_case_price=round(result.get("bull_case_price", 0) * multiplier, 2),
        bear_case_price=round(result.get("bear_case_price", 0) * multiplier, 2),
        key_risks=key_risks,
        recommendation=result.get("recommendation", ""),
    )

    if is_pharma:
        ps_info = state.get("pharma_valuation_details", {})
        print(f"  💊 PS基线: {ps_info.get('ps_baseline', '?')}x | "
              f"管线溢价: +{ps_info.get('pipeline_premium_pct', 0):.0%} | "
              f"最终PS: {ps_info.get('ps_final', '?')}x | "
              f"管线加总: {ps_info.get('pipeline_valuation_addon', 0):.1f}亿")

    return state
