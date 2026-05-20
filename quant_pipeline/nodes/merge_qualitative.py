"""
Merge Qualitative Node — L1/L2/L3 交叉验证 & 冲突裁决
=========================================================
After L1, L2, L3 agents complete in parallel, this node:
1. Reads all three raw outputs
2. Cross-validates for SOP framework consistency (6 mutual-exclusion rules)
3. Resolves conflicts (or flags for human review)
4. Writes the final unified labels to state

This is the KEY LangGraph value-add: independent agent outputs
are compared and validated before being used downstream.

INVEST_SOP mutual-exclusion rules (from INVEST_SOP.md):
  RULE_1: 红利/稳定驱动 vs 渗透率爆发期互斥
          C1/C2 稳定型标的不应有渗透率概念
  RULE_2: 周期涨价驱动 vs 高轻资产净现比互斥
          涨价周期品通常重资产+低净现比；轻资产+高净现比更接近稳定型
  RULE_3: 放量驱动 vs 成熟估值框架互斥
          放量型不应建议使用成熟PE+股息率或DCF/DDM
  RULE_4: ASP风控红线 (来自 INVEST_SOP 分支B定量红线)
          ASP年降>15%且份额下滑 → PEG强制≤0.7
  RULE_5: 渗透率>30% + 放量驱动互斥
          >30%是成熟期，应与放量驱动逻辑脱钩
  RULE_6: 涨价驱动 + 建议框架错配
          涨价驱动+重资产 → PB-ROE；涨价驱动+轻资产 → PE/PEG
"""

import datetime
from state import PipelineState


# ═══════════════════════════════════════════════════════════════
#  Cross-validation Rules (SOP Framework Consistency)
# ═══════════════════════════════════════════════════════════════

def _validate_consistency(state: PipelineState) -> tuple[bool, list[str], list[str]]:
    """
    Validate that L1, L2, L3 outputs are internally consistent with INVEST_SOP.

    Returns (is_consistent, conflict_descriptions, rules_fired).
    Each conflict is tagged with the RULE_X that triggered it.
    """
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})
    l3 = state.get("l3_output", {})
    fin = state.get("financials")

    conflicts = []
    rules_fired = []

    pd = l1.get("profit_driver", "")
    ps = l2.get("penetration_stage", "")
    asp = l3.get("asp_trend", "")
    share = l3.get("market_share_trend", "")
    peg = l3.get("peg_adjustment", 1.0)
    ha = l2.get("heavy_asset", False)
    framework = l2.get("suggested_framework", "")
    cash_ratio = getattr(fin, "cash_ratio_3y_avg", 0.0) if fin else 0.0
    fa_to_ta = getattr(fin, "fa_to_ta", 0.0) if fin else 0.0
    l1_confidence = l1.get("confidence", 0.0)
    l1_primary = l1.get("primary_segment", "")

    # ═══════════════════════════════════════════════════════════
    #  RULE_0: Agent 自身置信度不足
    # ═══════════════════════════════════════════════════════════
    if l1_confidence < 0.3:
        conflicts.append(
            f"[RULE_0] L1 Agent自身置信度仅{l1_confidence:.0%}——"
            f"利润驱动力分类不确定，下游估值框架选择需谨慎。"
        )
        rules_fired.append("RULE_0")

    # ═══════════════════════════════════════════════════════════
    #  RULE_1: 红利/稳定驱动 vs 渗透率爆发期互斥
    #
    #  INVEST_SOP 核心原则: C1(稳定分红)/C2(离散事件) 类标的没有「渗透率」概念。
    #  渗透率属于分支B(放量驱动)的专属维度。如果L1判C1/C2，
    #  L2不应给出5-30%爆发期或<5%导入期。
    # ═══════════════════════════════════════════════════════════
    if pd in ("稳定分红(C1)", "离散事件(C2)") and ps != "":
        conflicts.append(
            f"[RULE_1] L1判「{pd}」(分支C1/C2)但L2判「渗透率{ps}」。"
            f"稳定分红/离散事件型标的（类债券/管线/重组）不存在渗透率概念，"
            f"渗透率仅适用于放量驱动(分支B)。建议强制清空渗透率阶段。"
        )
        rules_fired.append("RULE_1")

    # ═══════════════════════════════════════════════════════════
    #  RULE_2: 周期涨价驱动 vs 高轻资产净现比互斥
    #
    #  INVEST_SOP 前置快筛: 涨价驱动的周期品(分支A)通常伴随重资产
    #  (固/总>40%)和较低的净现比。若数据呈现「轻资产+高净现比」
    #  模式，更接近稳定型(C1)特征，L1可能误判了利润驱动力。
    #
    #  触发条件: pd=涨价驱动 AND ha=false(轻资产) AND cash_ratio>1.0
    #  → 轻资产+高现金流 → 不像周期品 → 可能误判
    # ═══════════════════════════════════════════════════════════
    if pd == "涨价驱动" and not ha and cash_ratio > 1.0:
        conflicts.append(
            f"[RULE_2] L1判「涨价驱动」但L2判轻资产(固/总={fa_to_ta:.1%})"
            f"且净现比3Y均值={cash_ratio:.2f}>1.0。"
            f"涨价驱动型周期品通常呈现重资产+低净现比特征；"
            f"当前数据更接近稳定型(C1)或放量型(B)的财务画像，"
            f"L1利润驱动力分类可能偏误。建议重审L1判断。"
        )
        rules_fired.append("RULE_2")

    # 增强版本：涨价驱动 + 重资产 + 净现比极低 → 确认周期品
    # 但如果框架建议不使用PB-ROE，提示可能遗漏
    if pd == "涨价驱动" and ha and framework and "PB-ROE" not in framework and "PE/PEG" in framework:
        # 涨价驱动+重资产的标准框架是PB-ROE，使用PE/PEG可能不合适
        conflicts.append(
            f"[RULE_2b] L1判「涨价驱动」+ L2判重资产(固/总={fa_to_ta:.1%})，"
            f"但建议框架={framework}。涨价驱动+重资产的标准框架应为PB-ROE底部锚，"
            f"使用PE/PEG弹性锚可能高估周期底部的安全边际。"
        )
        rules_fired.append("RULE_2b")

    # ═══════════════════════════════════════════════════════════
    #  RULE_3: 放量驱动 vs 成熟估值框架互斥
    #
    #  INVEST_SOP 分支B: 放量驱动型标的使用 PEG+S曲线+SOTP 框架。
    #  如果L2建议使用「成熟PE+股息率」或「DCF/DDM」，这表明L2认为
    #  标的已进入成熟期，与L1的「放量驱动」判断矛盾。
    #  唯一例外：放量驱动+渗透率>30% → 确实应该切换到成熟期框架
    # ═══════════════════════════════════════════════════════════
    mature_frameworks = ["成熟PE+股息率", "DCF/DDM类债券"]
    if pd == "放量驱动" and any(f in framework for f in mature_frameworks):
        # Check if penetration stage justifies mature framework
        if ">30%" not in ps:
            conflicts.append(
                f"[RULE_3] L1判「放量驱动」但L2建议框架={framework}。"
                f"放量驱动型标的应使用PEG+S曲线+SOTP等成长框架；"
                f"成熟PE+股息率或DCF/DDM仅适用于渗透率>30%或稳定型标的。"
                f"当前渗透率={ps or '未给出'}，框架选择与驱动力矛盾。"
            )
            rules_fired.append("RULE_3")

    # ═══════════════════════════════════════════════════════════
    #  RULE_4: ASP风控红线 (INVEST_SOP 分支B 定量红线)
    #
    #  "核心产品ASP年化降幅>15%且份额下滑 → PEG强行打7折"
    #  兼容新旧两套 ASP 命名:
    #    - 新 (ASPTrend): "恶性崩塌(>15%)" "正常温和(<10%)" "稳定/涨价"
    #    - 旧 (legacy):   "年降>15%" "年降<10%" "稳定" "上行"
    # ═══════════════════════════════════════════════════════════
    _ASP_COLLAPSE = {"年降>15%", "恶性崩塌(>15%)"}
    _ASP_MILD = {"年降<10%", "正常温和(<10%)", "年降10-15%"}

    if asp in _ASP_COLLAPSE and share == "下滑":
        if peg > 0.7:
            conflicts.append(
                f"[RULE_4] L3判ASP恶性崩塌({asp})+份额下滑（双红灯条件），"
                f"但PEG调整={peg}。根据INVEST_SOP分支B定量风控红线："
                f"「ASP年化降幅>15%且份额下滑 → PEG强行打7折」。"
                f"当前PEG未触发强制打折，L3的ASP/份额判断与PEG调整自相矛盾。"
            )
            rules_fired.append("RULE_4")
    elif asp in _ASP_COLLAPSE and peg > 0.85:
        conflicts.append(
            f"[RULE_4b] L3判ASP恶性崩塌({asp})，红色预警，"
            f"但PEG调整={peg}>0.85。根据INVEST_SOP风控红线："
            f"ASP年降>15%即触发红色预警，PEG应≤0.85。"
        )
        rules_fired.append("RULE_4b")

    # ═══════════════════════════════════════════════════════════
    #  RULE_5: 渗透率>30% + 放量驱动 互斥
    #
    #  INVEST_SOP 分支B定义: 渗透率>30% 进入成熟期，应从放量逻辑
    #  切换到成熟PE+股息率逻辑。如果L1坚持放量驱动但L2判>30%，
    #  说明该标的已过爆发期，L1的利润驱动力判断需要下调。
    # ═══════════════════════════════════════════════════════════
    if pd == "放量驱动" and ">30%" in ps:
        conflicts.append(
            f"[RULE_5] L1判「放量驱动」但L2判「渗透率>30%成熟期」。"
            f"根据INVEST_SOP：渗透率>30%意味着标的已从爆发期进入成熟期，"
            f"增速放缓、份额战开启、应从PEG切换到成熟PE+股息率框架。"
            f"L1的「放量驱动」标签可能滞后——该标的的放量红利已接近尾声。"
        )
        rules_fired.append("RULE_5")

    # ═══════════════════════════════════════════════════════════
    #  RULE_6: 涨价驱动 + 渗透率非空（渗透率维度误用到周期品）
    #
    #  与 RULE_1 对称：涨价驱动型不应该有渗透率概念。
    #  周期品的估值锚是供需平衡表和CAPEX周期，不是渗透率S曲线。
    # ═══════════════════════════════════════════════════════════
    if pd == "涨价驱动" and ps != "":
        conflicts.append(
            f"[RULE_6] L1判「涨价驱动」(分支A·周期品)但L2判「渗透率{ps}」。"
            f"渗透率是放量驱动(分支B)的专属维度，周期品的估值锚是供需平衡表"
            f"和CAPEX周期，与渗透率无关。建议强制清空渗透率阶段，"
            f"将L2分析重心转向供给侧CAPEX状态和产能利用率。"
        )
        rules_fired.append("RULE_6")

    # ═══════════════════════════════════════════════════════════
    #  RULE_7: 治理红旗不容协商
    # ═══════════════════════════════════════════════════════════
    gov_flags = l3.get("governance_flags", [])
    if gov_flags:
        FATAL_KEYWORDS = ["财务造假", "立案调查", "审计非标", "减持"]
        fatal_flags = [f for f in gov_flags if any(kw in f for kw in FATAL_KEYWORDS)]
        if fatal_flags:
            conflicts.append(
                f"[RULE_7] L3发现治理红旗: {', '.join(fatal_flags[:3])}。"
                f"根据INVEST_SOP前置快筛Q3，任一命中即一票否决。"
                f"该标的应直接放弃，不进任何估值框架。"
            )
            rules_fired.append("RULE_7")

    return len(conflicts) == 0, conflicts, rules_fired


def _resolve_labels(state: PipelineState, conflicts: list[str]) -> PipelineState:
    """
    Write final labels to state, overriding agent outputs when conflicts exist.

    Resolution strategy (ordered by priority):
      1. RULE_7 (governance red flag) → force ABANDON verdict
      2. RULE_1/6 (profit driver vs penetration mismatch) → clear penetration
      3. RULE_5 (放量驱动+>30%) → downgrade profit driver or switch framework
      4. RULE_4 (ASP red line) → force PEG haircut
      5. RULE_2 (涨价驱动+light asset) → flag but don't override (needs human)
      6. RULE_3 (放量+成熟框架) → correct framework
    """
    l1 = state.get("l1_output", {})
    l2 = state.get("l2_output", {})
    l3 = state.get("l3_output", {})

    pd = l1.get("profit_driver", "涨价驱动")
    ps = l2.get("penetration_stage", "")
    ha = l2.get("heavy_asset", state.get("financials").fa_to_ta > 0.40 if state.get("financials") else False)
    framework = l2.get("suggested_framework", "")

    # ——— Resolution: RULE_1 / RULE_6 — penetration is invalid for non-放量驱动 ———
    if pd != "放量驱动" and ps != "":
        ps = ""
        # If 涨价驱动+重资产 → force PB-ROE
        if pd == "涨价驱动" and ha:
            framework = "PB-ROE底部锚"
        elif pd == "涨价驱动" and not ha:
            framework = "PE/PEG弹性锚"

    # ——— Resolution: RULE_5 — 放量驱动+>30% → downgrade ———
    if pd == "放量驱动" and ">30%" in ps:
        # Don't change profit_driver (it's still 放量 at core),
        # but fix the framework to mature PE
        if "成熟" not in framework:
            framework = "成熟PE+股息率"

    # ——— Resolution: RULE_4 — force PEG haircut for ASP red line ———
    asp = l3.get("asp_trend", "稳定")
    share = l3.get("market_share_trend", "稳定")
    peg = l3.get("peg_adjustment", 1.0)
    _ASP_COLLAPSE = {"年降>15%", "恶性崩塌(>15%)"}

    if asp in _ASP_COLLAPSE and share == "下滑":
        peg = min(peg, 0.7)
    elif asp in _ASP_COLLAPSE:
        peg = min(peg, 0.85)

    # ——— RULE_7: governance fatal → force ABANDON ———
    gov_flags = l3.get("governance_flags", [])
    fatal_keywords = ["财务造假", "立案调查", "审计非标"]
    if any(any(kw in f for kw in fatal_keywords) for f in gov_flags):
        state["screening_verdict"] = "熔断→放弃"
        state["error"] = f"治理红灯触发 {[f for f in gov_flags if any(kw in f for kw in fatal_keywords)]}"

    # ——— Extract new v3.2 fields from L3 output ———
    capex_status = l3.get("capex_status", "N/A")
    confirmed_framework = l3.get("confirmed_framework", "")

    # ——— Write resolved labels ———
    state["profit_driver"] = pd
    state["penetration_stage"] = ps
    state["fixed_asset_heavy"] = ha
    state["competition_asp_trend"] = asp       # legacy field
    state["asp_trend"] = asp                    # v3.2 field (ASPTrend enum)
    state["market_share_trend"] = share
    state["capex_status"] = capex_status        # v3.2 field (CapexStatus enum)
    state["peg_adjustment"] = peg
    state["competition_verdict"] = l3.get("competition_verdict", "")
    state["governance_flags"] = gov_flags
    state["suggested_framework"] = framework
    # If L3 provided a confirmed framework, prefer it over L2's suggestion
    if confirmed_framework and confirmed_framework in {
        "PB-ROE底部锚", "PE/PEG弹性锚", "PS+管线估值",
        "PEG+S曲线+SOTP", "成熟PE+股息率", "DCF/DDM类债券", "概率加权/rNPV",
    }:
        state["suggested_framework"] = confirmed_framework

    # ═══════════════════════════════════════════════════════════
    #  硬编码红线②: 渗透率阶段强制回填
    #  防止 L2 Agent 未识别导致下游估值节点状态断流
    # ═══════════════════════════════════════════════════════════
    _hard_fix_penetration_stage(state)

    return state


def _hard_fix_penetration_stage(state: PipelineState) -> None:
    """
    硬编码红线②: L2 渗透率阶段强制补齐。

    触发条件: penetration_stage 为空 / 缺失 / 无意义的占位符。
    规则:
      - 放量驱动 + 半导体材料/国产替代/新材料 → "5-30%爆发期"
      - 放量驱动 + 其他行业           → "5-30%爆发期" (保守默认)
      - 涨价驱动 + 重资产              → ""（周期品无需渗透率）
      - 稳定分红(C1)/离散事件(C2)       → ""（类债券/管线无需渗透率）
    """
    ps = state.get("penetration_stage", "")
    pd = state.get("profit_driver", "")

    # 定义无意义的占位符
    null_values = {"", "不适用", "None", "****", "N/A", "无", "—", "-"}

    if ps.strip() not in null_values and ps != "":
        return  # 已有有效值，无需干预

    if pd == "放量驱动":
        # 放量驱动型标的必须有渗透率阶段，强制回填默认值
        industry = state.get("industry", "")
        segment_names = [
            s.name if hasattr(s, 'name') else str(s)
            for s in state.get("segments", [])
        ]
        all_text = f"{industry} {' '.join(segment_names)}"

        tech_keywords = [
            "半导体", "前驱体", "光刻胶", "HBM", "high-k", "芯片",
            "国产替代", "新材料", "碳化硅", "氮化镓", "KrF", "ArF",
            "锗", "镓", "铟", "稀土",
        ]

        if any(kw in all_text for kw in tech_keywords):
            state["penetration_stage"] = "5-30%爆发期"
            print(f"  🔒 [硬编码红线②] 渗透率缺失，基于国产替代/新材料放量逻辑，强制回填: 5-30%爆发期")
        else:
            state["penetration_stage"] = "5-30%爆发期"
            print(f"  🔒 [硬编码红线②] 渗透率缺失，放量驱动默认回填: 5-30%爆发期")

    elif pd == "涨价驱动":
        # 周期品不需要渗透率，保持空值
        pass

    elif pd in ("稳定分红(C1)", "离散事件(C2)"):
        # 类债券/管线驱动不需要渗透率，保持空值
        pass


def merge_qualitative(state: PipelineState) -> PipelineState:
    """
    Merge node: wait for all three agents (L1, L2, L3) to complete,
    cross-validate their outputs, and write unified labels to state.

    This is the fan-in point after the L1/L2/L3 parallel dispatch.
    """
    l1_ok = bool(state.get("l1_output"))
    l2_ok = bool(state.get("l2_output"))
    l3_ok = bool(state.get("l3_output"))

    if not (l1_ok and l2_ok and l3_ok):
        missing = []
        if not l1_ok: missing.append("L1")
        if not l2_ok: missing.append("L2")
        if not l3_ok: missing.append("L3")
        state["error"] = f"Merge failed: agents not complete: {missing}"
        state["label_consensus"] = False
        state["label_conflicts"] = [f"Agent(s) {', '.join(missing)} did not produce output"]
        state["resolved_by"] = []
        return state

    # Run cross-validation
    is_consistent, conflicts, rules_fired = _validate_consistency(state)
    state = _resolve_labels(state, conflicts)

    # Store validation results
    state["label_consensus"] = is_consistent
    state["label_conflicts"] = conflicts
    state["resolved_by"] = rules_fired

    # Merge audit
    state["merge_audit"] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "rules_fired": rules_fired,
        "conflicts_found": len(conflicts),
        "verdict": "PASS" if is_consistent else "CONFLICT",
    }

    if not is_consistent:
        print(f"  ⚠ [Merge] L1/L2/L3 交叉验证发现 {len(conflicts)} 个冲突 ({', '.join(rules_fired)}):")
        for c in conflicts:
            print(f"    - {c[:150]}")
    else:
        print(f"  ✓ [Merge] L1/L2/L3 三 Agent 一致通过交叉验证 (0 冲突)")

    return state
