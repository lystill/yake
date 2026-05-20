"""
Node 3 — CIO 冲突调和与反思 (v3.1 双轨制)
=============================================
CIO (Chief Investment Officer) reviews the valuation output
and reconciles conflicts between two competing paradigms:

  1. 经典古典价值（清算底裤）— pure fundamental, no narrative premium
  2. 动态科技叙事（多头弹性）— moat premium for high-barrier segments

Decision logic (deterministic pre-processing + LLM synthesis):
  - 整体法 high valuation → force red team risk discount 10-15%
  - SOTP low valuation → allow narrative premium up to +15% for core moat segments
  - Output JSON: cio_verdict, narrative_premium_rate, classical_price, narrative_price

Also handles the reflection loop: if CIO-adjusted deviation still >30%,
triggers a loop-back through node_red_team → dispatch_parallel_qualitative.
"""

import json
import litellm
from state import PipelineState
from utils.config import get_model

MAX_REFLECTION_ROUNDS = 2

# ═══════════════════════════════════════════════════════════════
#  CIO System Prompt — 投资总监冲突调和
# ═══════════════════════════════════════════════════════════════

CIO_SYSTEM_PROMPT = """你是一名买方投资总监 (CIO)，负责对研究员团队的估值结论进行最终裁决。

你的核心职责不是重新估值，而是调和两种天然冲突的定价范式：

## 范式一：经典古典价值（清算底裤）
- 只承认可验证的硬资产、现有利润和可比的行业倍数
- 不承认任何「叙事溢价」——技术壁垒、国产替代、供应链卡脖子等均不构成溢价理由
- 这是卖方研报不敢写、但买方风控必须计算的「脱水底线」

## 范式二：动态科技叙事（多头弹性）
- 承认高壁垒材料的稀缺性溢价（认证周期>2年 = 准护城河）
- 承认放量驱动+渗透率爆发期的 PEG 弹性（PEG 1.2-1.8 在爆发期是常态）
- 承认进攻防线（流动性充裕）中市场给予成长股的合理情绪溢价

## 你的裁决框架

1. **识别估值路径**：
   - 如果研究员使用了 SOTP 分部加总 → 估值偏向保守（各板块分别用行业平均PE）
   - 如果研究员使用了整体法 PEG/SOTP 综合 → 估值偏向乐观（容易忽略板块间拖累）

2. **判断是否需要修正**：
   - 当 SOTP 压出极低估值时，检查核心主业是否具备「叙事溢价」资格：
     · 分支B（放量驱动）+ 渗透率5-30%爆发期 + 半导体/前驱体/光刻胶等关键词
     · 如果具备 → 可给予核心主业最高+15%的叙事溢价
   - 当整体法暴出极高估值时，检查红队是否发现了实质性风险：
     · 如果有未充分定价的 CAPEX/折旧/价格战风险 → 施加 10-15% 风险折扣
     · 如果红队风险是周期性的而非结构性的 → 仅施加 5-8% 温和折扣

3. **输出你的裁决**：
   - 古典价格：这是风控底线，永远呈现
   - 叙事价格：这是你的综合裁决均衡价，包含了合理的稀缺性/成长性溢价
   - 溢价率：-0.15 到 +0.15（负值=风险折扣，正值=叙事溢价）
   - 裁决理由：解释你为什么做出这个调整

## 禁止事项
- 禁止在两个价格之间简单取平均值
- 禁止无视红队发现的实质性风险
- 禁止对低壁垒周期品给予叙事溢价
- 禁止对治理瑕疵标的给予任何溢价"""

CIO_ANALYSIS_PROMPT = """请对以下标的进行 CIO 终裁。

## 系统计算的古典价值（清算底裤）
系统已计算: **{computed_classical:.2f} 元/股**
这个值基于纯基本面（SOTP折价后或单框架估值），不使用此值作为古典价。
你的任务是基于此值，考虑叙事溢价或风险折扣，给出叙事均衡价。

## 公司数据
{company_data}

## 研究员定性标签
{labels}

## 当前估值结论
{valuation_summary}

## 红队（空头杠精）发现
{red_team_summary}

## 估值路径
{valuation_path}

## 市场定价
当前股价: {current_price:.2f} 元

请输出严格 JSON（不要包含 markdown 代码块标记）:
{{
  "classical_price": {computed_classical:.2f}（必须等于系统计算的古典价值）,
  "narrative_price": 数字（动态科技叙事·CIO综合裁决均衡价，元/股）,
  "narrative_premium_rate": 数字（-0.15到+0.15，负值=风险折扣，正值=叙事溢价）,
  "cio_verdict": "CIO裁决结论（1-2句话）",
  "rationale": "详细裁决理由（3-5句话，说明为什么给予这个溢价率）",
  "key_judgment": "市场在定价什么？模型在忽略什么？",
  "classical_deviation": 数字（古典价 vs 市场价的偏离度，小数）,
  "narrative_deviation": 数字（叙事价 vs 市场价的偏离度，小数）
}}"""

# ═══════════════════════════════════════════════════════════════
#  Snapshot builders for CIO prompt
# ═══════════════════════════════════════════════════════════════

def _build_cio_company_snapshot(state: PipelineState) -> str:
    """Build a compact company data snapshot for the CIO."""
    fin = state.get("financials")
    segments = state.get("segments", [])
    gov = state.get("governance_flags", [])

    lines = [
        f"标的: {state.get('stock_name', '')} ({state.get('stock_code', '')})",
        f"行业: {state.get('industry', '')}",
        f"宏观防线: {state.get('macro_beta', '进攻防线')}",
        f"总股本: {state.get('total_shares', 0):.2f} 亿股",
        "",
    ]

    if fin:
        lines.extend([
            f"近3年营收 (亿): {fin.revenue}",
            f"近3年归母净利润 (亿): {fin.net_profit_parent}",
            f"近3年经营现金流 (亿): {fin.op_cash_flow}",
            f"资产负债率: {fin.debt_ratio:.1f}%",
            f"固定资产/总资产: {fin.fa_to_ta:.1%}",
            f"调整后(固+在)/总资产: {fin.adjusted_fa_to_ta:.1%}",
            f"净现比3Y均值: {fin.cash_ratio_3y_avg:.2f}",
            f"在建工程: {fin.construction_in_progress:.1f} 亿",
            "",
        ])

    if segments:
        lines.append("业务板块:")
        for seg in segments:
            lines.append(
                f"  - {seg.name}: 营收{seg.revenue:.1f}亿({seg.revenue_share:.1f}%), "
                f"利润{seg.profit_contribution:.1f}亿({seg.profit_share:.1f}%), "
                f"毛利率{seg.gross_margin:.1f}%, 增速{seg.yoy_growth:+.1f}%, "
                f"分支={seg.branch or '待定'}"
            )

    if gov:
        lines.append(f"\n治理红旗 ({len(gov)} 条):")
        for g in gov:
            lines.append(f"  - {g}")

    return "\n".join(lines)


def _build_cio_labels_summary(state: PipelineState) -> str:
    """Build qualitative labels summary for the CIO."""
    return (
        f"L1 利润驱动力: {state.get('profit_driver', '')}\n"
        f"L2 渗透率阶段: {state.get('penetration_stage', '')}\n"
        f"L2 重资产判定: {'是' if state.get('fixed_asset_heavy') else '否'}\n"
        f"L3 ASP趋势: {state.get('competition_asp_trend', '')}\n"
        f"L3 份额趋势: {state.get('market_share_trend', '')}\n"
        f"L3 PEG调整: {state.get('peg_adjustment', 1.0)}\n"
        f"建议框架: {state.get('suggested_framework', '')}\n"
        f"竞争研判: {state.get('competition_verdict', '')}"
    )


def _build_cio_valuation_summary(state: PipelineState) -> str:
    """Build a valuation summary for the CIO."""
    valuation = state.get("valuation")
    if valuation is None:
        return "无估值数据"

    sotp = state.get("sotp")
    lines = [
        f"估值框架: {valuation.framework}",
        f"基准市值: {valuation.base_case_value:.0f} 亿",
        f"基准股价: {valuation.base_case_price:.2f} 元",
        f"乐观/悲观: {valuation.bull_case_price:.2f} / {valuation.bear_case_price:.2f} 元",
    ]

    if sotp and sotp.triggered:
        lines.append(f"\nSOTP 分部估值 (触发):")
        lines.append(f"  综合PE: {sotp.composite_pe:.1f}x")
        lines.append(f"  隐含股价: {sotp.implied_price:.2f} 元")
        if sotp.conglomerate_discount > 0:
            lines.append(f"  集团折价: {sotp.conglomerate_discount:.0%}")
            lines.append(f"  折价后股价: {sotp.discounted_price:.2f} 元")
        lines.append("  板块估值明细:")
        for seg in sotp.segments:
            lines.append(
                f"    {seg.name}: PE {seg.comparable_pe:.0f}x, "
                f"利润{seg.profit_contribution:.1f}亿"
            )
    else:
        lines.append("\n整体法估值（单框架，非SOTP）")

    if valuation.key_risks:
        lines.append(f"\n关键风险:")
        for r in valuation.key_risks[:5]:
            lines.append(f"  - {r}")

    return "\n".join(lines)


def _build_cio_red_team_summary(state: PipelineState) -> str:
    """Build a summary of red team findings for the CIO."""
    red = state.get("red_team_output", {})
    if not red or not red.get("bearish_signals"):
        return "红队尚未运行（无反思轮次触发）"

    lines = [
        f"严重级别: {red.get('severity', 'N/A')}",
        f"核心空头逻辑: {red.get('reasoning', 'N/A')}",
        "",
        "发现的负面信号:",
    ]
    for i, s in enumerate(red.get("bearish_signals", []), 1):
        lines.append(f"  {i}. {s[:200]}")

    most_dangerous = red.get("most_dangerous_signal", "")
    if most_dangerous:
        lines.append(f"\n最有杀伤力的信号: {most_dangerous}")

    adj = red.get("suggested_adjustments", {})
    if adj:
        lines.append("\n红队建议修正:")
        if adj.get("peg_haircut_recommendation"):
            lines.append(f"  PEG: {adj['peg_haircut_recommendation']}")
        if adj.get("valuation_floor_risk"):
            lines.append(f"  极端下限: {adj['valuation_floor_risk']}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Deterministic Classical Price computation
# ═══════════════════════════════════════════════════════════════

NARRATIVE_PREMIUM_KEYWORDS = [
    "前驱体", "HBM", "high-k", "光刻胶", "半导体", "KrF", "ArF",
    "锗", "镓", "铟", "稀土", "碳化硅", "氮化镓",
]


def _compute_classical_price(state: PipelineState) -> float:
    """
    Compute the classical (liquidation-floor) price deterministically.

    This is the pure fundamental value without any narrative premium.
    It is always the most conservative estimate available:
      - SOTP path: use discounted_price (with conglomerate discount)
      - Single-framework path: use base_case_price
    """
    sotp = state.get("sotp")
    sotp_triggered = state.get("sotp_triggered", False)
    valuation = state.get("valuation")

    if sotp_triggered and sotp and sotp.triggered:
        if sotp.discounted_price > 0:
            return sotp.discounted_price
        return sotp.implied_price

    if valuation:
        return valuation.base_case_price

    return 0.0


def _compute_valuation_path(state: PipelineState) -> str:
    """Describe which valuation path was taken."""
    sotp_triggered = state.get("sotp_triggered", False)
    verification = ""

    if sotp_triggered:
        segments = state.get("segments", [])
        branches = set()
        for s in segments:
            if s.branch:
                branches.add(s.branch)
        verification = f"SOTP 分部加总 ({len(segments)}个板块)"
        if len(branches) >= 2:
            verification += f", 跨越分支: {', '.join(sorted(branches))}"
    else:
        verification = "整体法单一框架估值"
        framework = state.get("suggested_framework", "")
        if framework:
            verification += f" ({framework})"

    return verification


# ═══════════════════════════════════════════════════════════════
#  CIO LLM Call
# ═══════════════════════════════════════════════════════════════

def _run_cio_reconciliation(state: PipelineState, classical_price: float) -> dict:
    """
    Run the CIO LLM reconciliation and return the result dict.

    On LLM failure, returns a deterministic fallback that applies:
      - 8% risk discount if deviation > 30% (market >> model)
      - 8% narrative premium if deviation > 30% (model >> market)
      - no adjustment otherwise
    """
    current_price = state.get("current_price", 0.0)
    deviation = abs(classical_price - current_price) / current_price if current_price > 0 else 0

    company_data = _build_cio_company_snapshot(state)
    labels = _build_cio_labels_summary(state)
    valuation_summary = _build_cio_valuation_summary(state)
    red_team_summary = _build_cio_red_team_summary(state)
    valuation_path = _compute_valuation_path(state)

    prompt = CIO_ANALYSIS_PROMPT.format(
        computed_classical=classical_price,
        company_data=company_data,
        labels=labels,
        valuation_summary=valuation_summary,
        red_team_summary=red_team_summary,
        valuation_path=valuation_path,
        current_price=current_price,
    )

    print(f"  🏛  [CIO] 启动投资总监冲突调和... (古典价={classical_price:.2f}, 偏离={deviation:.1%})")

    try:
        response = litellm.completion(
            model=get_model(),
            messages=[
                {"role": "system", "content": CIO_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        result = json.loads(content)

        print(f"  ✓ [CIO] 裁决: {result.get('cio_verdict', 'N/A')[:100]}")
        print(f"    叙事溢价率: {result.get('narrative_premium_rate', 0):+.1%} | "
              f"古典价={result.get('classical_price', 0):.2f} | "
              f"叙事价={result.get('narrative_price', 0):.2f}")

        return result

    except Exception as e:
        print(f"  ✗ [CIO] LLM调用失败: {e}，使用确定性回退")

        # Deterministic fallback
        return _cio_deterministic_fallback(state, classical_price, deviation)


def _cio_deterministic_fallback(
    state: PipelineState,
    classical_price: float,
    deviation: float,
) -> dict:
    """
    Deterministic CIO fallback when LLM is unavailable.

    Rules:
      - SOTP path + deviation > 30% + 分支B semiconductor → +10% narrative premium
      - Single-framework + deviation > 30% → -10% risk discount
      - Otherwise → no adjustment
    """
    sotp_triggered = state.get("sotp_triggered", False)
    current_price = state.get("current_price", 0.0)
    segments = state.get("segments", [])
    profit_driver = state.get("profit_driver", "")
    penetration = state.get("penetration_stage", "")

    # Check for narrative-eligible segments
    has_narrative_segments = any(
        seg.branch == "分支B" and any(kw in seg.name for kw in NARRATIVE_PREMIUM_KEYWORDS)
        for seg in segments
    )

    narrative_active = (
        profit_driver == "放量驱动"
        and penetration == "5-30%爆发期"
        and has_narrative_segments
    )

    premium_rate = 0.0
    rationale_parts = []

    if deviation > 0.30:
        if sotp_triggered and narrative_active:
            # SOTP undervalues high-barrier segments → apply narrative premium
            premium_rate = 0.10
            rationale_parts.append(
                "SOTP路径低估了核心半导体材料的稀缺性溢价（认证壁垒>2年=准护城河）"
            )
            rationale_parts.append(
                f"放量驱动+爆发期+半导体材料 → 触发确定性+{premium_rate:.0%}叙事溢价"
            )
        elif not sotp_triggered:
            # Single-framework likely overvalues → apply risk discount
            premium_rate = -0.10
            rationale_parts.append(
                "整体法可能高估（未充分分离板块间拖累效应）→ 触发确定性风险折扣"
            )
            rationale_parts.append(f"施加{abs(premium_rate):.0%}风控折扣以对冲模型乐观偏差")
        else:
            rationale_parts.append(f"偏离度{deviation:.0%}但无明确调校方向，保持古典价")

    narrative_price = classical_price * (1 + premium_rate)
    classical_dev = (classical_price - current_price) / current_price if current_price > 0 else 0
    narrative_dev = (narrative_price - current_price) / current_price if current_price > 0 else 0

    return {
        "classical_price": round(classical_price, 2),
        "narrative_price": round(narrative_price, 2),
        "narrative_premium_rate": round(premium_rate, 4),
        "cio_verdict": "确定性回退裁决（LLM不可用）",
        "rationale": "; ".join(rationale_parts) if rationale_parts else "偏离可控，无需调整",
        "key_judgment": f"市场价{current_price:.2f} vs 古典价{classical_price:.2f}，偏离{deviation:.0%}",
        "classical_deviation": round(classical_dev, 4),
        "narrative_deviation": round(narrative_dev, 4),
    }


# ═══════════════════════════════════════════════════════════════
#  v3.2 智能偏离分类器 (Intelligent Deviation Classifier)
# ═══════════════════════════════════════════════════════════════

def _classify_deviation(state: PipelineState) -> str:
    """
    5-way classification of valuation deviation for intelligent routing.

    Reads qualitative labels and financial structure to determine WHY
    the model price deviates >30% from market price, then assigns a
    remedial focus string consumed by intelligent_deviation_router.

    Returns one of:
      - "基本面质变"            → driver flipped, re-analyse L1→L2→L3
      - "Value Trap: 供给侧产能风险" → CAPEX red team
      - "Value Trap: 恶性价格战风险" → Competition red team
      - "泡沫定价容忍"           → accept narrative premium, go to report
      - "未知严重偏离"           → re-screen from quick screen
    """
    latest = state.get("latest_quarter_signals", {})
    profit_driver = state.get("profit_driver", "")
    capex_status = state.get("capex_status", "N/A")
    asp_trend = state.get("asp_trend", "N/A")
    suggested_framework = state.get("suggested_framework", "")
    macro_beta = state.get("macro_beta", "")

    # 1. 基本面质变: profit driver may have flipped (e.g. 涨价→放量 or vice versa)
    if latest.get("driver_flipped"):
        return "基本面质变"

    # 2. Value Trap CAPEX: 涨价驱动 + CAPEX高企 → supply-side overcapacity risk
    if profit_driver == "涨价驱动" or capex_status == "CAPEX高企/产能投放":
        return "Value Trap: 供给侧产能风险"

    # 3. Value Trap Competition: 放量驱动 + ASP恶性崩塌 → price war / share erosion
    if profit_driver == "放量驱动" and asp_trend == "恶性崩塌(>15%)":
        return "Value Trap: 恶性价格战风险"

    # 4. 泡沫定价容忍: PE/PEG框架 + 进攻防线 → market may be pricing growth/moat premium
    if suggested_framework in ("PE/PEG弹性锚", "PEG+S曲线+SOTP") and macro_beta == "进攻防线":
        return "泡沫定价容忍"

    # 5. Fallback: 未知严重偏离 → re-screen from scratch
    return "未知严重偏离"


# ═══════════════════════════════════════════════════════════════
#  Main Node 3: CIO Reflection
# ═══════════════════════════════════════════════════════════════

def node3_reflection(state: PipelineState) -> PipelineState:
    """
    CIO conflict reconciliation + reflection loop trigger.

    1. Compute classical (liquidation-floor) price deterministically
    2. Run CIO LLM to get narrative-adjusted price
    3. Store cio_reconciliation in state for dual-track report
    4. Check if reflection loop is needed (deviation > 30%)

    Sets:
      - cio_reconciliation: dict with dual-track prices and verdict
      - valuation_deviation: based on narrative (CIO-adjusted) price
      - reflection_triggered: True if >30% deviation
      - reflection_round: incremented
    """
    valuation = state.get("valuation")
    if valuation is None:
        state["valuation_deviation"] = 0.0
        state["reflection_triggered"] = False
        state["reflection_round"] = 0
        state["cio_reconciliation"] = {}
        return state

    current_price = state.get("current_price", 0.0)
    if current_price <= 0:
        state["valuation_deviation"] = 0.0
        state["reflection_triggered"] = False
        state["cio_reconciliation"] = {}
        return state

    # Step 1: Compute classical (liquidation-floor) price
    classical_price = _compute_classical_price(state)
    if classical_price <= 0:
        classical_price = valuation.base_case_price

    # Step 2: Run CIO LLM reconciliation
    cio_result = _run_cio_reconciliation(state, classical_price)
    state["cio_reconciliation"] = cio_result

    # Step 3: Use CIO narrative price for deviation check
    narrative_price = cio_result.get("narrative_price", classical_price)
    deviation = abs(narrative_price - current_price) / current_price
    state["valuation_deviation"] = round(deviation, 4)

    # Step 4: Reflection loop logic with intelligent deviation classification
    round_num = state.get("reflection_round", 0) + 1
    state["reflection_round"] = round_num

    if deviation > 0.30 and round_num <= MAX_REFLECTION_ROUNDS:
        focus = _classify_deviation(state)
        state["red_team_focus"] = focus

        if focus == "泡沫定价容忍":
            state["reflection_triggered"] = False
            print(f"  ⊘ [Reflect] 偏离{deviation:.1%}但属泡沫定价容忍（PE/PEG+进攻防线），接受叙事溢价")
        else:
            state["reflection_triggered"] = True
            print(f"  ⟲ [Reflect] CIO叙事价偏离{deviation:.1%}>30%，触发{ focus }反思闭环 (第{round_num}轮)")
    else:
        state["reflection_triggered"] = False
        if deviation > 0.30:
            print(f"  ⊘ [Reflect] 已达最大反思轮次({MAX_REFLECTION_ROUNDS})，进入报告")

    return state


# ═══════════════════════════════════════════════════════════════
#  Red Team Injection (for reflection loop)
# ═══════════════════════════════════════════════════════════════

REFLECTION_PROMPT_INJECT = """【⚠ 投资总监二元强制反思指令】

当前模型估值与市场价格严重偏离（偏差 > 30%）。市场定价不一定代表错误 ——
它可能发现了模型忽略的因子。请从正反两个方向做均衡审视，不可单向悲观。

████████████████████████████████████████████████████████████████
█  方向一：模型可能正确 — 市场过度乐观（股价高估）
████████████████████████████████████████████████████████████████

1. 供给侧 CAPEX 大扩产：该行业是否正在经历大规模资本开支周期？
   → 在建产能何时释放？行业有效产能增速是否远超需求增速？

2. 价格战与隐性卷度：是否有新进入者/二线厂商在牺牲利润抢夺份额？
   → 产品是否同质化？客户是否有议价优势？

3. 盈利预测下修：当前市场一致预期是否过于乐观？
   → 是否存在未体现在财报中的经营恶化信号？

4. 技术迭代风险：新材料是否面临替代技术路线威胁？
   → 例如：HBM 堆叠方案是否可能减少 single-wafer 材料用量？

████████████████████████████████████████████████████████████████
█  方向二：市场可能正确 — 模型过于保守（新材料溢价被低估）
████████████████████████████████████████████████████████████████

1. 放量市新材料稀缺溢价：
   → 在进攻防线（流动性充裕）中，高壁垒新材料天然享有 PE 溢价。
   → 放量驱动型公司（分支B）处于渗透率爆发期时，PEG > 1.5 是常态而非异常。
   → 市场给予的估值可能包含了「技术独占期」的期权价值。

2. 供给刚性 → 利润非线性爆发：
   → 高端半导体材料（前驱体/光刻胶）的客户认证周期 2-3 年，产能建设 18-24 个月。
   → 需求爆发（HBM/AI）时供给无法线性响应 → 量价齐升 → 利润超预期。
   → 模型如果仅用线性 PE 估值，会系统性地低估「供需错配期」的利润弹性。

3. 国产替代加速溢价：
   → 地缘政治驱动的国产替代是结构性而非周期性变量。
   → 一旦进入头部客户（如海力士/长存）供应链，份额爬坡速度可能远超预期。

4. 伴生矿/稀缺资源属性：
   → 如果标的涉及伴生矿（如锗、镓等半导体金属），供给受主矿产出约束，
   无法独立扩产 → 天然壁垒 → 应享有稀缺溢价。

████████████████████████████████████████████████████████████████

请基于以上两个方向的博弈，重新审视并修正你的 L1 / L2 / L3 标签。
关键要求：
  - 不要单向悲观修正。在进攻防线的放量市场里，新材料溢价是合理的。
  - PEG 调整系数：如果在方向二中找到了可信的「供给刚性」或「独占期」
    证据，PEG 可以上修至 1.1-1.4（而非只降不升）。
  - 最终判断应明确回答：「市场在定价什么，模型在忽略什么？」

--- 原始分析任务 ---

"""


def build_red_team_injection(state: PipelineState) -> str:
    """
    补丁3: 将红队发现注入反思 Prompt。

    Called before dispatch_parallel_qualitative to augment the reflection
    prompt with adversarial findings from the red team agent.
    """
    red = state.get("red_team_output", {})
    if not red or not red.get("bearish_signals"):
        return ""

    signals = red.get("bearish_signals", [])
    severity = red.get("severity", "moderate")
    most_dangerous = red.get("most_dangerous_signal", "")
    adjustments = red.get("suggested_adjustments", {})
    reasoning = red.get("reasoning", "")

    lines = [
        "",
        "████████████████████████████████████████████████████████████",
        "█  🩸 红队（空头杠精）对抗性质询",
        "████████████████████████████████████████████████████████████",
        "",
        f"严重级别: {severity}",
        f"核心空头逻辑: {reasoning}",
        "",
        "发现的具体负面信号:",
    ]
    for i, s in enumerate(signals, 1):
        lines.append(f"  {i}. {s}")

    if most_dangerous:
        lines.append(f"\n⚠ 最有杀伤力的信号: {most_dangerous}")

    if adjustments:
        lines.append("\n红队建议的修正方向:")
        if adjustments.get("profit_driver_concern"):
            lines.append(f"  - 利润驱动力: {adjustments['profit_driver_concern']}")
        if adjustments.get("peg_haircut_recommendation"):
            lines.append(f"  - PEG 建议调整至: {adjustments['peg_haircut_recommendation']}")
        if adjustments.get("segment_pe_concern"):
            lines.append(f"  - PE 假设质疑: {adjustments['segment_pe_concern']}")
        if adjustments.get("valuation_floor_risk"):
            lines.append(f"  - 极端悲观估值下限: {adjustments['valuation_floor_risk']}")

    lines.append("")
    lines.append("请逐条回应以上红队质询。如果红队的担忧成立，你必须修正判断。")
    lines.append("如果红队的担忧不成立，你必须给出明确的反驳证据。")
    lines.append("禁止无视红队信号、禁止和稀泥取平均值。")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Conditional Routing
# ═══════════════════════════════════════════════════════════════

