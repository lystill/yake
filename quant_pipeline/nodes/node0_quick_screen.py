"""
Node 0 — 投前 2 分钟熔断快筛
================================
Hard-coded If-Else logic replicating the SOP 〇 pre-flight panel.
No LLM call — pure Python rules engine.

Thresholds (from INVEST_SOP.md):
  轻资产/预收型 (SaaS/白酒/游戏/高端消费): 净现比 < 0.9  → 熔断
  标准制造/传统周期 (电子布/化工/重卡/设备): 净现比 < 0.6 → 熔断
  ToG/B端强势重资产 (建筑工程/军工/PPP)   : 净现比 < 0.3 → 熔断

Governance red-flags (one-vote veto):
  - 近3年财务造假记录/立案调查
  - 大股东持续减持且无合理解释
  - 审计意见为非标
"""

from typing import Literal
from state import (
    PipelineState,
    FinancialMetrics,
    ScreeningVerdict,
    MacroBeta,
)


# ═══════════════════════════════════════════════════════════════
#  v3.2: 战略成长股豁免 — 高壁垒硬科技烧钱期标的自动晋级分支E
# ═══════════════════════════════════════════════════════════════

# Keywords that signal strategic high-barrier hard-tech (deserve growth exemption)
STRATEGIC_TECH_KEYWORDS: list[str] = [
    # 半导体/AI 硬科技
    "AI芯片", "GPU", "TPU", "NPU", "算力芯片", "训练芯片", "推理芯片",
    "智能芯片", "人工智能芯片", "AI加速卡", "AI训练", "AI推理",
    "先进封装", "IC载板", "ABF载板", "BT载板", "FC-BGA",
    "光刻机", "光刻胶", "KrF", "ArF", "EUV",
    "HBM", "高带宽存储", "high-k", "前驱体",
    "碳化硅", "SiC", "氮化镓", "GaN", "第三代半导体",
    "高端芯片", "车规级芯片", "存算一体", "云端芯片",
    "chiplet", "Chiplet", "2.5D封装", "3D封装",
    "EDA", "IP授权", "指令集", "芯片设计",
    # v3.3 热修复: 半导体设备 — 薄膜沉积/刻蚀/CMP 高壁垒硬科技
    "薄膜沉积", "PECVD", "ALD", "刻蚀机", "CVD", "PVD",
    "量测检测", "CMP", "清洗设备",
    # v3.3: 创新药/Biotech — 管线驱动，无营收/亏损期同理硬科技烧钱期
    "创新药", "生物制药", "biotech", "Biotech",
    "单抗", "双抗", "ADC药物", "基因治疗", "靶向药",
    "CAR-T", "mRNA", "细胞治疗", "基因编辑",
    "PD-1", "PD-L1", "BTK抑制剂", "PROTAC",
    "临床Ⅲ期", "FDA突破性疗法", "孤儿药",
]

# Well-known strategic growth companies that should always get exemption
STRATEGIC_GROWTH_COMPANIES: set[str] = {
    # 半导体/AI 硬科技
    "寒武纪", "海光信息", "景嘉微", "龙芯中科", "中芯国际",
    "通富微电", "长电科技", "华天科技", "甬矽电子",
    "江波龙", "兆易创新", "北京君正",
    # v3.3 热修复: 半导体设备五巨头 — 薄膜沉积/刻蚀/CMP 硬科技
    "拓荆科技", "北方华创", "中微公司", "盛美上海", "华海清科",
    # v3.3: 创新药/Biotech
    "百济神州", "信达生物", "荣昌生物", "君实生物", "恒瑞医药",
    "康希诺", "科伦博泰", "传奇生物", "和黄医药", "再鼎医药",
    "康方生物", "诺诚健华", "亚盛医药",
}

# v3.3: Biotech keywords — unconditional exemption from revenue growth threshold
PHARMA_BIOTECH_KEYWORDS: set[str] = {
    "创新药", "生物制药", "biotech", "Biotech", "单抗", "双抗", "ADC药物",
    "基因治疗", "靶向药", "CAR-T", "mRNA", "细胞治疗", "基因编辑",
    "PD-1", "PD-L1", "BTK抑制剂", "PROTAC", "临床Ⅲ期",
    "FDA突破性疗法", "孤儿药",
}

# Well-known biotech/pharma that get unconditional exemption
PHARMA_BIOTECH_COMPANIES: set[str] = {
    "百济神州", "信达生物", "荣昌生物", "君实生物", "康希诺",
    "科伦博泰", "传奇生物", "和黄医药", "再鼎医药", "康方生物",
    "诺诚健华", "亚盛医药",
}

# Minimum YoY revenue growth to qualify (latest year over previous year)
STRATEGIC_GROWTH_THRESHOLD: float = 0.30  # 30%


def _compute_max_rev_growth(fin: FinancialMetrics) -> float:
    """Compute latest-year YoY revenue growth. Returns -inf if not computable."""
    if len(fin.revenue) >= 2 and fin.revenue[-2] > 0:
        return fin.revenue[-1] / fin.revenue[-2] - 1
    return float("-inf")


def _check_strategic_growth_exemption(
    stock_name: str,
    industry: str,
    research_ctx: str,
    yoy_growth: float,
) -> tuple[bool, list[str]]:
    """
    Determine if the stock qualifies for v3.2/v3.3 strategic growth exemption.

    Two-tier detection:
      Tier 1 (v3.3 biotech): Pipeline-driven, unconditional — revenue growth irrelevant
      Tier 2 (v3.2 hard-tech): Revenue-growth-dependent — YoY must be >= 30%

    Returns (exempt: bool, reasons: list[str]).
    """
    corpus = f"{stock_name} {industry} {research_ctx}"

    # ═══════════════════════════════════════════════════════════
    #  Tier 1: v3.3 创新药/Biotech — 管线驱动，无条件豁免营收增速
    # ═══════════════════════════════════════════════════════════
    pharma_name_match = any(c in stock_name for c in PHARMA_BIOTECH_COMPANIES)
    pharma_kw_hits = [kw for kw in PHARMA_BIOTECH_KEYWORDS if kw.lower() in corpus.lower()]

    if pharma_name_match or pharma_kw_hits:
        reasons = []
        if pharma_name_match:
            reasons.append(f"知名创新药/biotech: {stock_name}")
        if pharma_kw_hits:
            reasons.append(f"创新药标签: {', '.join(pharma_kw_hits[:5])}")
        reasons.append("v3.3豁免: 创新药管线驱动，无条件豁免营收增速限制（无营收/亏损期同硬科技烧钱期）")
        reasons.append("v3.3豁免: 高壁垒医药硬资产，不适用PE/PEG估值，强制分流分支E-动态PS+管线溢价")
        return True, reasons

    # ═══════════════════════════════════════════════════════════
    #  Tier 2: v3.2 硬科技 — 需要营收增速证明
    # ═══════════════════════════════════════════════════════════
    name_match = any(c in stock_name for c in STRATEGIC_GROWTH_COMPANIES)
    tech_hits = [kw for kw in STRATEGIC_TECH_KEYWORDS if kw.lower() in corpus.lower()]

    if not name_match and not tech_hits:
        return False, []

    if yoy_growth < STRATEGIC_GROWTH_THRESHOLD:
        label = "知名战略成长股" if name_match else f"科技关键词{tech_hits[:3]}"
        return False, [f"{label}，但YoY增速{yoy_growth:.0%}<{STRATEGIC_GROWTH_THRESHOLD:.0%}阈值，不豁免"]

    reasons = []
    if name_match:
        reasons.append(f"知名战略成长股: {stock_name}")
    if tech_hits:
        reasons.append(f"科技标签: {', '.join(tech_hits[:5])}")
    reasons.append(f"营收YoY增速: {yoy_growth:.0%} ≥ {STRATEGIC_GROWTH_THRESHOLD:.0%}阈值")
    reasons.append("v3.2豁免: 高壁垒硬科技烧钱期标的，不适用传统净现比快筛")

    return True, reasons


# —— Industry → (category, 净现比熔断线) ——
INDUSTRY_THRESHOLDS: dict[str, tuple[str, float]] = {
    # 轻资产/预收型 — 净现比理应>1.0
    "SaaS":           ("轻资产/预收型", 0.9),
    "白酒":           ("轻资产/预收型", 0.9),
    "高端消费":       ("轻资产/预收型", 0.9),
    "游戏":           ("轻资产/预收型", 0.9),
    "广告":           ("轻资产/预收型", 0.9),
    "软件":           ("轻资产/预收型", 0.9),

    # 标准制造/传统周期 — 允许正常账期差
    "电子布":         ("标准制造/传统周期", 0.6),
    "化工":           ("标准制造/传统周期", 0.6),
    "重卡":           ("标准制造/传统周期", 0.6),
    "设备":           ("标准制造/传统周期", 0.6),
    "面板":           ("标准制造/传统周期", 0.6),
    "水泥":           ("标准制造/传统周期", 0.6),
    "有色金属":       ("标准制造/传统周期", 0.6),
    "钢铁":           ("标准制造/传统周期", 0.6),
    "铅锌":           ("标准制造/传统周期", 0.6),
    "矿业":           ("标准制造/传统周期", 0.6),
    "半导体材料":     ("标准制造/传统周期", 0.6),
    "电子材料":       ("标准制造/传统周期", 0.6),
    "LNG材料":        ("标准制造/传统周期", 0.6),
    "船舶":           ("标准制造/传统周期", 0.6),
    "煤炭":           ("标准制造/传统周期", 0.6),

    # ToG/B端强势重资产 — 政府回款慢但烂账率低
    "建筑工程":       ("ToG/B端重资产", 0.3),
    "军工配套":       ("ToG/B端重资产", 0.3),
    "PPP":            ("ToG/B端重资产", 0.3),
    "特种重工":       ("ToG/B端重资产", 0.3),
}


# —— Macro Beta heuristic (simplified) ——
def _assess_macro_beta() -> str:
    """
    In production this would pull宏观指标 (两市成交额/两融余额/利率).
    Hard-coded to 进攻防线 for May 2026.
    """
    return MacroBeta.OFFENSIVE.value


def _classify_industry(industry: str) -> tuple[str, float]:
    """Fuzzy-match industry name → (category, threshold)."""
    for key, (cat, threshold) in INDUSTRY_THRESHOLDS.items():
        if key in industry or industry in key:
            return cat, threshold
    # fallback: 标准制造
    return "标准制造/传统周期", 0.6


def node0_quick_screen(state: PipelineState) -> PipelineState:
    """
    Execute the 3-question quick screen.
    Writes: screening_verdict, cash_ratio_ok, governance_ok, macro_beta
    """
    fin: FinancialMetrics = state["financials"]
    industry: str = state.get("industry", "标准制造")
    governance_flags: list[str] = state.get("governance_flags", [])
    stock_name: str = state.get("stock_name", "")
    research_ctx: str = state.get("research_context", "")

    # ---- Q1: Macro Beta ----
    state["macro_beta"] = _assess_macro_beta()

    # ---- Compute derived financials (needed for both screening and exemption) ----
    ratios = []
    for i in range(len(fin.net_profit_parent)):
        if fin.net_profit_parent[i] > 0:
            ratios.append(fin.op_cash_flow[i] / fin.net_profit_parent[i])
    fin.cash_ratio_3y_avg = sum(ratios) / len(ratios) if ratios else 0.0
    fin.fa_to_ta = fin.fixed_assets / fin.total_assets if fin.total_assets > 0 else 0.0
    fin.adjusted_fa_to_ta = (fin.fixed_assets + fin.construction_in_progress) / fin.total_assets if fin.total_assets > 0 else 0.0
    latest_rev = fin.revenue[-1] if fin.revenue else 0.0
    fin.ato = latest_rev / fin.total_assets if fin.total_assets > 0 else 0.0
    if len(fin.revenue) >= 2 and fin.revenue[-2] > 0 and fin.total_assets > 0:
        rev_growth = fin.revenue[-1] / fin.revenue[-2] - 1
        if rev_growth > 0.15:
            fin.ato_trend = "改善" if fin.adjusted_fa_to_ta < 0.40 else "恶化（增收但资产膨胀更快）"
        elif rev_growth > 0:
            fin.ato_trend = "稳定"
        else:
            fin.ato_trend = "恶化"
    else:
        fin.ato_trend = "稳定"

    # ═══════════════════════════════════════════════════════════
    #  v3.2: 战略成长股豁免 — 高壁垒硬科技烧钱期标的自动晋级分支E
    # ═══════════════════════════════════════════════════════════
    yoy_growth = _compute_max_rev_growth(fin)
    exempt, exempt_reasons = _check_strategic_growth_exemption(
        stock_name, industry, research_ctx, yoy_growth
    )
    if exempt:
        state["cash_ratio_ok"] = False  # 技术上未通过，但豁免
        state["governance_ok"] = True
        state["screening_verdict"] = ScreeningVerdict.BRANCH_E.value
        state["exemption_reasons"] = exempt_reasons
        print(f"  🚀 [v3.2] 战略成长股豁免触发: {'; '.join(exempt_reasons[:3])}")
        return state

    # ---- Q2: Cash Flow Health ----
    category, threshold = _classify_industry(industry)

    if fin.cash_ratio_3y_avg < threshold:
        state["cash_ratio_ok"] = False
        state["screening_verdict"] = ScreeningVerdict.BRANCH_D.value
        state["error"] = (
            f"熔断: 净现比3Y均值 {fin.cash_ratio_3y_avg:.2f} < "
            f"{category}阈值 {threshold}. 行业={industry}"
        )
        return state

    state["cash_ratio_ok"] = True

    # ---- Q3: Governance Red Flags ----
    FATAL_FLAGS = [
        "财务造假", "立案调查", "大股东持续减持且无合理解释", "审计非标",
    ]
    triggered = [f for f in governance_flags if any(kw in f for kw in FATAL_FLAGS)]

    if triggered:
        state["governance_ok"] = False
        state["screening_verdict"] = ScreeningVerdict.ABANDON.value
        state["error"] = f"熔断: 治理红灯触发 — {', '.join(triggered)}"
        return state

    state["governance_ok"] = True
    state["screening_verdict"] = ScreeningVerdict.PASS.value
    return state


# —— Routing function for LangGraph ——
def route_after_screening(state: PipelineState) -> Literal["node_qualitative_chain", "node_branch_D", "node_branch_E", "__end__"]:
    verdict = state.get("screening_verdict", "")
    if verdict == ScreeningVerdict.ABANDON.value:
        return "__end__"
    if verdict == ScreeningVerdict.BRANCH_E.value:
        return "node_branch_E"
    if verdict == ScreeningVerdict.BRANCH_D.value:
        return "node_branch_D"
    return "node_qualitative_chain"
