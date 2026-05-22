"""
INVEST SOP State Definition
——
Structured state object for the LangGraph investment analysis pipeline.
All fields carry through the DAG; conditional routing uses typed labels.

Parallel safety: keys written by concurrent agents use Annotated reducers
to prevent LangGraph channel conflicts. Audit trail fields capture
per-agent metadata for full reproducibility.
"""

from typing import TypedDict, Literal, Optional, Annotated
from pydantic import BaseModel, Field
from enum import Enum
from operator import add


# ——— Enums for SOP fixed vocabularies ———


class MacroBeta(str, Enum):
    OFFENSIVE = "进攻防线"
    DEFENSIVE = "防御防线"


class ProfitDriver(str, Enum):
    PRICE_DRIVEN = "涨价驱动"           # 分支 A — 周期品/大宗
    VOLUME_DRIVEN = "放量驱动"          # 分支 B — 技术渗透/消费爆品
    STABLE_DIVIDEND = "稳定分红(C1)"    # 分支 C1 — 类债券/红利（水电、高速、银行）
    DISCRETE_EVENT = "离散事件(C2)"     # 分支 C2 — 并购/管线/重组/ST摘帽


class CapexStatus(str, Enum):
    """分支 A 专属：供给侧 CAPEX 状态"""
    CAPEX_EXPANDING = "CAPEX高企/产能投放"   # 行业大规模扩产，警惕博傻
    CAPEX_CONTRACTING = "CAPEX萎缩/供给刚性"  # 供给侧出清，供给刚性
    NOT_APPLICABLE = "N/A"


class ASPTrend(str, Enum):
    """分支 B 专属：产品 ASP 年化变动趋势"""
    COLLAPSE = "恶性崩塌(>15%)"    # 价格战，利润快速压缩 → 红色预警
    MILD = "正常温和(<10%)"        # 温和降价，规模效应消化
    STABLE_UP = "稳定/涨价"        # 供需平衡或供不应求
    NOT_APPLICABLE = "N/A"


class PenetrationStage(str, Enum):
    IMPORT = "<5%导入期"
    BURST = "5-30%爆发期"
    MATURE = ">30%成熟期"


class ValuationFramework(str, Enum):
    PB_ROE = "PB-ROE底部锚"
    PE_PEG = "PE/PEG弹性锚"
    PS_PIPELINE = "PS+管线估值"
    PEG_SCURVE_SOTP = "PEG+S曲线+SOTP"
    MATURE_PE_DIVIDEND = "成熟PE+股息率"
    DCF_DDM = "DCF/DDM类债券"
    RNPV = "概率加权/rNPV"
    LIQUIDATION = "清算价值/资产打折"  # 分支 D


class ScreeningVerdict(str, Enum):
    PASS = "通过"
    BRANCH_D = "熔断→分支D"
    BRANCH_E = "战略成长股豁免→分支E"  # v3.2: AI芯片/先进封装等高壁垒硬科技烧钱期标的
    ABANDON = "熔断→放弃"


# ——— Custom reducers for parallel-safe state keys ———

def _last_write_wins(x, y):
    """Last-write-wins reducer for single-value channels.

    Used when multiple parallel branches send different values for
    the same key (e.g. _segment_index in SOTP fan-out). Always
    returns y — the latest write.
    """
    return y


def _merge_dicts(a: dict, b: dict) -> dict:
    """Merge two dicts. Used for agent_audit channel to merge per-agent timestamps."""
    return {**a, **b}


# ——— Structured sub-models ———


class FinancialMetrics(BaseModel):
    """Core financial indicators extracted / hard-coded."""
    revenue: list[float]          # 3-year [2023, 2024, 2025] in 亿元
    net_profit_parent: list[float]  # 3-year 归母净利润 亿元
    op_cash_flow: list[float]     # 3-year 经营活动现金流净额 亿元
    fixed_assets: float           # latest 固定资产 亿元
    construction_in_progress: float = 0.0  # latest 在建工程 亿元
    total_assets: float           # latest 总资产 亿元
    debt_ratio: float             # latest 资产负债率 %
    cash_ratio_3y_avg: float = 0.0   # computed: 3Y avg 净现比
    fa_to_ta: float = 0.0            # computed: 固定资产/总资产
    adjusted_fa_to_ta: float = 0.0   # computed: (固定资产+在建工程)/总资产
    ato: float = 0.0                 # computed: 资产周转率 = 最新收入/总资产
    ato_trend: str = ""              # computed: ATO变动趋势 "改善"/"稳定"/"恶化"


class BusinessSegment(BaseModel):
    """One business segment for SOTP analysis."""
    name: str
    revenue: float          # 亿元
    revenue_share: float    # %
    profit_contribution: float  # 亿元
    profit_share: float     # %
    gross_margin: float     # %
    yoy_growth: float       # % revenue growth
    roe: float = 0.0        # segment ROE if available
    branch: str = ""        # which SOP branch this segment maps to
    comparable_pe: float = 0.0  # comparable company PE


class SOTPResult(BaseModel):
    """SOTP analysis output."""
    triggered: bool
    trigger_reasons: list[str] = []
    segments: list[BusinessSegment] = []
    total_valuation: float = 0.0     # 亿元（折价前）
    implied_price: float = 0.0       # 元/股（折价前）
    composite_pe: float = 0.0
    conglomerate_discount: float = 0.0  # 补丁2: 集团折价系数 (0=无折价, 0.10=10%折价)
    discounted_valuation: float = 0.0   # 亿元（折价后）
    discounted_price: float = 0.0       # 元/股（折价后）


class ValuationConclusion(BaseModel):
    """Final output of the pipeline."""
    framework: str = ""                    # which framework used
    sotp: Optional[SOTPResult] = None
    base_case_value: float = 0.0          # 亿元
    base_case_price: float = 0.0          # 元/股
    bull_case_price: float = 0.0
    bear_case_price: float = 0.0
    key_risks: list[str] = []
    recommendation: str = ""              # free-text buy-side memo


# ——— Agent Audit Trail sub-model ———

class AgentAuditEntry(BaseModel):
    """Per-agent execution metadata for full audit trail."""
    agent: str                       # "L1" / "L2" / "L3" / "SEG_0" / ...
    model: str = ""                  # LLM model used
    timestamp: str = ""              # ISO timestamp
    latency_ms: float = 0.0          # wall-clock latency
    tokens_used: int = 0             # total tokens consumed
    fallback: bool = False           # True if LLM call failed, used default
    error: str = ""                  # error message if fallback triggered


# ——— Main Pipeline State ———


class PipelineState(TypedDict, total=False):
    """
    The single state object flowing through all LangGraph nodes.

    Parallel-safety notes:
      - Fields with Annotated[..., _last_write_wins] are safe for concurrent
        Send-based fan-out (multiple branches writing different values).
      - Fields with Annotated[..., add] are safe for concurrent appends
        (e.g. segment_valuations in SOTP fan-out).
      - Fields without Annotated are written by exactly ONE node within any
        parallel step — no conflict possible.
    """

    # --- Input ---
    stock_code: str
    stock_name: str
    current_price: float
    total_shares: float               # 亿股

    # --- Financial data (node0_fetch or manual) ---
    financials: FinancialMetrics
    segments: list[BusinessSegment]

    # --- Node 0: Quick Screen ---
    macro_beta: str                     # MacroBeta value
    cash_ratio_ok: bool
    governance_ok: bool
    governance_flags: list[str]
    screening_verdict: str             # ScreeningVerdict value
    exemption_reasons: list[str]       # v3.2: 战略成长股豁免触发原因

    # --- Qualitative labels (written by dispatch_chain + merge_qualitative) ---
    profit_driver: str                 # ProfitDriver value
    penetration_stage: str             # PenetrationStage value
    fixed_asset_heavy: bool            # 固/总 > 40% ?
    competition_asp_trend: str         # legacy: "上行"/"稳定"/"年降>15%"/"年降>10-15%"/"年降<10%"
    asp_trend: str                     # 分支B专属: ASPTrend Literal (恶性崩塌/正常温和/稳定涨价/N/A)
    market_share_trend: str            # "扩张"/"稳定"/"下滑"
    capex_status: str                  # 分支A专属: CapexStatus Literal (CAPEX高企/CAPEX萎缩/N/A)
    competition_verdict: str           # free-text competitive landscape summary
    peg_adjustment: float              # 1.0 = no adjustment, 0.7 = 30% haircut
    suggested_framework: str           # from L2 agent, constrained to ValuationFramework enum

    # --- Parallel Agent Raw Outputs ---
    # Written by node_qualitative_chain (single writer for each key).
    # No Annotated reducer needed — each key has exactly one writer.
    l1_output: dict                    # raw L1 agent output
    l2_output: dict                    # raw L2 agent output
    l3_output: dict                    # raw L3 agent output

    # --- Cross-Validation Results (written by merge_qualitative, single writer) ---
    label_consensus: bool              # True if all 3 agents agree
    label_conflicts: list[str]         # conflict descriptions from cross-validation
    resolved_by: list[str]             # which rules resolved conflicts (e.g. "RULE_1", "RULE_3")

    # --- Segment Agent Outputs (for SOTP fan-out) ---
    # ThreadPoolExecutor collects all results within dispatch_parallel_sotp,
    # so there are no concurrent writes — plain list, no reducer needed.
    # Default "last write wins" ensures reflection loops replace (not append).
    segment_valuations: list[dict]

    # SOTP trigger condition details (single writer)
    sotp_trigger_reasons: list[str]
    sotp_force_single_path: bool        # 硬编码红线: 多主业单板块强制走SOTP提示词

    # --- Node 2: SOTP + Valuation ---
    sotp_triggered: bool
    sotp: Optional[SOTPResult]
    valuation: ValuationConclusion

    # --- Node 2: Pharma/Biotech v3.3 details ---
    pharma_valuation_details: dict        # v3.3: PS baseline, pipeline premium, rNPV addon

    # --- Node 3: Reflection ---
    reflection_round: int               # 0 = first pass, 1+ = loop-back rounds
    valuation_deviation: float          # abs(theory - price) / price
    reflection_triggered: bool          # True if >30% deviation
    red_team_output: dict               # 补丁3: 红队对抗性质询结果
    red_team_focus: str                 # v3.2: 智能路由注入的红队专项指令
    cio_reconciliation: dict            # v3.1: CIO双轨制调和
    latest_quarter_signals: dict        # v3.2: 最新季度信号 {driver_flipped: bool, ...} {classical_price, narrative_price, narrative_premium_rate, cio_verdict, rationale}

    # --- Research context (fetcher) ---
    industry: str                       # 行业分类
    research_context: str               # 研报/行业摘要文本

    # --- Audit Trail ---
    # Per-agent execution metadata. Written by dispatch nodes.
    # Single writer (node_qualitative_chain writes all 3 sequentially).
    agent_audit: dict                   # {"L1": AgentAuditEntry, "L2": ..., "L3": ...}
    # Merge decision log
    merge_audit: dict                   # {"timestamp": ..., "rules_fired": [...], "conflicts_found": int}
    # Pipeline-level metadata
    pipeline_start_ts: str              # ISO timestamp when pipeline started
    graph_topology: str                 # topology identifier for report footer

    # --- Output control ---
    generate_pdf: bool                  # False by default — PDF is slow and large

    # --- Flow control ---
    error: str
