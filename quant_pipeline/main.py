"""
INVEST SOP 量化投研流水线 — 生产入口
=====================================
Usage:
  python main.py --stock 600497           # 真实抓取 + 全流程分析
  python main.py --stock 600497 --price 9.95  # 手动指定股价
  python main.py --demo-xinrui            # 晶瑞新材虚拟测试
  python main.py --demo-yake              # 雅克科技虚拟测试
"""

import sys
import os
from dotenv import load_dotenv

load_dotenv()

# —— Configure LLM before importing graph nodes ——
from utils import config as _llm_config

from graph import build_graph, run_pipeline
from state import PipelineState, FinancialMetrics, BusinessSegment
from fetcher.financials import fetch_financial_data, fetch_current_price
from fetcher.research import fetch_research_report


# ═══════════════════════════════════════════════════════════
#  虚拟数据（仅供快速测试）
# ═══════════════════════════════════════════════════════════

MOCK_INDUSTRY_REPORT = """
┌─────────────────────────────────────────────────────────────┐
│           华泰证券 — 半导体材料行业 2026Q2 深度跟踪            │
├─────────────────────────────────────────────────────────────┤
│  【HBM 需求端】                                               │
│  SK 海力士 HBM4 将于 2026H2 量产，对 high-k 前驱体材料需求     │
│  同比增长 80%+。公司 A 作为国内唯一进入海力士供应链的厂商。     │
│  【ASP 风险端】                                               │
│  二线厂商在建产能预计 2026Q3-2027Q1 集中释放，封装基板 ASP      │
│  自 2025Q4 开始松动，预计 2026 全年 ASP 同比下降 15-20%。      │
│  【一致预期】                                                  │
│  市场一致预期 2026E 净利润 12.5 亿，forward PE 30x。           │
│  若封装基板 ASP 全年下滑 20%，净利润或下调至 10-11 亿。        │
└─────────────────────────────────────────────────────────────┘
"""

DEMO_STATE_XR: PipelineState = {
    "stock_code": "688126",
    "stock_name": "晶瑞新材",
    "current_price": 105.00,
    "total_shares": 5.20,
    "industry": "半导体材料",
    "financials": FinancialMetrics(
        revenue=[35.20, 48.70, 68.50],
        net_profit_parent=[4.80, 7.20, 10.50],
        op_cash_flow=[3.20, 5.10, 8.90],
        fixed_assets=38.50,
        total_assets=92.30,
        debt_ratio=35.20,
    ),
    "segments": [
        BusinessSegment(
            name="半导体前驱体（HBM用high-k材料）",
            revenue=28.50, revenue_share=41.6,
            profit_contribution=5.50, profit_share=52.4,
            gross_margin=52.0, yoy_growth=45.0,
            roe=22.0, branch="分支B",
        ),
        BusinessSegment(
            name="先进封装基板材料（ASP年降~15%，二线扩产冲击）",
            revenue=22.00, revenue_share=32.1,
            profit_contribution=3.00, profit_share=28.6,
            gross_margin=22.0, yoy_growth=8.0,
            roe=9.0, branch="分支A",
        ),
        BusinessSegment(
            name="光刻胶及配套试剂",
            revenue=18.00, revenue_share=26.3,
            profit_contribution=2.00, profit_share=19.0,
            gross_margin=35.0, yoy_growth=35.0,
            roe=15.0, branch="分支B",
        ),
    ],
    "governance_flags": [],
}

DEMO_STATE_YAKE: PipelineState = {
    "stock_code": "002409",
    "stock_name": "雅克科技",
    "current_price": 107.46,
    "total_shares": 4.76,
    "industry": "半导体材料",
    "financials": FinancialMetrics(
        revenue=[47.38, 68.62, 86.11],
        net_profit_parent=[5.79, 8.72, 10.00],
        op_cash_flow=[5.89, 6.04, 10.31],
        fixed_assets=51.89,
        construction_in_progress=20.0,  # 补丁1: 在建工程（新厂房产线未转固）
        total_assets=169.62,
        debt_ratio=42.6,
    ),
    "segments": [
        BusinessSegment(
            name="半导体前驱体",
            revenue=21.11, revenue_share=24.5,
            profit_contribution=4.0, profit_share=40.0,
            gross_margin=44.79, yoy_growth=8.0, branch="分支B",
        ),
        BusinessSegment(
            name="光刻胶及配套",
            revenue=19.60, revenue_share=22.8,
            profit_contribution=1.8, profit_share=18.0,
            gross_margin=16.72, yoy_growth=27.7, branch="分支B",
        ),
        BusinessSegment(
            name="LNG保温绝热材料",
            revenue=23.70, revenue_share=27.5,
            profit_contribution=3.0, profit_share=30.0,
            gross_margin=31.0, yoy_growth=44.9, branch="分支A/C1",
        ),
    ],
    "governance_flags": ["大股东沈氏家族持续减持套现超9亿(2020-2023)"],
}

DEMOS = {"xinrui": DEMO_STATE_XR, "yake": DEMO_STATE_YAKE}


# ═══════════════════════════════════════════════════════════
#  生产模式：构建最小输入状态（node0_fetch 负责抓取）
# ═══════════════════════════════════════════════════════════

def build_minimal_state(
    stock_code: str,
    stock_name: str = "",
    industry: str = "",
    current_price: float = 0.0,
    governance_flags: list[str] | None = None,
) -> PipelineState:
    """
    Build a minimal PipelineState — node0_fetch handles the rest.

    Only stock_code is required. All financial data, price, shares
    will be fetched dynamically by node0_fetch via akshare.
    """
    state: PipelineState = {
        "stock_code": stock_code,
        "stock_name": stock_name or f"股票{stock_code}",
        "current_price": current_price,       # 0 = auto-fetch
        "total_shares": 0.0,                  # will be fetched
        "industry": industry,                 # "" = auto-infer
        "governance_flags": governance_flags or [],
    }
    return state


# ═══════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════

def print_usage():
    print("INVEST SOP 量化投研流水线 — 生产版本 v3.0")
    print("=" * 56)
    print()
    print("用法:")
    print("  python main.py --stock 600497              # 全自动：抓取+分析+报告")
    print("  python main.py --stock 600497 --name 驰宏锌锗 --industry 铅锌")
    print("  python main.py --stock 002409 --name 雅克科技 --industry 半导体材料")
    print("  python main.py --demo-xinrui               # 晶瑞新材虚拟测试")
    print("  python main.py --demo-yake                 # 雅克科技虚拟测试(含治理黄旗)")
    print()
    print("数据源:")
    print("  akshare (同花顺) → 利润表/现金流/资产负债表 → 近3年真实财务")
    print("  research_reports/[代码].txt → 本地研报上下文 (可选)")
    print("  .env → LLM API 密钥配置")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    # ——— Demo modes ———
    if "--demo-xinrui" in args:
        print(MOCK_INDUSTRY_REPORT)
        state = DEMO_STATE_XR
        print(f"\n▶ 启动 INVEST SOP 流水线 (LangGraph — 虚拟测试模式)")
        print(f"  标的: {state['stock_name']} ({state['stock_code']})")
        print(f"  行业: {state.get('industry', 'N/A')}")
        result = run_pipeline(state)
        sys.exit(0)

    if "--demo-yake" in args:
        state = DEMO_STATE_YAKE
        print(f"\n▶ 启动 INVEST SOP 流水线 (LangGraph — 虚拟测试模式)")
        print(f"  标的: {state['stock_name']} ({state['stock_code']})")
        print(f"  行业: {state.get('industry', 'N/A')}")
        print(f"  🚩 治理黄旗: {len(state.get('governance_flags', []))} 条")
        result = run_pipeline(state)
        sys.exit(0)

    # ——— Production mode: --stock CODE ———
    if "--stock" in args:
        idx = args.index("--stock")
        stock_code = args[idx + 1] if idx + 1 < len(args) else ""

        if not stock_code or len(stock_code) != 6:
            print("错误: 请提供6位股票代码")
            print_usage()
            sys.exit(1)

        stock_name = ""
        if "--name" in args:
            ni = args.index("--name")
            stock_name = args[ni + 1] if ni + 1 < len(args) else ""

        industry = ""
        if "--industry" in args:
            ii = args.index("--industry")
            industry = args[ii + 1] if ii + 1 < len(args) else ""

        price = 0.0
        if "--price" in args:
            pi = args.index("--price")
            try:
                price = float(args[pi + 1])
            except (ValueError, IndexError):
                print("警告: 股价参数无效，将自动抓取")

        # Build minimal state — node0_fetch does the rest
        state = build_minimal_state(
            stock_code=stock_code,
            stock_name=stock_name,
            industry=industry,
            current_price=price,
        )

        print(f"\n▶ 启动 INVEST SOP 流水线 (LangGraph 生产模式)")
        print(f"  标的: {state['stock_name']} ({state['stock_code']})")
        print(f"  数据源: akshare 同花顺实时抓取")

        result = run_pipeline(state)
        sys.exit(0)

    # ——— No args ———
    print_usage()
