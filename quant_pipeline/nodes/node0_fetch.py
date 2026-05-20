"""
Node 0 — 数据抓取与状态初始化
===============================
First-stage node: accepts a minimal stock_code, fetches all real
financial data via akshare, and enriches the PipelineState before
the screening/qualitative nodes run.

数据源:
  - 同花顺 (THS): 利润表 / 现金流 / 资产负债表 → akshare
  - 东方财富 (EM):  实时股价 / 行业分类 / 个股信息
  - 本地研报:       research_reports/ 目录
  - 选股池:        stock_pool.txt

Architecture:
  Input:  state with stock_code, stock_name, industry (optional)
  Output: state enriched with financials, current_price, total_shares,
          segments, research_context, pipeline_start_ts
"""

import datetime
import time
import os
from state import PipelineState, FinancialMetrics, BusinessSegment

try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False


# ═══════════════════════════════════════════════════════════
#  Stock Pool Reader
# ═══════════════════════════════════════════════════════════

def read_stock_pool(pool_path: str = "stock_pool.txt") -> list[dict]:
    """
    Parse stock_pool.txt and return list of {code, name, industry, note} dicts.

    File format (one stock per line):
      002409 雅克科技    半导体材料   国产替代龙头
      Lines starting with # are comments, blank lines ignored.
    """
    stocks = []
    if not os.path.exists(pool_path):
        return stocks

    with open(pool_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                stocks.append({
                    "code": parts[0],
                    "name": parts[1],
                    "industry": parts[2] if len(parts) > 2 else "",
                    "note": " ".join(parts[3:]) if len(parts) > 3 else "",
                })
    return stocks


# ═══════════════════════════════════════════════════════════
#  Industry → (分类, 熔断阈值) lookup
# ═══════════════════════════════════════════════════════════

INDUSTRY_MAP = {
    "半导体": "半导体材料",
    "电子": "电子材料",
    "芯片": "半导体材料",
    "软件": "软件",
    "白酒": "白酒",
    "医药": "高端消费",
    "化工": "化工",
    "有色": "有色金属",
    "钢铁": "钢铁",
    "水泥": "水泥",
    "煤炭": "煤炭",
    "军工": "军工配套",
    "建筑": "建筑工程",
    "银行": "ToG/B端重资产",
    "地产": "PPP",
    "电力": "ToG/B端重资产",
}


def _infer_industry(stock_code: str) -> str:
    """Try to infer industry from akshare, fallback to generic."""
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        row = info[info["item"] == "行业"]
        if not row.empty:
            raw = str(row.iloc[0]["value"])
            for keyword, industry in INDUSTRY_MAP.items():
                if keyword in raw:
                    return industry
            return raw
    except Exception:
        pass
    return ""


def _parse_cn_amount(value) -> float:
    """Parse Chinese-unit amount string to float (亿)."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    value = value.strip()
    if not value or value in ("False", "None", "—", "-", ""):
        return 0.0
    if "亿" in value:
        return float(value.replace("亿", "").strip())
    if "万" in value:
        return float(value.replace("万", "").strip()) / 10000.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_pct(value) -> float:
    """Parse percentage string to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    value = value.strip().replace("%", "").strip()
    try:
        return float(value)
    except ValueError:
        return 0.0


def _latest_n_years(df, n: int = 3):
    """Return most recent N fiscal years, sorted chronologically."""
    import pandas as pd
    df = df.copy()
    df["_year"] = df["报告期"].astype(str).str.extract(r"(\d{4})")
    df = df.dropna(subset=["_year"])
    df["_year"] = df["_year"].astype(int)
    recent = df.sort_values("_year", ascending=False).head(n)
    return recent.sort_values("_year", ascending=True)


# ═══════════════════════════════════════════════════════════
#  Main node
# ═══════════════════════════════════════════════════════════

def node0_fetch(state: PipelineState) -> PipelineState:
    """
    Fetch real financial data, price, and shares for the stock.

    数据抓取顺序:
      1. 同花顺利润表 → revenue, net_profit, debt_ratio
      2. 同花顺现金流表 → op_cash_flow
      3. 同花顺资产负债表 → fixed_assets, total_assets, total_shares
      4. 东方财富历史行情 → current_price
      5. 东方财富个股信息 → industry
      6. 本地研报目录 → research_context
      7. 构建 FinancialMetrics + BusinessSegment

    If state already contains populated financials (demo mode), this is a no-op.
    On critical failure: sets state['error'] so the pipeline routes to END.
    Non-critical failures: fallback to defaults, pipeline continues.
    """
    t_start = time.perf_counter()

    # ——— No-op if financial data already populated (demo mode) ———
    fin = state.get("financials")
    if fin is not None and getattr(fin, "revenue", None) and len(fin.revenue) == 3:
        print(f"\n  ⏩ [Node 0 Fetch] 检测到预填充财务数据，跳过实时抓取 (demo/手动模式)")
        _init_defaults(state)
        state["pipeline_start_ts"] = datetime.datetime.now().isoformat()
        state["graph_topology"] = "并行 DAG (3 Agent + SOTP + CIO双轨调和 + 反思闭环) v3.1"
        return state

    stock_code = state.get("stock_code", "")
    stock_name = state.get("stock_name", "")
    if not stock_code or len(stock_code) != 6:
        state["error"] = f"无效股票代码: {stock_code}"
        return state

    if not AK_AVAILABLE:
        state["error"] = "akshare 未安装，无法抓取实时数据。请执行: pip install akshare"
        return state

    print(f"\n  ⏳ [Node 0 Fetch] 抓取 {stock_name}({stock_code}) 实时数据...")

    years = []
    revenue = []
    net_profit = []
    op_cash_flow = []
    fixed_assets = 0.0
    construction_in_progress = 0.0
    total_assets = 0.0
    total_shares = 0.0
    debt_ratio = 0.0
    current_price = 0.0
    fetch_errors = []

    # ——— 1. Profit / Revenue / Debt ratio (同花顺摘要) ———
    try:
        abst = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        recent = _latest_n_years(abst, 3)
        for _, row in recent.iterrows():
            years.append(int(row["_year"]))
            revenue.append(_parse_cn_amount(row.get("营业总收入", 0)))
            net_profit.append(_parse_cn_amount(row.get("净利润", 0)))

        latest_abst = abst[abst["报告期"].astype(str).str.contains(r"\d{4}")].tail(1)
        if not latest_abst.empty:
            debt_ratio = _parse_pct(latest_abst.iloc[0].get("资产负债率", "0"))

        print(f"    ✓ 利润表: 营收 {[f'{v:.1f}亿' for v in revenue]} | 负债率 {debt_ratio:.1f}%")
    except Exception as e:
        fetch_errors.append(f"利润表: {e}")
        print(f"    ✗ 利润表抓取失败: {e}")

    if not years:
        state["error"] = f"利润表抓取失败，无法继续: {'; '.join(fetch_errors)}"
        return state

    # ——— 2. Cash flow (同花顺现金流) ———
    try:
        cf = ak.stock_financial_cash_ths(symbol=stock_code, indicator="按年度")
        cf["_year"] = cf["报告期"].astype(str).str.extract(r"(\d{4})").astype(float)
        target_years = set(years)
        cf_match = cf[cf["_year"].isin(target_years)].sort_values("_year", ascending=True)
        for _, row in cf_match.iterrows():
            op_cash_flow.append(_parse_cn_amount(row.get("*经营活动产生的现金流量净额", 0)))
        if not op_cash_flow:
            recent_cf = _latest_n_years(cf, 3)
            for _, row in recent_cf.iterrows():
                op_cash_flow.append(_parse_cn_amount(row.get("*经营活动产生的现金流量净额", 0)))
        print(f"    ✓ 现金流: {[f'{v:.2f}亿' for v in op_cash_flow]}")
    except Exception as e:
        fetch_errors.append(f"现金流: {e}")
        print(f"    ⚠ 现金流抓取失败: {e}")

    # ——— 3. Balance sheet (同花顺资产负债表) ———
    try:
        debt = ak.stock_financial_debt_ths(symbol=stock_code, indicator="按年度")
        debt["_year"] = debt["报告期"].astype(str).str.extract(r"(\d{4})").astype(float)
        latest_year = max(years) if years else None
        if latest_year:
            latest_row = debt[debt["_year"] == latest_year]
            if not latest_row.empty:
                row = latest_row.iloc[0]
                fixed_assets = _parse_cn_amount(row.get("固定资产合计", 0))
                construction_in_progress = _parse_cn_amount(row.get("在建工程", 0))
                total_assets = _parse_cn_amount(row.get("*资产合计", 0))
                total_shares = _parse_cn_amount(row.get("实收资本（或股本）", 0))
        # Fallback: use latest available row
        if total_assets == 0.0 and not debt.empty:
            latest_row = debt.sort_values("_year", ascending=False).iloc[0]
            fixed_assets = _parse_cn_amount(latest_row.get("固定资产合计", 0))
            construction_in_progress = _parse_cn_amount(latest_row.get("在建工程", 0))
            total_assets = _parse_cn_amount(latest_row.get("*资产合计", 0))
            total_shares = _parse_cn_amount(latest_row.get("实收资本（或股本）", 0))
        # If total_shares still 0, try to get from 东方财富
        if total_shares <= 0:
            try:
                info = ak.stock_individual_info_em(symbol=stock_code)
                row = info[info["item"] == "总股本"]
                if not row.empty:
                    raw = str(row.iloc[0]["value"])
                    total_shares = _parse_cn_amount(raw.replace("股", "").strip())
            except Exception:
                pass
        print(f"    ✓ 资产负债表: 固资{fixed_assets:.1f}亿 在建{construction_in_progress:.1f}亿 总资产{total_assets:.1f}亿 股本{total_shares:.2f}亿")
    except Exception as e:
        fetch_errors.append(f"资产负债表: {e}")
        print(f"    ⚠ 资产负债表抓取失败: {e}")

    # ——— 4. Current price (6 源回退: akshare→东财API→腾讯→新浪→雪球→手动) ———
    price_sources = []
    start = (datetime.date.today() - datetime.timedelta(days=30)).strftime("%Y%m%d")
    end = datetime.date.today().strftime("%Y%m%d")

    # Source A: akshare 东方财富历史行情
    try:
        hist = ak.stock_zh_a_hist(symbol=stock_code, period="daily",
                                  start_date=start, end_date=end, adjust="qfq")
        if not hist.empty:
            price_sources.append(float(hist.iloc[-1]["收盘"]))
    except Exception:
        pass

    # Source B: akshare 东方财富实时快照
    if not price_sources:
        try:
            spot = ak.stock_zh_a_spot_em()
            match = spot[spot["代码"] == stock_code]
            if not match.empty:
                price_sources.append(float(match.iloc[0]["最新价"]))
        except Exception:
            pass

    # Source C: akshare 腾讯历史行情
    if not price_sources:
        try:
            hist_tx = ak.stock_zh_a_hist_tx(symbol=stock_code, period="daily",
                                             start_date=start, end_date=end, adjust="qfq")
            if hist_tx is not None and not hist_tx.empty:
                price_sources.append(float(hist_tx.iloc[-1]["收盘"]))
        except Exception:
            pass

    # Source D: akshare 雪球个股报价
    if not price_sources:
        try:
            xq = ak.stock_individual_spot_xq(symbol=stock_code)
            if xq is not None and not xq.empty:
                price_sources.append(float(xq.iloc[0]["现价"]))
        except Exception:
            pass

    # Source E: 东方财富 push API (直接 HTTP, 不依赖 akshare)
    if not price_sources:
        try:
            import requests as _req
            exchange = "1" if stock_code.startswith("6") else "0"
            url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={exchange}.{stock_code}&fields=f43"
            r = _req.get(url, timeout=5)
            p = r.json().get("data", {}).get("f43", 0) / 100
            if p > 0:
                price_sources.append(p)
        except Exception:
            pass

    # Source F: 腾讯行情 (直接 HTTP, 不依赖 akshare)
    if not price_sources:
        try:
            import requests as _req
            prefix = "sh" if stock_code.startswith("6") else "sz"
            url = f"https://qt.gtimg.cn/q={prefix}{stock_code}"
            r = _req.get(url, timeout=5)
            p = float(r.text.split("~")[3])
            if p > 0:
                price_sources.append(p)
        except Exception:
            pass

    if price_sources:
        current_price = price_sources[0]
        print(f"    ✓ 实时股价: {current_price:.2f} 元")
    else:
        current_price = state.get("current_price", 0.0)
        if current_price > 0:
            print(f"    ⚠ 全部 6 源抓取失败，使用手动指定: {current_price} 元")
        else:
            fetch_errors.append("股价: 全部 6 源失败，请用 --price 手动指定")
            print(f"    ✗ 股价抓取全部失败，请用 --price 手动指定股价")

    # ——— 5. Industry inference ———
    industry = state.get("industry", "")
    if not industry:
        industry = _infer_industry(stock_code)
        if not industry:
            # Try stock_pool.txt for pre-defined industry
            pool = read_stock_pool()
            for s in pool:
                if s["code"] == stock_code:
                    industry = s["industry"]
                    break
        if not industry:
            industry = "标准制造"
        state["industry"] = industry
        print(f"    ✓ 行业推断: {industry}")

    # ——— 6. Research context ———
    research_context = state.get("research_context", "")
    if not research_context:
        try:
            from fetcher.research import fetch_research_report
            research_context = fetch_research_report(stock_code, stock_name)
            if research_context:
                print(f"    ✓ 研报加载: {len(research_context)} 字符")
            else:
                print(f"    ○ 研报: 无可用研报")
        except Exception as e:
            print(f"    ○ 研报加载跳过: {e}")
    state["research_context"] = research_context or ""

    # ——— 7. Build FinancialMetrics ———
    fin = FinancialMetrics(
        revenue=revenue if revenue else [0, 0, 0],
        net_profit_parent=net_profit if net_profit else [0, 0, 0],
        op_cash_flow=op_cash_flow if op_cash_flow else [0, 0, 0],
        fixed_assets=fixed_assets,
        construction_in_progress=construction_in_progress,
        total_assets=total_assets,
        debt_ratio=debt_ratio,
    )
    state["financials"] = fin
    state["current_price"] = current_price if current_price > 0 else state.get("current_price", 0.0)
    state["total_shares"] = total_shares if total_shares > 0 else state.get("total_shares", 0.0)

    # ——— 8. Segments (try real data, fallback to merged) ———
    if not state.get("segments"):
        state["segments"] = _build_fallback_segments(fin, stock_name or stock_code)

    # ——— 9. Defaults for flow control ———
    _init_defaults(state)
    state["pipeline_start_ts"] = datetime.datetime.now().isoformat()
    state["graph_topology"] = "并行 DAG (3 Agent + SOTP + 反思闭环)"

    elapsed = (time.perf_counter() - t_start) * 1000
    if fetch_errors:
        print(f"    ⚠ [Node 0 Fetch] 完成 ({elapsed:.0f}ms), {len(fetch_errors)} 个非关键错误")
    else:
        print(f"    ✓ [Node 0 Fetch] 数据抓取完成 ({elapsed:.0f}ms)\n")

    return state


def _init_defaults(state: PipelineState) -> None:
    """Set pipeline defaults."""
    state.setdefault("governance_flags", [])
    state.setdefault("reflection_round", 0)
    state.setdefault("valuation_deviation", 0.0)
    state.setdefault("reflection_triggered", False)
    state.setdefault("macro_beta", "进攻防线")
    state.setdefault("segment_valuations", [])
    state.setdefault("label_conflicts", [])
    state.setdefault("sotp_trigger_reasons", [])


def _build_fallback_segments(fin: FinancialMetrics, stock_name: str) -> list[BusinessSegment]:
    """Single-segment fallback when no segment breakdown is available."""
    rev = fin.revenue
    np_ = fin.net_profit_parent
    latest_rev = rev[-1] if rev else 0
    latest_np = np_[-1] if np_ else 0
    rev_growth = (rev[-1] / rev[-2] - 1) * 100 if len(rev) >= 2 and rev[-2] > 0 else 0

    return [
        BusinessSegment(
            name=f"{stock_name}（合并口径）",
            revenue=latest_rev,
            revenue_share=100.0,
            profit_contribution=latest_np,
            profit_share=100.0,
            gross_margin=0.0,
            yoy_growth=rev_growth,
            roe=0.0,
            branch="",
        )
    ]
