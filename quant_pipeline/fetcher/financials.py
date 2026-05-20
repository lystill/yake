"""
Production financial data fetcher using akshare (同花顺数据源).
Fetches real A-share income statement, balance sheet, and cash flow data.
"""

import re
import pandas as pd
from typing import Optional

try:
    import akshare as ak
    AK_AVAILABLE = True
except ImportError:
    AK_AVAILABLE = False


def _parse_amount(value) -> float:
    """Parse Chinese-unit amount string to float (亿)."""
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return 0.0
    value = value.strip()
    if not value or value in ("False", "None", "—", "-"):
        return 0.0
    if "亿" in value:
        return float(value.replace("亿", ""))
    if "万" in value:
        return float(value.replace("万", "")) / 10000.0
    # Assume already in 亿 or plain number
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


def _latest_n_years(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Return the most recent N complete fiscal years, sorted chronologically."""
    # Filter to rows where 报告期 is a 4-digit year
    df = df.copy()
    df["_year"] = df["报告期"].astype(str).str.extract(r"(\d{4})")
    df = df.dropna(subset=["_year"])
    df["_year"] = df["_year"].astype(int)
    # Sort descending, take top N, then re-sort ascending
    recent = df.sort_values("_year", ascending=False).head(n)
    return recent.sort_values("_year", ascending=True)


def _get_total_shares(stock_code: str) -> float:
    """Fetch total shares (亿股) for the given stock code."""
    # Method 1: stock_individual_info_em
    try:
        info = ak.stock_individual_info_em(symbol=stock_code)
        row = info[info["item"] == "总股本"]
        if not row.empty:
            shares_str = str(row.iloc[0]["value"])
            result = _parse_shares_to_yi(shares_str)
            if result > 0:
                return result
    except Exception:
        pass

    # Method 2: parse from balance sheet (实收资本)
    try:
        debt = ak.stock_financial_debt_ths(symbol=stock_code, indicator="按年度")
        row = debt.head(1).iloc[0]
        实收资本 = _parse_amount(row.get("实收资本（或股本）", "0"))
        # 实收资本 in 亿元 → convert to 亿股 (assuming 1元 par value)
        if 实收资本 > 0:
            return 实收资本
    except Exception:
        pass

    return 0.0


def _parse_shares_to_yi(value: str) -> float:
    """Parse shares string to 亿股."""
    value = value.strip()
    if "亿" in value:
        return float(value.replace("亿", ""))
    if "万" in value:
        return float(value.replace("万", "")) / 10000.0
    # Assume raw number in 股
    try:
        return float(value) / 1e8
    except ValueError:
        return 0.0


def fetch_financial_data(stock_code: str) -> dict:
    """
    Fetch real financial data for an A-share stock.

    Args:
        stock_code: 6-digit A-share code (e.g. '600497', '002409')

    Returns:
        dict with keys:
          - revenue: list[float]           — 近3年营收 (亿), chrono order
          - net_profit_parent: list[float] — 近3年归母净利润 (亿)
          - op_cash_flow: list[float]      — 近3年经营现金流 (亿)
          - fixed_assets: float            — 最新年固定资产 (亿)
          - total_assets: float            — 最新年总资产 (亿)
          - debt_ratio: float              — 最新年资产负债率 (%)
          - total_shares: float            — 总股本 (亿股)
          - years: list[int]               — 对应的财年
    """
    if not AK_AVAILABLE:
        raise ImportError("akshare is required for real data fetching")

    result = {
        "revenue": [],
        "net_profit_parent": [],
        "op_cash_flow": [],
        "fixed_assets": 0.0,
        "total_assets": 0.0,
        "debt_ratio": 0.0,
        "total_shares": 0.0,
        "years": [],
    }

    # ——— 1. Abstract (营收, 净利润, 资产负债率) ———
    try:
        abst = ak.stock_financial_abstract_ths(symbol=stock_code, indicator="按年度")
        recent = _latest_n_years(abst, 3)
        for _, row in recent.iterrows():
            result["revenue"].append(_parse_amount(row.get("营业总收入", 0)))
            result["net_profit_parent"].append(_parse_amount(row.get("净利润", 0)))
            result["years"].append(int(row["_year"]))
        # Debt ratio from latest year
        latest = abst[abst["报告期"].astype(str).str.contains(r"\d{4}")].tail(1)
        if not latest.empty:
            result["debt_ratio"] = _parse_pct(latest.iloc[0].get("资产负债率", "0"))
    except Exception as e:
        print(f"  [WARN] Failed to fetch abstract data: {e}")

    # ——— 2. Cash flow (经营现金流) ———
    try:
        cf = ak.stock_financial_cash_ths(symbol=stock_code, indicator="按年度")
        # Cash flow is in descending order; filter to same years
        target_years = set(result["years"])
        cf["_year"] = cf["报告期"].astype(str).str.extract(r"(\d{4})").astype(float)
        cf_match = cf[cf["_year"].isin(target_years)].sort_values("_year", ascending=True)
        for _, row in cf_match.iterrows():
            result["op_cash_flow"].append(
                _parse_amount(row.get("*经营活动产生的现金流量净额", 0))
            )
        # If no match (different year range), take latest 3
        if not result["op_cash_flow"]:
            recent_cf = _latest_n_years(cf, 3)
            for _, row in recent_cf.iterrows():
                result["op_cash_flow"].append(
                    _parse_amount(row.get("*经营活动产生的现金流量净额", 0))
                )
    except Exception as e:
        print(f"  [WARN] Failed to fetch cash flow data: {e}")

    # ——— 3. Balance sheet (固定资产, 总资产) ———
    try:
        debt = ak.stock_financial_debt_ths(symbol=stock_code, indicator="按年度")
        # Balance sheet is in descending order; latest year is row 0
        debt["_year"] = debt["报告期"].astype(str).str.extract(r"(\d{4})").astype(float)
        latest_year = max(result["years"]) if result["years"] else None
        if latest_year:
            latest_row = debt[debt["_year"] == latest_year]
            if not latest_row.empty:
                row = latest_row.iloc[0]
                result["fixed_assets"] = _parse_amount(row.get("固定资产合计", 0))
                result["total_assets"] = _parse_amount(row.get("*资产合计", 0))
        # Fallback: use first row (most recent)
        if result["total_assets"] == 0.0 and not debt.empty:
            row = debt.iloc[0]
            result["fixed_assets"] = _parse_amount(row.get("固定资产合计", 0))
            result["total_assets"] = _parse_amount(row.get("*资产合计", 0))
    except Exception as e:
        print(f"  [WARN] Failed to fetch balance sheet data: {e}")

    # ——— 4. Total shares (总股本) ———
    try:
        result["total_shares"] = _get_total_shares(stock_code)
    except Exception as e:
        print(f"  [WARN] Failed to fetch total shares: {e}")

    return result


def fetch_current_price(stock_code: str) -> float:
    """Fetch latest market price via akshare, then 4 direct HTTP fallbacks."""
    import requests

    # Source 1: akshare daily history
    try:
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",
            start_date="20260101",
            end_date="20261231",
            adjust="qfq",
        )
        if not df.empty:
            price = float(df.iloc[-1]["收盘"])
            if price > 0:
                return price
    except Exception:
        pass

    # Source 2: akshare spot market (东方财富)
    try:
        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == stock_code]
        if not row.empty:
            price = float(row.iloc[0]["最新价"])
            if price > 0:
                return price
    except Exception:
        pass

    # Source 3: 东方财富 push API
    try:
        exchange = "1" if stock_code.startswith("6") else "0"
        url = f"https://push2.eastmoney.com/api/qt/stock/get?secid={exchange}.{stock_code}&fields=f43"
        r = requests.get(url, timeout=5)
        price = r.json().get("data", {}).get("f43", 0) / 100
        if price > 0:
            return price
    except Exception:
        pass

    # Source 4: 腾讯行情
    try:
        prefix = "sh" if stock_code.startswith("6") else "sz"
        url = f"https://qt.gtimg.cn/q={prefix}{stock_code}"
        r = requests.get(url, timeout=5)
        price = float(r.text.split("~")[3])
        if price > 0:
            return price
    except Exception:
        pass

    # Source 5: 新浪财经
    try:
        prefix = "sh" if stock_code.startswith("6") else "sz"
        url = f"https://hq.sinajs.cn/list={prefix}{stock_code}"
        r = requests.get(url, timeout=5)
        price = float(r.text.split(",")[3])
        if price > 0:
            return price
    except Exception:
        pass

    # Source 6: 雪球
    try:
        exchange = "SH" if stock_code.startswith("6") else "SZ"
        url = f"https://stock.xueqiu.com/v5/stock/quote.json?symbol={exchange}{stock_code}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        price = r.json()["data"]["quote"]["current"]
        if price > 0:
            return price
    except Exception:
        pass

    return 0.0
