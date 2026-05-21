"""
Batch quick screen for 100+ semiconductor stocks.
Pure Python rules engine — no LLM calls.
Outputs classification: BRANCH_E (exempt) / PASS / BRANCH_D / ABANDON
"""
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from nodes.node0_quick_screen import (
    _compute_max_rev_growth,
    _check_strategic_growth_exemption,
    _classify_industry,
)
from fetcher.financials import fetch_financial_data, fetch_current_price
from state import FinancialMetrics

# 100 semiconductor stocks from user list
STOCKS = [
    ("688146", "中船特气"), ("301369", "联动科技"), ("688449", "联芸科技"),
    ("688381", "帝奥微"), ("688693", "锴威特"), ("688478", "晶升股份"),
    ("688008", "澜起科技"), ("300604", "长川科技"), ("688807", "优迅股份"),
    ("688409", "富创精密"), ("688515", "裕太微-U"), ("001309", "德明利"),
    ("688535", "华海诚科"), ("688072", "拓荆科技"), ("603931", "格林达"),
    ("300672", "国科微"), ("688256", "寒武纪"), ("688120", "华海清科"),
    ("600584", "长电科技"), ("603061", "金海通"), ("301308", "江波龙"),
    ("603986", "兆易创新"), ("688141", "杰华特"), ("688200", "华峰测控"),
    ("688037", "芯源微"), ("688234", "天岳先进"), ("603078", "江化微"),
    ("688536", "思瑞浦"), ("688268", "华特气体"), ("688012", "中微公司"),
    ("688107", "安路科技"), ("688368", "晶丰明源"), ("688545", "兴福电子"),
    ("300661", "圣邦股份"), ("688766", "普冉股份"), ("300054", "鼎龙股份"),
    ("688596", "正帆科技"), ("688126", "沪硅产业"), ("688052", "纳芯微"),
    ("688361", "中科飞测"), ("688652", "京仪装备"), ("002371", "北方华创"),
    ("688048", "长光华芯"), ("688702", "盛科通信-U"), ("688359", "三孚新科"),
    ("688082", "盛美上海"), ("688785", "恒运昌"), ("688220", "翱捷科技-U"),
    ("688783", "西安奕材-U"), ("688347", "华虹公司"), ("688045", "必易微"),
    ("605358", "立昂微"), ("688279", "峰岹科技"), ("688530", "欧莱新材"),
    ("688508", "芯朋微"), ("688809", "强一股份"), ("301297", "富乐德"),
    ("688498", "源杰科技"), ("301629", "矽电股份"), ("002409", "雅克科技"),
    ("300831", "派瑞股份"), ("300782", "卓胜微"), ("300613", "富瀚微"),
    ("002156", "通富微电"), ("605111", "新洁能"), ("603375", "盛景微"),
    ("600360", "华微电子"), ("300706", "阿石创"), ("688699", "明微电子"),
    ("301095", "广立微"), ("688584", "上海合晶"), ("688416", "恒烁股份"),
    ("688525", "佰维存储"), ("688401", "路维光电"), ("688172", "燕东微"),
    ("300398", "飞凯材料"), ("688432", "有研硅"), ("688249", "晶合集成"),
    ("688213", "XD思特威-W"), ("300666", "江丰电子"), ("688123", "聚辰股份"),
    ("688099", "晶晨股份"), ("600641", "先导基电"), ("688362", "甬矽电子"),
    ("688720", "艾森股份"), ("688593", "新相微"), ("003026", "中晶科技"),
    ("300537", "广信材料"), ("688019", "安集科技"), ("688130", "晶华微"),
    ("688486", "龙迅股份"), ("688041", "海光信息"), ("600667", "太极实业"),
    ("300236", "上海新阳"), ("600206", "有研新材"), ("688729", "屹唐股份"),
    ("603005", "晶方科技"), ("688035", "德邦科技"), ("688605", "先锋精科"),
    ("688233", "神工股份"), ("688135", "利扬芯片"), ("688061", "灿瑞科技"),
    ("688981", "中芯国际"), ("688419", "耐科装备"), ("688548", "广钢气体"),
    ("688403", "汇成股份"), ("688601", "力芯微"), ("002213", "大为股份"),
    ("300077", "国民技术"), ("688173", "希荻微"), ("301348", "蓝箭电子"),
    ("688550", "瑞联新材"), ("688206", "概伦电子"), ("603690", "至纯科技"),
]


def screen_one(code: str, name: str) -> dict:
    """Fetch financials + run quick screen for one stock. Returns verdict dict."""
    result = {
        "code": code, "name": name,
        "verdict": "ERROR", "cash_ratio": 0.0, "category": "",
        "threshold": 0.0, "detail": "", "price": 0.0,
    }

    try:
        data = fetch_financial_data(code)
        if not data or not data.get("revenue") or sum(data["revenue"]) == 0:
            result["verdict"] = "FETCH_FAIL"
            result["detail"] = "财务数据抓取失败"
            return result

        # Build FinancialMetrics from dict
        fin = FinancialMetrics(
            revenue=data["revenue"],
            net_profit_parent=data["net_profit_parent"],
            op_cash_flow=data["op_cash_flow"],
            fixed_assets=data["fixed_assets"],
            total_assets=data["total_assets"],
            debt_ratio=data["debt_ratio"],
        )

        # Compute derived metrics
        ratios = []
        for i in range(len(fin.net_profit_parent)):
            if fin.net_profit_parent[i] > 0:
                ratios.append(fin.op_cash_flow[i] / fin.net_profit_parent[i])
        fin.cash_ratio_3y_avg = sum(ratios) / len(ratios) if ratios else 0.0
        result["cash_ratio"] = round(fin.cash_ratio_3y_avg, 2)

        # Get industry from research context (or default to 半导体)
        industry = "半导体"
        research_ctx = "半导体"

        # Fetch price
        try:
            price = fetch_current_price(code)
            result["price"] = price if price else 0.0
        except Exception:
            pass

        # Check strategic exemption
        yoy_growth = _compute_max_rev_growth(fin)
        exempt, exempt_reasons = _check_strategic_growth_exemption(
            code, name, industry, research_ctx, yoy_growth
        )

        if exempt:
            result["verdict"] = "BRANCH_E"
            result["detail"] = exempt_reasons[0][:80] if exempt_reasons else "战略豁免"
            return result

        # Q2: Cash flow health
        cat, threshold = _classify_industry(industry)
        result["category"] = cat
        result["threshold"] = threshold

        if fin.cash_ratio_3y_avg < threshold:
            result["verdict"] = "BRANCH_D"
            result["detail"] = f"净现比{fin.cash_ratio_3y_avg:.2f}<{cat}阈值{threshold}"
        else:
            result["verdict"] = "PASS"
            result["detail"] = f"净现比{fin.cash_ratio_3y_avg:.2f}≥{cat}阈值{threshold}"

    except Exception as e:
        result["verdict"] = "ERROR"
        result["detail"] = str(e)[:100]

    return result


def main():
    print(f"⚡ 批量快筛启动 — {len(STOCKS)} 只半导体标的")
    print(f"{'代码':<8} {'名称':<10} {'股价':>8} {'净现比':>7} {'行业分类':<20} {'阈值':>5} {'裁决':<12} 详情")
    print("-" * 130)

    results = []
    # Parallel fetch — 8 workers to avoid overwhelming akshare
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(screen_one, c, n): (c, n) for c, n in STOCKS}
        for i, future in enumerate(as_completed(futures), 1):
            r = future.result()
            results.append(r)
            verdict_mark = {"BRANCH_E": "🔥豁免", "PASS": "✅通过", "BRANCH_D": "❌熔断",
                          "ABANDON": "☠放弃", "FETCH_FAIL": "⚠数据缺失", "ERROR": "💥错误"}.get(r["verdict"], r["verdict"])
            print(f"{r['code']:<8} {r['name']:<10} {r['price']:>8.2f} {r['cash_ratio']:>7.2f} "
                  f"{r['category']:<20} {r['threshold']:>5.1f} {verdict_mark:<12} {r['detail'][:60]}")

    # Summary
    print("\n" + "=" * 130)
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    total = len(results)
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        label = {"BRANCH_E": "🔥 战略豁免", "PASS": "✅ 通过快筛", "BRANCH_D": "❌ 熔断(分支D)",
                 "FETCH_FAIL": "⚠ 数据缺失", "ERROR": "💥 异常"}.get(k, k)
        print(f"  {label}: {v}/{total} ({v/total:.0%})")

    # Export BRANCH_E list for full pipeline
    exempt_list = [r for r in results if r["verdict"] == "BRANCH_E"]
    pass_list = [r for r in results if r["verdict"] == "PASS"]
    if exempt_list:
        print(f"\n🔥 战略豁免标的（建议全流程分析）: {len(exempt_list)} 只")
        for r in exempt_list:
            print(f"  {r['code']} {r['name']}")
    if pass_list:
        print(f"\n✅ 通过快筛标的（建议精选全流程）: {len(pass_list)} 只")


if __name__ == "__main__":
    main()
