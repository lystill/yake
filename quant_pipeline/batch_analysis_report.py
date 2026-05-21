"""Generate clean analysis report from batch results."""
import json, sys

with open("reports/batch_summary_20260521_223116.json") as f:
    data = json.load(f)

ok = [r for r in data if r["status"] == "OK"]
errs = [r for r in data if r["status"] != "OK"]

# Separate A and B tier
a_tier = [r for r in ok if r["tier"] == "A"]
b_tier = [r for r in ok if r["tier"] == "B"]

def fmt_dev(d):
    if d > 0:
        return f"+{d:.0%}" if d < 10 else f"+{d:.0%} (极高偏离)"
    return f"{d:.0%}"

print("=" * 120)
print("  INVEST SOP v3.3 — 半导体板块批量全流程分析报告")
print("=" * 120)
print(f"  时间: 2026-05-21 22:23-22:31 | 耗时: 459s")
print(f"  标的: 30只 (A档豁免7 + B档头部23) | 成功: {len(ok)} | 失败: {len(errs)}")
print()

# ─── Section 1: A-tier (Exempt) ───
print("━" * 120)
print("  【A 档】战略豁免标的 — Branch E 动态PS估值（半导体设备/AI芯片防火墙）")
print("━" * 120)
print(f"  {'代码':<8} {'名称':<10} {'股价':>8} {'古典估值':>10} {'偏离':>14} {'涨幅空间':>10} {'估值框架'}")
print("  " + "-" * 110)
a_tier_sorted = sorted(a_tier, key=lambda r: -(r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0)
for r in a_tier_sorted:
    upside = (r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0
    flag = "🚀" if upside > 1.0 else "📈" if upside > 0.3 else "➡️" if upside > -0.1 else "⚠️"
    print(f"  {flag} {r['code']:<6} {r['name']:<10} {r['price']:>8.2f} {r['base_case_price']:>10.2f} "
          f"{fmt_dev(r['deviation']):>14} {upside:>+9.0%}   {r['framework']}")

print(f"\n  A档小结: 7只标的模型估算市值均显著高于当前股价，涨幅空间 {sum((r['base_case_price']/r['price']-1) for r in a_tier_sorted if r['price']>0)/len(a_tier_sorted):.0%} 均值。")
print("  注意: 高偏离度反映PS估值框架与传统PE框架的系统性差异，不构成买入建议。")

# ─── Section 2: B-tier top performers ───
print()
print("━" * 120)
print("  【B 档】高现金流头部标的 — 净现比>2.0，全流程定性+估值")
print("━" * 120)

# Sort by upside (descending)
b_tier_sorted = sorted(b_tier, key=lambda r: -(r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0)

# 2a: Undervalued (base > price + 20%)
undervalued = [r for r in b_tier_sorted if r["price"] > 0 and r["base_case_price"] / r["price"] > 1.2]
print(f"\n  ——— 模型低估（估值 > 现价 +20%）———")
print(f"  {'代码':<8} {'名称':<10} {'股价':>8} {'古典估值':>10} {'偏离':>12} {'涨幅空间':>10} {'利润驱动力':<16} {'框架'}")
print("  " + "-" * 110)
for r in undervalued:
    upside = (r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0
    print(f"  {r['code']:<8} {r['name']:<10} {r['price']:>8.2f} {r['base_case_price']:>10.2f} "
          f"{fmt_dev(r['deviation']):>12} {upside:>+9.0%}   {r.get('profit_driver',''):<16} {r['framework']}")

# 2b: Fairly valued (±20%)
fair = [r for r in b_tier_sorted if r["price"] > 0 and 0.8 <= r["base_case_price"] / r["price"] <= 1.2]
print(f"\n  ——— 估值合理（现价 ±20%）———")
print(f"  {'代码':<8} {'名称':<10} {'股价':>8} {'古典估值':>10} {'偏离':>12} {'涨幅空间':>10} {'利润驱动力':<16} {'框架'}")
print("  " + "-" * 110)
for r in fair:
    upside = (r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0
    print(f"  {r['code']:<8} {r['name']:<10} {r['price']:>8.2f} {r['base_case_price']:>10.2f} "
          f"{fmt_dev(r['deviation']):>12} {upside:>+9.0%}   {r.get('profit_driver',''):<16} {r['framework']}")

# 2c: Overvalued (base < price - 20%)
overvalued = [r for r in b_tier_sorted if r["price"] > 0 and r["base_case_price"] / r["price"] < 0.8]
print(f"\n  ——— 模型高估（估值 < 现价 -20%）———")
print(f"  {'代码':<8} {'名称':<10} {'股价':>8} {'古典估值':>10} {'偏离':>12} {'涨幅空间':>10} {'利润驱动力':<16} {'框架'}")
print("  " + "-" * 110)
for r in overvalued:
    upside = (r["base_case_price"] / r["price"] - 1) if r["price"] > 0 else 0
    flag = "💀" if upside < -0.8 else "🔻"
    print(f"  {flag} {r['code']:<6} {r['name']:<10} {r['price']:>8.2f} {r['base_case_price']:>10.2f} "
          f"{fmt_dev(r['deviation']):>12} {upside:>+9.0%}   {r.get('profit_driver',''):<16} {r['framework']}")

# ─── Section 3: Framework distribution ───
print()
print("━" * 120)
print("  【估值框架分布】")
print("━" * 120)
from collections import Counter
frameworks = Counter(r["framework"] for r in ok)
for fw, cnt in frameworks.most_common():
    stocks = [r["name"] for r in ok if r["framework"] == fw]
    print(f"  {fw:<25} {cnt:>2} 只  {', '.join(stocks)}")

# ─── Section 4: Profit driver distribution ───
print()
print("━" * 120)
print("  【利润驱动力分布】")
print("━" * 120)
drivers = Counter(r.get("profit_driver", "?") for r in ok)
for dr, cnt in drivers.most_common():
    stocks = [r["name"] for r in ok if r.get("profit_driver") == dr]
    print(f"  {dr:<20} {cnt:>2} 只  {', '.join(stocks)}")

# ─── Section 5: Failures ───
if errs:
    print()
    print("━" * 120)
    print("  【失败标的】— 需修复 bug 后重跑")
    print("━" * 120)
    for r in errs:
        print(f"  ✗ {r['code']} {r['name']}: {r['error']}")

print()
print("=" * 120)
print("  报告完毕")
print("=" * 120)
