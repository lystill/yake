"""
Batch full-pipeline analysis for A+B tier stocks.
A-tier: 7 exempt (BRANCH_E) → Branch E dynamic PS valuation
B-tier: 23 top cash-ratio (>2.0) → full qualitative chain + valuation

Runs with limited parallelism to avoid LLM API rate limits.
Saves individual reports to reports/ directory.
"""
import sys
import os
import time
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))
from utils import config as _llm_config

from main import build_minimal_state
from graph import run_pipeline

# ─── A-tier: 7 exempt stocks ───
A_TIER = [
    ("688072", "拓荆科技"),
    ("688256", "寒武纪"),
    ("688120", "华海清科"),
    ("688012", "中微公司"),
    ("002371", "北方华创"),
    ("688082", "盛美上海"),
    ("688041", "海光信息"),
]

# ─── B-tier: cash-ratio > 2.0 (23 stocks) ───
B_TIER = [
    ("688362", "甬矽电子"),
    ("688213", "XD思特威-W"),
    ("605358", "立昂微"),
    ("002156", "通富微电"),
    ("688135", "利扬芯片"),
    ("688347", "华虹公司"),
    ("301348", "蓝箭电子"),
    ("688045", "必易微"),
    ("688368", "晶丰明源"),
    ("688981", "中芯国际"),
    ("300672", "国科微"),
    ("300537", "广信材料"),
    ("688206", "概伦电子"),
    ("600360", "华微电子"),
    ("603986", "兆易创新"),
    ("688249", "晶合集成"),
    ("600584", "长电科技"),
    ("688403", "汇成股份"),
    ("688584", "上海合晶"),
    ("688233", "神工股份"),
    ("300398", "飞凯材料"),
    ("688548", "广钢气体"),
    ("688550", "瑞联新材"),
]


def analyze_one(code: str, name: str, tier: str) -> dict:
    """Run full pipeline for one stock. Returns summary dict."""
    start = time.time()
    result = {
        "code": code, "name": name, "tier": tier,
        "status": "ERROR", "verdict": "", "framework": "",
        "price": 0.0, "base_case_price": 0.0, "deviation": 0.0,
        "latency_s": 0.0, "error": "",
    }
    try:
        state = build_minimal_state(stock_code=code, stock_name=name, industry="半导体")
        final = run_pipeline(state)

        result["status"] = "OK"
        result["verdict"] = final.get("screening_verdict", "")
        result["price"] = final.get("current_price", 0.0)
        val = final.get("valuation", {})
        if hasattr(val, "base_case_price"):
            result["base_case_price"] = val.base_case_price
        elif isinstance(val, dict):
            result["base_case_price"] = val.get("base_case_price", 0.0)
        result["framework"] = final.get("suggested_framework", "")
        result["deviation"] = final.get("valuation_deviation", 0.0)
        result["profit_driver"] = final.get("profit_driver", "")

    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]

    result["latency_s"] = round(time.time() - start, 1)
    return result


def main():
    os.makedirs("reports", exist_ok=True)

    all_stocks = [(c, n, "A") for c, n in A_TIER] + [(c, n, "B") for c, n in B_TIER]
    total = len(all_stocks)
    print(f"⚡ 批量全流程分析启动 — {total} 只标的 (A档{len(A_TIER)} + B档{len(B_TIER)})")
    print(f"  并发: 4 workers | 模型: DeepSeek")
    print(f"  开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print("-" * 100)

    results = []
    completed = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(analyze_one, c, n, t): (c, n, t) for c, n, t in all_stocks}
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            completed += 1
            elapsed = time.time() - t0
            status_icon = {"OK": "✓", "ERROR": "✗"}.get(r["status"], "?")
            print(f"[{completed:>2}/{total}] {status_icon} {r['code']} {r['name']:<10} "
                  f"| {r['tier']}档 | 裁决:{r.get('verdict','?'):<12} | "
                  f"股价:{r.get('price',0):.2f} | 估值:{r.get('base_case_price',0):.2f} | "
                  f"偏离:{r.get('deviation',0):.1%} | {r['latency_s']:.0f}s")

    # ─── Summary ───
    print("\n" + "=" * 100)
    total_time = time.time() - t0
    print(f"⏱ 总耗时: {total_time:.0f}s ({total_time/total:.1f}s/标的)")
    print(f"  结束时间: {datetime.now().strftime('%H:%M:%S')}")

    ok = [r for r in results if r["status"] == "OK"]
    errs = [r for r in results if r["status"] != "OK"]
    print(f"\n  成功: {len(ok)}/{total} | 失败: {len(errs)}/{total}")

    if errs:
        print(f"\n  ✗ 失败列表:")
        for r in errs:
            print(f"    {r['code']} {r['name']}: {r['error'][:100]}")

    # Save summary JSON
    summary_path = f"reports/batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()
