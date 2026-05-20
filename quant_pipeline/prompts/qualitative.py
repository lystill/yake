"""
Prompt templates for Node 1 — 行业定性 (L1/L2/L3 classification).
All prompts force the LLM to output strict JSON matching SOP vocabulary.
"""

# —— L1: Core Profit Driver ——
L1_PROFIT_DRIVER_PROMPT = """你是一名买方行业研究员。请根据以下公司的业务构成和盈利数据，
判断该公司的【核心利润驱动力】属于 INVEST_SOP 框架中的哪一类。

框架定义:
- "涨价驱动": 利润主要受大宗商品/产品价格波动驱动（如周期品、矿业、化工品、面板）
- "放量驱动": 利润主要受技术渗透/国产替代/消费爆品放量驱动（如半导体材料、SaaS、新能车）
- "稳定分红(C1)": 利润来自稳定特许经营权（如水电、高速、红利消费龙头）
- "离散事件(C2)": 利润来自离散不可预测事件（如创新药管线、重组、ST摘帽）

请输出严格 JSON（不要任何额外文字）:
{
  "profit_driver": "涨价驱动" | "放量驱动" | "稳定分红(C1)" | "离散事件(C2)",
  "confidence": 0.0-1.0,
  "reasoning": "不超过80字的定性依据，引用关键数据",
  "primary_segment": "核心利润贡献的业务板块名称"
}

---

公司数据:
{company_data}
"""

# —— L2: Penetration Stage (for 放量驱动 / Branch B) ——
L2_PENETRATION_PROMPT = """判断该公司核心业务的行业渗透率所处阶段。

阶段定义:
- "<5%导入期": 技术刚落地，主流客户未大规模采用，仍在烧钱阶段
- "5-30%爆发期": 渗透率快速爬坡，业绩加速兑现，竞争格局初定但未饱和
- ">30%成熟期": 增速放缓，从份额战转向利润战，开始拼成本和分红

判断依据: 行业国产化率、目标客户渗透率、渗透率年化提升速度。

请输出严格 JSON:
{
  "penetration_stage": "<5%导入期" | "5-30%爆发期" | ">30%成熟期",
  "current_rate_estimate": "当前渗透率估计值（如: 约18%）",
  "annual_improvement": "年化渗透率提升幅度（如: +3-5个百分点/年）",
  "reasoning": "不超过100字的判断依据"
}

---

公司数据:
{company_data}

定性标签（来自L1）:
{l1_labels}
"""

# —— L3: Competition / 卷度诊断 (for Branch B) ——
L3_COMPETITION_PROMPT = """对该公司的核心产品进行【卷度诊断】。

你需要判断两个关键变量:
1. 核心产品 ASP 年化变动趋势 (上行/稳定/年降<10%/年降10-15%/年降>15%)
2. 公司在该核心产品的市场份额趋势 (扩张/稳定/下滑)

根据 INVEST_SOP 风控红线:
- 若 ASP年化降幅>15% 且份额下滑 → PEG强行打7折，或直接切入成熟期PE
- 若 ASP降幅<10% 且份额稳定/上升 → 维持原PEG，按渗透率节奏正常切换

请输出严格 JSON:
{
  "asp_trend": "上行" | "稳定" | "年降<10%" | "年降10-15%" | "年降>15%",
  "market_share_trend": "扩张" | "稳定" | "下滑",
  "peg_haircut": 1.0或0.7或0.85,
  "competition_verdict": "不超过100字的竞争格局总结",
  "key_competitors": ["竞争对手1", "竞争对手2"],
  "supply_side_note": "供给侧CAPEX状态 / 二线厂商扩产节奏"
}

---

公司数据:
{company_data}

定性标签:
{l1_labels}
渗透率判断:
{l2_labels}
"""

# —— Combined: One-shot qualitative analysis ——
L1_TO_L3_COMBINED_PROMPT = """你是一名买方行业研究员，正在使用 INVEST_SOP 框架分析标的。
请基于以下公司数据，一次性完成三层定性判断，输出严格 JSON。

INVEST_SOP 框架速查:
—— L1 利润驱动力 ——
- "涨价驱动": 利润由产品价格波动主导 (周期品/大宗/矿业/化工)
- "放量驱动": 利润由技术渗透/国产替代/消费放量主导 (半导体材料/SaaS/新能车)
- "稳定/事件": 利润来自特许经营权 (水电/高速) 或离散事件 (创新药管线/重组)

—— L2 渗透率阶段 (仅放量驱动需答，其他标为空) ——
- "<5%导入期": 烧钱阶段，看愿景
- "5-30%爆发期": 业绩兑现，PEG + S曲线 + SOTP
- ">30%成熟期": 成熟PE + 股息率

—— L3 卷度诊断 ——
- ASP年化变动: 上行/稳定/年降<10%/年降10-15%/年降>15%
- 份额趋势: 扩张/稳定/下滑
- 红线: ASP降>15%且份额下滑 → PEG × 0.7

请输出严格 JSON (不要任何额外文字):
{{
  "L1": {{
    "profit_driver": "涨价驱动" | "放量驱动" | "稳定分红(C1)" | "离散事件(C2)",
    "confidence": 0.0-1.0,
    "reasoning": "不超过80字的依据",
    "primary_segment": "核心利润贡献板块"
  }},
  "L2": {{
    "penetration_stage": "<5%导入期" | "5-30%爆发期" | ">30%成熟期" | "",
    "current_rate_estimate": "渗透率估计",
    "annual_improvement": "年化提升幅度",
    "reasoning": "不超过100字的依据"
  }},
  "L3": {{
    "asp_trend": "上行" | "稳定" | "年降<10%" | "年降10-15%" | "年降>15%",
    "market_share_trend": "扩张" | "稳定" | "下滑",
    "peg_haircut": 1.0,
    "competition_verdict": "不超过100字的竞争格局总结",
    "key_competitors": [],
    "supply_side_note": "供给侧状态"
  }}
}}

---

公司数据:
{company_data}
"""
