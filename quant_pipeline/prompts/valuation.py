"""
Prompt templates for Node 2 — Valuation (SOTP check + branch-specific prompts).
Each branch of the SOP tree gets a carefully scoped valuation prompt.
"""

# ——— SOTP Valuation Prompt ———
SOTP_VALUATION_PROMPT = """你是一名买方估值分析师。该标的已触发 SOTP（分部加总估值）条件，
必须对各业务板块独立估值后加总。

请基于以下信息，为每个业务板块给出:
1. 对该板块适用的 SOP 分支/框架
2. 可比公司及可比 PE/PEG/PB
3. 该板块 2026E 净利润贡献
4. 合理的估值倍数及折溢价理由
5. 该板块估值 = 净利润 × PE

最后加总，并给出综合隐含 PE。

{macro_context}

请输出严格 JSON:
{{
  "segments": [
    {{
      "name": "板块名称",
      "branch": "分支A/B/C1/C2",
      "framework": "PB-ROE/PE-PEG/PS/DCF/...",
      "net_profit_2026e": 数字(亿元),
      "pe_assigned": 数字,
      "comparable_companies": ["可比1", "可比2"],
      "discount_premium_reason": "折溢价理由",
      "segment_value": 数字(亿元)
    }}
  ],
  "total_value": 数字(亿元),
  "total_shares": 数字(亿股),
  "implied_price": 数字(元/股),
  "composite_pe_2026e": 数字,
  "key_assumptions": ["假设1", "假设2"]
}}

---

公司数据:
{company_data}

定性标签:
{labels}

SOTP触发依据:
{sotp_reasons}
"""

# ——— Single Framework Valuation Prompt (Branch A / C) ———
SINGLE_FRAMEWORK_VALUATION_PROMPT = """你是一名买方估值分析师。该标的适用单一估值框架（SOTP未触发）。

请基于以下信息，确定合理的估值锚:
1. 识别适用的 SOP 框架（分支 A/B/C1/C2）
2. 选择估值倍数（PE/PB/股息率 等）并给出对标依据
3. 计算估值区间（悲观/中性/乐观）
4. 列出关键风险

{macro_context}

请输出严格 JSON:
{{
  "framework": "分支A-PE弹性/分支A-PB底部/分支C1-DDM/...",
  "framework_reasoning": "为什么选这个框架",
  "base_metric_2026e": 数字,
  "multiple_assigned": 数字,
  "multiple_reasoning": "倍数选择的依据和对标",
  "base_case_value": 数字(亿元),
  "base_case_price": 数字(元/股),
  "bull_case_price": 数字(元/股),
  "bear_case_price": 数字(元/股),
  "comparable_companies": ["可比1", "可比2"],
  "key_risks": ["风险1", "风险2"],
  "recommendation": "不超过150字的买方立场总结"
}}

---

公司数据:
{company_data}

定性标签:
{labels}
"""

# ——— Branch E: Strategic Growth PS Valuation (v3.2) ———
BRANCH_E_VALUATION_PROMPT = """你是一名买方成长股估值分析师。该标的已触发 **v3.2 战略成长股豁免（分支E）**。

## 核心原则
这是一个高壁垒硬科技烧钱期标的（AI芯片/先进封装/IC载板等），**禁止使用清算价值法或PE估值法**。
这类公司的价值锚是**收入规模 × 行业市销率（PS）**，而非当期利润或现金流。

## 动态行业 PS 定价规则
基线 PS: **35x**（战略硬科技成长股标准倍数）
超高增速溢价: 若最新年度收入 YoY ≥ 100%，PS 上调至 **40x**
注意: 此倍数只适用于硬科技明星赛道，普通工业品不得使用。

## 估值步骤
1. 确认公司所处的科技赛道（AI芯片/先进封装/IC载板/光刻胶/碳化硅 等）
2. 选取可比 PS 倍数（对标国内外同赛道公司：英伟达/AMD/博通/台积电/通富微电/深南电路 等）
3. 确定 2026E 预测收入（基于最新收入 + 行业增速 + 公司产能）
4. 计算估值 = 2026E 收入 × 行业PS
5. 识别核心风险（流片失败/英伟达降价/订单能见度/技术路线替代）

## 风险矩阵（必须覆盖）
| 风险类别 | 审查要点 |
|:---|:---|
| 技术卡脖子风险 | 晶圆代工依赖/先进制程流片受阻/设备禁运 |
| 巨头降维打击 | 英伟达/AMD 降价挤压国产芯片价格空间 |
| 订单能见度 | 大客户意向锁定率/长协覆盖比例/政府订单持续性 |
| 技术路线替代 | 是否有相邻技术路径可绕过公司核心壁垒 |
| 产能爬坡 | 良率爬坡曲线/产能利用率/折旧拐点时间 |

请输出严格 JSON:
{{
  "framework": "动态行业PS（分支E·战略成长豁免）",
  "tech_track": "AI训练芯片/先进封装载板/IC载板/光刻胶/碳化硅/...",
  "revenue_2024": 数字(亿元),
  "revenue_2025": 数字(亿元),
  "revenue_2026e": 数字(亿元),
  "yoy_growth": 数字(0-1之间),
  "ps_multiple_assigned": 数字(35或40),
  "ps_multiple_reasoning": "倍数选择依据，对标哪些公司",
  "comparable_companies_ps": [
    {{"name": "对标公司", "ps": 数字, "market": "A股/美股/台股"}}
  ],
  "implied_valuation": 数字(亿元),
  "implied_price": 数字(元/股),
  "bull_case_price": 数字(元/股),
  "bear_case_price": 数字(元/股),
  "key_tech_risks": ["深水区风险1", "风险2", "风险3"],
  "moat_assessment": "护城河评估（技术独占期/认证壁垒/客户粘性）",
  "recommendation": "不超过150字的买方立场总结（必须提及烧钱期容忍度和风控措施）"
}}

---
公司数据:
{company_data}

定性标签:
{labels}
"""

# ——— Branch E: Pharma/Biotech PS Valuation (v3.3) ———
BRANCH_E_PHARMA_VALUATION_PROMPT = """你是一名买方医药股估值分析师。该标的已触发 **v3.3 创新药/Biotech 豁免（分支E·医药）**。

## ⛔ 禁用手法
**严禁使用任何基于 PE/PEG 的估值算法。** 创新药/Biotech 在管线驱动期无利润/负利润是正常状态，PE 估值会导致数学暴走（分母为负或微利导致倍数爆炸）。

## 核心定价锚：动态 PS + 临床管线溢价

### 第一步：PS 基线定价
- **标准 Biotech**: 给予 **15x - 20x** 市销率（PS）
- 收入基准：取最近完整财年营收，或 2026E 一致预期营收
- 对标参考：美股 Biotech（如 Moderna/BioNTech 管线期 12-18x PS）、港股 18A 未盈利生物科技公司（8-15x PS）、A 股创新药（恒瑞/信达/荣昌 15-25x PS）

### 第二步：管线稀缺性叙事溢价（+30% 上限）
若满足以下任一条件，在 PS 基线基础上额外加成：
- **全球化出海**: 核心产品（如泽布替尼/PD-1/ADC）获 FDA 批准或欧盟 CE 认证，海外收入占比 >20% 或有明确海外商业化路径
- **FIC/BIC 品种**: 核心管线为 First-in-Class 或 Best-in-Class，且已进入临床 Ⅲ 期或 NDA 阶段
- **大适应症独占**: 核心品种覆盖 NSCLC/BC/HCC 等大癌种，且竞争格局 ≤3 家（含原研）
- **国际 BD 兑现**: 与跨国药企（MNC）达成 license-out 交易，首付款 >1 亿美元

加成规则：每满足一项 +8%（累计上限 +30%），由 CIO 判断适用的加成项。

### 第三步：管线风险矩阵（必须覆盖）
| 风险类别 | 审查要点 | 对 PS 的影响 |
|:---|:---|:---|
| 临床Ⅲ期失败风险 | 核心管线未来 3 年内是否有 readout？历史成功率？ | 若关键临床 readout 在 12 个月内且不确定性高，PS 下调 20% |
| FDA/监管拒绝风险 | CRL（完整回复函）风险、CMC 合规性、临床数据充分性 | 若 PDUFA 在 6 个月内但有 CRL 先例，PS 下调 15% |
| 医保谈判降价风险 | 核心品种是否进入 NRDL？降价幅度预期？ | 若核心品种即将面临医保续约谈判，PS 下调 10% |
| 海外专利诉讼风险 | 是否存在 Paragraph IV 挑战？核心专利到期时间？ | 若 3 年内有关键专利到期，PS 下调 15% |
| 地缘政治内卷风险 | 出海品种是否面临美国生物安全法案（BIOSECURE）制裁风险？ | 若涉及地缘敏感市场，PS 下调 10-20% |

### 第四步：管线价值加总
对于已进入临床 Ⅱ/Ⅲ 期的核心管线（非上市品种），使用 rNPV 方法单独估值后加总到 PS 估值之上：
- PoS（成功概率）：临床 Ⅲ 期 60-70%，临床 Ⅱ 期 30-40%
- 峰值销售预测 × PoS × 折现因子（r=12%）
- 仅加总市值排名前 3 的管线品种

请输出严格 JSON:
{{
  "framework": "动态PS+临床管线溢价（分支E·医药·v3.3）",
  "pharma_track": "创新药/Biotech/生物制药",
  "revenue_2024": 数字(亿元),
  "revenue_2025": 数字(亿元),
  "revenue_2026e": 数字(亿元),
  "yoy_growth": 数字(0-1之间),
  "ps_baseline": 数字(15-20),
  "pipeline_premium_pct": 数字(0-0.30),
  "pipeline_premium_reasons": ["溢价理由1", "溢价理由2"],
  "ps_final": 数字,
  "comparable_companies": [
    {{"name": "对标公司", "ps": 数字, "market": "美股/港股/A股", "note": "可比理由"}}
  ],
  "pipeline_valuation_addon": 数字(亿元, 管线 rNPV 加总),
  "implied_valuation": 数字(亿元),
  "implied_price": 数字(元/股),
  "bull_case_price": 数字(元/股),
  "bear_case_price": 数字(元/股),
  "key_pharma_risks": [
    {{"risk": "深水区风险", "ps_impact": "下调X%", "probability": "高/中/低"}}
  ],
  "moat_assessment": "护城河评估（核心品种独占性/专利悬崖时间/管线梯队深度）",
  "recommendation": "不超过150字的买方立场总结（必须提及管线风险容忍度和仓位控制）"
}}

---
公司数据:
{company_data}

定性标签:
{labels}
"""

# ——— Branch D: Distressed / Liquidation ———
BRANCH_D_VALUATION_PROMPT = """你正在评估一个触发熔断机制的标的（未通过现金流健康度熔断）。
根据 INVEST_SOP 分支 D，估值降维至清算价值法/资产打折法。

不听故事，只看硬资产安全垫。

请输出严格 JSON:
{{
  "framework": "清算价值/资产打折",
  "net_asset_value": 数字(亿元),
  "liquidation_discount": 数字(0-1之间),
  "distressed_value": 数字(亿元),
  "distressed_price": 数字(元/股),
  "catalyst": "可能的催化剂（资产注入/实控人变更/战投入局）",
  "max_position_pct": 2,
  "hard_stop_loss": "-15%"
}}

---

公司数据:
{company_data}
"""
