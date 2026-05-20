"""
Strategic Exemption Context Injector — v3.3
=============================================

Reads config/exemption_list.yaml at runtime.  If the current stock_code
matches any entry in the registry, builds a system-level strategic
security notification block that gets prepended to LLM prompts.

This ensures the LLM is informed that the asset under analysis is a
strategic national-security-grade asset with offensive-style bleeding
(R&D-fueled negative cash flow), and that traditional manufacturing
liquidation logic must NOT be blindly applied.
"""

import os
import yaml


_STRATEGIC_CONTEXT_TEMPLATE = """
┌─────────────────────────────────────────────────────────────────────────────┐
│             ⚠ 【系统级战略安全豁免通知 — INVEST SOP v3.3】                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  当前标的已被买方最高战略资产库收录，具体豁免信息如下：                        │
│                                                                             │
│    标的名称: {name}                                                          │
│    标的代码: {code}                                                          │
│    豁免分类: {category}                                                      │
│    入库日期: {added_date}                                                    │
│    复审周期: {audit_cycle_months} 个月                                       │
│                                                                             │
│  【核心豁免依据】                                                            │
│  {reason}                                                                   │
│                                                                             │
│  【强制约束指令 — 请严格遵循以下分析纪律】                                    │
│                                                                             │
│  1. 当前标的属于买方最高战略资产，其高强度研发导致的现金流或利润失血           │
│     属于【进攻型失血】，而非传统制造业的「有毒增长」。                         │
│                                                                             │
│  2. 在进行后续估值、红队对抗或 CIO 冲突调和时，严禁使用传统的普通制造业        │
│     清算逻辑（如净现比熔断、PE机械打折、DCF清算估值）对其盲目唱空。           │
│                                                                             │
│  3. 必须还原其真实自由现金流和平台化吞噬的潜力。估值锚应聚焦于：               │
│     a) 技术独占期的收入规模与行业 PS 倍数                                    │
│     b) 订单能见度（长协/意向锁定率）而非当期会计利润                          │
│     c) 认证壁垒/客户粘性构成的准护城河价值                                    │
│     d) 平台化扩张的潜在协同与吞噬效应                                         │
│                                                                             │
│  4. 红队对抗时，禁止使用「无利润/负现金流/高PS/低毛利率/亏损扩大」            │
│     作为攻击点——这些都是高壁垒硬科技烧钱期的正常特征。                        │
│     必须从深水区维度进行攻击：技术卡脖子、巨头竞争、订单能见度、               │
│     技术路线替代、产能爬坡良率。                                              │
│                                                                             │
│  5. CIO 冲突调和时，叙事溢价率应充分考虑该资产的战略稀缺性和国产替代的          │
│     不可逆趋势，不得机械地以当期财务指标压缩估值。                             │
│                                                                             │
│  ⚠ 如果你违反上述约束，你的分析结论将被系统标记为「方法论误用」并驳回。        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
"""


def _load_registry() -> list[dict]:
    """Load exemption registry from YAML. Returns list of entry dicts."""
    yaml_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "config", "exemption_list.yaml"
    )
    yaml_path = os.path.normpath(yaml_path)

    if not os.path.exists(yaml_path):
        return []

    try:
        with open(yaml_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data.get("STRATEGIC_GROWTH_COMPANIES", [])
    except Exception:
        return []


def get_strategic_context(stock_code: str) -> str:
    """
    If stock_code matches an entry in the exemption registry, return
    a system-level strategic context block. Otherwise return empty string.
    """
    if not stock_code:
        return ""

    registry = _load_registry()
    for entry in registry:
        entry_code = str(entry.get("code", "")).strip()
        if entry_code == stock_code:
            return _STRATEGIC_CONTEXT_TEMPLATE.format(
                name=entry.get("name", "未知"),
                code=entry_code,
                category=entry.get("category", "战略资产"),
                added_date=entry.get("added_date", "未知"),
                audit_cycle_months=entry.get("audit_cycle_months", 12),
                reason=entry.get("reason", "").strip(),
            ).strip()

    return ""


def is_strategic_asset(stock_code: str) -> bool:
    """Quick check: is this stock in the exemption registry?"""
    if not stock_code:
        return False
    registry = _load_registry()
    for entry in registry:
        if str(entry.get("code", "")).strip() == stock_code:
            return True
    return False
