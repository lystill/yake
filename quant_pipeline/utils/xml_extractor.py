"""
Universal XML-tag JSON Extractor — v3.3 防断流加固
====================================================

Provides a single authoritative extraction+repair function that:
  1. Extracts JSON from <json_response>...</json_response> XML tags.
  2. Falls back to raw brace extraction if tags are missing.
  3. Applies 5-level repair pipeline: strip fences → collapse multiline
     strings → truncate trailing garbage → brute-force brace/suffix
     completion → last-line amputation re-close.

Usage:
    from utils.xml_extractor import extract_json
    result = extract_json(llm_raw_text)
"""

import re
import json


# ═══════════════════════════════════════════════════════════════
#  XML Tag Extraction — 严格提取 <json_response> 标签内文本
# ═══════════════════════════════════════════════════════════════

_XML_PATTERN = re.compile(
    r"<json_response>(.*?)</json_response>",
    re.DOTALL | re.IGNORECASE,
)


def extract_json(raw_text: str) -> dict:
    """
    Master entry point.  Extracts JSON from LLM output with full repair pipeline.

    Strategy (in order):
      1. Try <json_response> XML tag extraction.
      2. Fall back to raw { ... } extraction.
      3. 5-level repair before giving up.
    """
    if not raw_text or not raw_text.strip():
        return _fallback("空响应")

    text = raw_text.strip()

    # ── Phase 1: XML tag extraction ──
    xml_match = _XML_PATTERN.search(text)
    if xml_match:
        inner = xml_match.group(1).strip()
        result = _parse_with_repair(inner)
        if result is not None:
            return result
        # XML extraction failed — fall through to raw extraction

    # ── Phase 2: Raw brace extraction ──
    result = _parse_with_repair(text)
    if result is not None:
        return result

    # ── Ultimate fallback ──
    print(f"  ⚠ [XMLExtractor] 全部提取+修复策略失败，使用兜底")
    return _fallback(f"JSON解析失败，原始片段: {text[:200]}")


def _parse_with_repair(text: str) -> dict | None:
    """5-level repair pipeline. Returns parsed dict or None."""

    # Level 0: Strip markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    # Extract outermost { ... }
    match = re.search(r'\{.*\}', cleaned, re.DOTALL)
    json_str = match.group(0) if match else cleaned
    json_str = json_str.replace('\r', '')

    # Attempt 0: Raw parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 1: Collapse multiline strings
    repaired = _repair_multiline_strings(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Truncate trailing garbage
    repaired = _repair_trailing_garbage(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 3: Brute-force brace/bracket/quote completion
    repaired = _repair_brace_completion(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    # Attempt 4: Last-line amputation — cut the last incomplete kv pair
    repaired = _repair_last_amputation(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        pass

    return None


# ═══════════════════════════════════════════════════════════════
#  Repair functions
# ═══════════════════════════════════════════════════════════════

def _repair_multiline_strings(json_str: str) -> str:
    """Collapse literal newlines inside JSON string values."""
    result = []
    in_string = False
    escape_next = False
    for ch in json_str:
        if escape_next:
            result.append(ch)
            escape_next = False
            continue
        if ch == '\\':
            result.append(ch)
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            result.append(ch)
            continue
        if in_string and ch == '\n':
            result.append('\\n')
            continue
        result.append(ch)
    return ''.join(result)


def _repair_trailing_garbage(json_str: str) -> str:
    """Find the last structurally valid closing brace and truncate."""
    depth = 0
    last_open = len(json_str)
    for i in range(len(json_str) - 1, -1, -1):
        ch = json_str[i]
        if ch == '}':
            depth += 1
        elif ch == '{':
            depth -= 1
            if depth == 0:
                last_open = i
    candidate = json_str[:last_open] + '}'
    return candidate


def _repair_brace_completion(json_str: str) -> str:
    """Brute-force: pad missing closing braces/brackets/quotes."""
    s = json_str.strip()

    # Remove trailing comma before closing
    s = re.sub(r',\s*$', '', s)

    open_braces = s.count('{') - s.count('}')
    open_brackets = s.count('[') - s.count(']')

    if open_brackets > 0:
        s += ']' * open_brackets
    if open_braces > 0:
        s += '}' * open_braces

    # Fix unclosed string values
    if s.count('"') % 2 != 0:
        last_quote = s.rfind('"')
        last_colon = s.rfind(':')
        if last_colon > last_quote:
            s += '"'

    return s


def _repair_last_amputation(json_str: str) -> str:
    """
    v3.3 新增: 大模型断流专用修复。

    When the LLM output is truncated mid-stream, the last line is
    often a partial key-value pair with missing closing quote or brace.
    This function:
      1. Finds the last structurally complete line (ending with " or }, or ], or a digit)
      2. Amputates everything after it
      3. Forces a closing } at the end
    """
    lines = json_str.split('\n')
    if len(lines) <= 2:
        return json_str  # too short to amputate

    # Walk backwards to find the last line that ends with a "complete" token
    cut_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].rstrip().rstrip(',').rstrip()
        if not stripped:
            continue
        # Complete line indicators:
        #   ends with " (string value closed)
        #   ends with } or ] (structure closed)
        #   ends with a digit (numeric value closed)
        #   ends with true/false/null
        if (stripped.endswith('"') or
            stripped.endswith('}') or
            stripped.endswith(']') or
            stripped[-1].isdigit() or
            stripped.endswith('true') or
            stripped.endswith('false') or
            stripped.endswith('null')):
            cut_idx = i + 1
            break

    if cut_idx < len(lines):
        truncated = '\n'.join(lines[:cut_idx])
        # Remove trailing comma
        truncated = re.sub(r',\s*$', '', truncated)
        # Ensure closing brace
        open_braces = truncated.count('{') - truncated.count('}')
        truncated += '}' * max(open_braces, 0)
        return truncated

    return json_str


def _fallback(reason: str) -> dict:
    """Deterministic fallback when all strategies fail."""
    return {
        "bearish_signals": [f"JSON解析全部失败: {reason}"],
        "severity": "high",
        "reasoning": "系统全面解析崩溃，强制转入人工审计",
        "most_dangerous_signal": "无法自动分析——模型输出完全不可解析",
        "suggested_adjustments": {
            "peg_haircut_recommendation": "PEG建议强制打7折以计入不确定性",
            "valuation_floor_risk": "极端悲观估值下限: 净资产打8折",
        },
    }
