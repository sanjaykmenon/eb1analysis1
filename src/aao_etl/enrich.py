from __future__ import annotations
import re
from typing import Dict, Any, Optional

_CFR_PAT = re.compile(r'(?:8\s*C\.?\s*F\.?\s*R\.?\s*§\s*[\d\.]+(?:\([a-z0-9]+\))*)', re.IGNORECASE)
_PM_PAT  = re.compile(r'USCIS Policy Manual[^,\n]*', re.IGNORECASE)
_KAZARIAN_PAT = re.compile(r'Kazarian v\.?\s*USCIS[^,\n]*', re.IGNORECASE)
_DATE_LINE_PAT = re.compile(r'\bDate:\s*([A-Z]{3}\.?\s*\d{1,2},\s*\d{4})', re.IGNORECASE)
_IN_RE_PAT = re.compile(r'In\s+Re:\s*([^\n]+)')

def regex_enrichments(text: str) -> Dict[str, Any]:
    cfrs = sorted(set(_CFR_PAT.findall(text)))
    policy = sorted(set(_PM_PAT.findall(text)))
    kaz = sorted(set(_KAZARIAN_PAT.findall(text)))
    decision_date = (_DATE_LINE_PAT.search(text).group(1) if _DATE_LINE_PAT.search(text) else None)
    case_number = (_IN_RE_PAT.search(text).group(1).strip() if _IN_RE_PAT.search(text) else None)

    outcome = None
    if re.search(r'appeal\s+will\s+be\s+dismissed|appeal\s+is\s+dismissed', text, re.IGNORECASE):
        outcome = "dismissed"
    elif re.search(r'appeal\s+(?:will\s+be|is)\s+sustained', text, re.IGNORECASE):
        outcome = "sustained"
    elif re.search(r'appeal\s+(?:will\s+be|is)\s+rejected', text, re.IGNORECASE):
        outcome = "rejected"

    return {
        "cfr_citations": cfrs,
        "policy_manual_mentions": policy,
        "authorities": kaz,
        "decision_date_text": decision_date,
        "case_number": case_number,
        "outcome_guess": outcome,
    }

def parse_decision_date(s: Optional[str]):
    if not s:
        return None
    s2 = s.replace(".", "")
    from datetime import datetime
    try:
        return datetime.strptime(s2, "%b %d, %Y").date()
    except Exception:
        return None