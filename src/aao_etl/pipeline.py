from __future__ import annotations
import os, json
from typing import Dict, Any, Optional
from .pdf import extract_pdf_text
from .enrich import regex_enrichments, parse_decision_date
from .db import upsert_decision
from .config import settings
from .models import DecisionExtraction, DocumentInfo
from . import llm

def process_single_pdf(*, pdf_path: str, engine, dry_run: bool) -> Dict[str, Any]:
    try:
        print(f"\n📄 {os.path.basename(pdf_path)}")
        text = extract_pdf_text(pdf_path)
        quick = regex_enrichments(text)
        print(f"  case_number={quick.get('case_number')}, date={quick.get('decision_date_text')}, outcome≈{quick.get('outcome_guess')}")
        if quick["cfr_citations"]:
            print(f"  CFR cites: {', '.join(quick['cfr_citations'][:6])}{' …' if len(quick['cfr_citations'])>6 else ''}")

        if dry_run:
            # Dry run mode - minimal extraction without LLM
            print("  🔍 Dry run mode - regex enrichments only")
            doc = create_minimal_extraction(text, quick)
        else:
            # Full LLM-powered structured extraction (REQUIRED)
            if not settings.openai_api_key:
                raise RuntimeError(
                    f"OpenAI API key required for processing {pdf_path}. "
                    "Set OPENAI_API_KEY environment variable or use --dry-run flag."
                )
            
            print("  🧠 Running structured extraction with LLM...")
            filing_date_str = None
            if quick.get("decision_date_text"):
                # Try to infer filing date from decision date (rough heuristic)
                decision_date = parse_decision_date(quick.get("decision_date_text"))
                if decision_date:
                    filing_date_str = str(decision_date.replace(year=decision_date.year - 1))
            
            # This will raise an exception if LLM extraction fails
            doc = llm.structured_extraction_monolithic(text, filing_date=filing_date_str)
            print(f"  ✅ Extracted {len(doc.criteria)} criteria, {len(doc.evidence)} evidence items")

        decision_id = upsert_decision(engine, pdf_path=pdf_path, quick=quick, doc=doc)
        return {
            "pdf": os.path.basename(pdf_path), 
            "decision_id": decision_id, 
            "case_number": doc.case_number or quick.get("case_number"),
            "outcome": doc.outcome,
            "criteria_count": len(doc.criteria) if hasattr(doc, 'criteria') else 0,
            "evidence_count": len(doc.evidence) if hasattr(doc, 'evidence') else 0,
            "status": "success"
        }
    
    except Exception as e:
        error_msg = str(e)
        print(f"  ❌ FAILED: {error_msg}")
        return {
            "pdf": os.path.basename(pdf_path),
            "status": "failed",
            "error": error_msg,
            "decision_id": None,
            "case_number": None,
            "outcome": None,
            "criteria_count": 0,
            "evidence_count": 0,
        }

def create_minimal_extraction(text: str, quick: Dict[str, Any]) -> DecisionExtraction:
    """Create a minimal DecisionExtraction without LLM calls."""
    from .models import FinalMeritsResult
    
    decision_date = parse_decision_date(quick.get("decision_date_text"))
    outcome_map = {"dismissed": "dismissed", "sustained": "sustained", "rejected": "denied"}
    outcome = outcome_map.get(quick.get("outcome_guess") or "", "unknown")
    
    return DecisionExtraction(
        case_number=quick.get("case_number"),
        petition_type="EB1A",  # assume EB1A for now
        decision_date=decision_date,
        outcome=outcome,
        final_merits=FinalMeritsResult.NOT_REACHED,  # conservative default
        criteria=[],  # empty for dry run
        evidence=[],  # empty for dry run  
        authorities=[],  # could populate from quick["authorities"]
        global_issue_tags=[],
    )
