# src/aao_etl/models.py
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import date

# ---------- ENUMS ----------
class Finding(str, Enum):
    MET = "met"
    NOT_MET = "not_met"
    NOT_ANALYZED = "not_analyzed"  # e.g., "assume arguendo", "need not reach"
    UNKNOWN = "unknown"            # ambiguous in the record

class Criterion(str, Enum):
    AWARD = "AWARD"                       # 8 CFR 204.5(h)(3)(i)
    MEMBERSHIP = "MEMBERSHIP"             # (ii)
    PUBLISHED_MATERIAL = "PUBLISHED_MATERIAL"  # (iii) about beneficiary
    JUDGE = "JUDGE"                       # (iv)
    ORIGINAL_CONTRIBUTION = "ORIGINAL_CONTRIBUTION"  # (v)
    AUTHOR = "AUTHOR"                     # (vi) scholarly articles
    DISPLAY = "DISPLAY"                   # (vii) (arts)
    LEADING_ROLE = "LEADING_ROLE"         # (viii)
    HIGH_SALARY = "HIGH_SALARY"           # (ix)
    COMMERCIAL_SUCCESS = "COMMERCIAL_SUCCESS"  # (x) (performing arts)

class EvidenceType(str, Enum):
    PUBLICATION = "PUBLICATION"
    LETTER = "LETTER"
    JUDGING = "JUDGING"
    AWARD = "AWARD"
    MEMBERSHIP = "MEMBERSHIP"
    PRESS = "PRESS"
    SOFTWARE = "SOFTWARE"
    PATENT = "PATENT"
    ADOPTION = "ADOPTION"        # clinical/vendor deployment, benchmark baseline, etc.
    SALARY = "SALARY"
    ROLE = "ROLE"
    OTHER = "OTHER"

class ExclusionReason(str, Enum):
    POST_FILING = "post_filing"
    LACK_METHOD = "lack_of_methodology"
    NOT_INDEPENDENT = "not_independent"
    IRRELEVANT_SUBFIELD = "irrelevant_subfield"
    UNDATED = "undated"
    UNVERIFIABLE = "unverifiable"
    INSUFFICIENT_CORROBORATION = "insufficient_corroboration"
    HEARSAY_OR_LOW_WEIGHT = "hearsay_or_low_weight"

class FinalMeritsResult(str, Enum):
    FAVORABLE = "favorable"
    UNFAVORABLE = "unfavorable"
    NOT_REACHED = "not_reached"

# ---------- PRIMITIVES ----------
class Quote(BaseModel):
    text: str = Field(..., description="Direct quote from the decision.")
    page: Optional[int] = Field(None, ge=1, description="1-indexed page if known.")
    start_char: Optional[int] = Field(None, ge=0)
    end_char: Optional[int] = Field(None, ge=0)

class CriterionAnalysis(BaseModel):
    criterion: Criterion
    director_finding: Finding
    aao_finding: Finding
    rationale: str = Field(..., description="Neutral summary of why the finding was reached.")
    quotes: List[Quote] = Field(default_factory=list)
    issue_tags: List[str] = Field(default_factory=list, description="Taxonomy tags indicating why evidence was discounted (e.g., 'post_filing_evidence','letters_conclusory','no_single_paper_comparison').")

    @model_validator(mode="after")
    def _require_quotes_when_known(self):
        definite = {Finding.MET, Finding.NOT_MET, Finding.NOT_ANALYZED}
        if (self.director_finding in definite or self.aao_finding in definite) and not self.quotes:
            raise ValueError("Provide at least one quote for any non-UNKNOWN finding.")
        return self

class EvidenceItem(BaseModel):
    e_type: EvidenceType
    title: Optional[str] = None
    description: str
    event_date: Optional[date] = None
    is_pre_filing: Optional[bool] = None      # compute post-hoc if you have filing_date
    accepted_by_uscis: Optional[bool] = None
    exclusion_reasons: List[ExclusionReason] = Field(default_factory=list)
    quotes: List[Quote] = Field(default_factory=list)

class Authority(BaseModel):
    name: str
    type: str  # "AAO precedent" | "Circuit case" | "Policy Manual" | "Regulation"
    citation: Optional[str] = None
    quotes: List[Quote] = Field(default_factory=list)

class DecisionExtraction(BaseModel):
    case_number: Optional[str] = None
    petition_type: Optional[str] = None              # e.g., "EB1A"
    field_of_endeavor: Optional[str] = None
    decision_date: Optional[date] = None
    filing_date: Optional[date] = None
    outcome: Optional[str] = None                    # 'denied' | 'dismissed' | 'sustained' | ...
    criteria: List[CriterionAnalysis] = Field(default_factory=list)
    final_merits: FinalMeritsResult
    final_merits_rationale: Optional[str] = None
    final_merits_quotes: List[Quote] = Field(default_factory=list)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    authorities: List[Authority] = Field(default_factory=list)
    global_issue_tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _final_merits_quote_rule(self):
        if self.final_merits in {FinalMeritsResult.FAVORABLE, FinalMeritsResult.UNFAVORABLE} and not self.final_merits_quotes:
            raise ValueError("Final merits reached but no quotes supplied.")
        return self

# (Optional) keep your simpler doc model if you still use it elsewhere
class DocumentInfo(BaseModel):
    title: Optional[str] = None
    beneficiary_details: List[str] = Field(default_factory=list)
    beneficiary_status: Optional[str] = None
    key_reasons: List[str] = Field(default_factory=list)
    summary: Optional[str] = None
    date_of_application: Optional[date] = None
    summary_embedding: Optional[List[float]] = None
    footnotes: List[str] = Field(default_factory=list)
    cfr_code: List[str] = Field(default_factory=list)
    full_text: Optional[str] = None

    @field_validator("key_reasons")
    @classmethod
    def _strip_empty(cls, v: List[str]) -> List[str]:
        return [s.strip() for s in v if s and s.strip()]
