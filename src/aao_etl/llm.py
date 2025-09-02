# src/aao_etl/llm.py
from __future__ import annotations
from typing import List, Optional, Dict, Any
from .models import DecisionExtraction, Criterion
from .config import settings

from openai import OpenAI
import instructor

_openai_client = None
_instructor_client = None

def get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client

def get_instructor_client() -> OpenAI:
    global _instructor_client
    if _instructor_client is None:
        _instructor_client = instructor.patch(get_openai_client())
    return _instructor_client

def truncate_for_prompt(text: str, max_chars: int = 12000) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."

# ---------- ENHANCED CRITERION DEFINITIONS ----------

CRITERION_DEFS = {
    "AWARD": "8 CFR 204.5(h)(3)(i): Lesser nationally or internationally recognized prizes or awards for excellence in the field of endeavor. Must be recognized prizes/awards (not nominations), show national/international scope, and demonstrate excellence in the specific field.",
    "MEMBERSHIP": "8 CFR 204.5(h)(3)(ii): Membership in associations in the field for which classification is sought, which require outstanding achievements of their members, as judged by recognized national or international experts. Must show: (1) membership requires outstanding achievements, (2) judged by recognized experts, (3) association is in petitioner's field.",
    "PUBLISHED_MATERIAL": "8 CFR 204.5(h)(3)(iii): Published material about the alien in professional or major trade publications or other major media, relating to the alien's work in the field. Must be ABOUT the beneficiary (not BY), in professional/major publications, relating to their work.",
    "JUDGE": "8 CFR 204.5(h)(3)(iv): Evidence of participation, either individually or on a panel, as a judge of the work of others in the same or an allied field. Includes peer review, editorial board service, conference program committees, grant review panels.",
    "ORIGINAL_CONTRIBUTION": "8 CFR 204.5(h)(3)(v): Evidence of original scientific, scholarly, artistic, athletic, or business-related contributions of major significance to the field. Must show: (1) original work, (2) major significance to the field (not just to a subfield), (3) contributions have been recognized/adopted.",
    "AUTHOR": "8 CFR 204.5(h)(3)(vi): Evidence of authorship of scholarly articles in the field, in professional or major trade publications or other major media. Must be scholarly articles (not just any publication), in professional/major publications.",
    "DISPLAY": "8 CFR 204.5(h)(3)(vii): Evidence of the display of work at artistic exhibitions or showcases. Applies primarily to artists - exhibitions must be artistic in nature.",
    "LEADING_ROLE": "8 CFR 204.5(h)(3)(viii): Evidence of performance in a leading or critical role for organizations or establishments that have a distinguished reputation. Must show: (1) leading/critical role, (2) organization has distinguished reputation.",
    "HIGH_SALARY": "8 CFR 204.5(h)(3)(ix): Evidence of a high salary or other significantly high remuneration in relation to others in the field. Must compare salary to others in the same field using reliable market data/surveys.",
    "COMMERCIAL_SUCCESS": "8 CFR 204.5(h)(3)(x): Evidence of commercial successes in the performing arts, as shown by box office receipts or record, cassette, compact disk, or video sales. Applies only to performing arts with commercial metrics.",
}

# Enhanced legal language patterns specific to AAO decisions
DIRECTOR_FINDING_PATTERNS = {
    "MET": [
        "the director found", "director concluded", "director determined", "director accepted",
        "initially approved", "service center approved", "director granted"
    ],
    "NOT_MET": [
        "director denied", "director found insufficient", "director concluded the petitioner failed",
        "director determined the evidence does not", "service center denied"
    ]
}

AAO_FINDING_PATTERNS = {
    "MET": [
        "we find", "we conclude", "we determine", "the AAO finds", "established",
        "petitioner has demonstrated", "evidence establishes", "record supports"
    ],
    "NOT_MET": [
        "we are not persuaded", "has not demonstrated", "insufficient evidence",
        "does not establish", "the record does not support", "petitioner failed to",
        "evidence does not rise to", "not persuasive", "lacks probative value"
    ],
    "NOT_ANALYZED": [
        "we need not reach", "need not analyze", "assume arguendo", "even if",
        "without reaching", "not necessary to determine"
    ]
}

POST_FILING_INDICATORS = [
    "post-filing", "after the filing date", "subsequent to filing",
    "dated after", "undated", "no date provided"
]

EVIDENCE_EXCLUSION_CUES = {
    "conclusory": ["conclusory", "lacking detail", "vague statements", "unsupported assertions"],
    "not_independent": ["not independent", "prepared by", "self-serving", "internal"],
    "methodology": ["no methodology", "unclear methodology", "unreproducible", "insufficient detail"],
    "subfield": ["subfield", "specialized area", "narrow focus", "limited scope"]
}

# ---------- HIGHLY SPECIFIC SYSTEM PROMPT ----------

DECISION_SYSTEM_PROMPT = """\
You are analyzing a U.S. Administrative Appeals Office (AAO) decision for an EB-1A extraordinary ability petition. Your task is to extract structured data with surgical precision from this immigration law document.

CRITICAL EXTRACTION RULES:

1. CASE IDENTIFICATION:
   - Extract case number from "In Re:" line (format typically starts with letters like EAC, MSC, WAC, SRC, etc.)
   - Decision date from "Date:" line (format: Month DD, YYYY)
   - Outcome: "dismissed", "sustained", "remanded", or "denied" based on final disposition language

2. CRITERION ANALYSIS (8 CFR 204.5(h)(3)):
   - For EACH criterion discussed, capture BOTH director and AAO findings separately
   - Director findings: Look for "director found/concluded/determined" language in background/procedural history
   - AAO findings: Look for "we find/conclude/determine" or "the AAO finds" in analysis sections
   - Use exact enum values: "met", "not_met", "not_analyzed", "unknown"
   - "not_analyzed" for "assume arguendo", "need not reach", "without determining" language

3. QUOTE REQUIREMENTS:
   - Every non-"unknown" finding MUST include direct quotes from the decision
   - Quote the exact AAO/Director language that supports the finding
   - Include page numbers if visible (format like "Page 3" or numbered headers)
   - For "not_met" findings, quote the specific rejection language

4. EVIDENCE EXTRACTION:
   - Identify specific evidence items: letters, publications, awards, judging roles, salary data
   - Determine if evidence was accepted or rejected by USCIS
   - Look for post-filing evidence disclaimers and date exclusions
   - Tag exclusion reasons: post_filing, lack_of_methodology, not_independent, etc.

5. FINAL MERITS (Kazarian Step Two):
   - "not_reached" if AAO states "we need not reach final merits" or similar
   - "unfavorable" if AAO reaches merits and denies (even if 3+ criteria met)
   - "favorable" only if AAO reaches merits and approves/sustains
   - Quote the specific final merits analysis language

6. AUTHORITIES:
   - Capture case citations: "Kazarian v. USCIS", circuit court cases
   - USCIS Policy Manual references with volume/chapter
   - AAO precedent decisions
   - Include short quotes showing how authority was applied

7. LINGUISTIC PRECISION:
   - "We withdraw the director's finding" = Director: met, AAO: not_met
   - "Has not demonstrated" = not_met
   - "Insufficient evidence" = not_met
   - "We are not persuaded" = not_met
   - "Even if arguendo" = not_analyzed for that criterion

8. FIELD OF ENDEAVOR:
   - Extract from decision text, often in background section
   - Examples: "computer science", "biomedical engineering", "artificial intelligence"

DO NOT:
- Speculate beyond what is explicitly stated
- Combine findings across different criteria
- Assume dates not provided in the decision
- Create quotes that paraphrase rather than directly quote
- Infer evidence acceptance without explicit AAO/Director statements

OUTPUT FORMAT: Return only valid JSON conforming to the DecisionExtraction schema. No commentary or explanation outside the JSON structure.
"""

# ---------- DETAILED USER PROMPT BUILDER ----------

def build_decision_user_prompt(
    full_text: str,
    filing_date: Optional[str] = None,
    petition_type_hint: Optional[str] = "EB1A",
    extra_notes: Optional[str] = None,
) -> str:
    """Builds a highly specific user prompt for AAO decision extraction."""
    
    prompt_sections = []
    
    # Task specification
    prompt_sections.append("""EXTRACTION TASK: Analyze this AAO EB-1A decision and extract all required data points according to the DecisionExtraction schema.

DOCUMENT TYPE: Administrative Appeals Office decision on EB-1A extraordinary ability petition appeal.""")
    
    if filing_date:
        prompt_sections.append(f"""FILING DATE: {filing_date}
POST-FILING ANALYSIS: Any evidence dated after {filing_date} should be marked as post-filing and typically excluded from consideration.""")
    
    # Detailed criterion guide
    prompt_sections.append("""
CRITERION ANALYSIS REQUIREMENTS:
Extract findings for ANY of these 10 criteria that are discussed in the decision:""")
    
    for criterion, definition in CRITERION_DEFS.items():
        prompt_sections.append(f"- {criterion}: {definition}")
    
    # Evidence extraction guide
    prompt_sections.append("""
EVIDENCE IDENTIFICATION GUIDE:
Look for these types of evidence and determine acceptance/rejection:
- PUBLICATION: Peer-reviewed papers, journal articles, conference proceedings
- LETTER: Reference letters, testimonials, expert opinions
- JUDGING: Peer review activities, editorial boards, conference committees
- AWARD: Prizes, honors, recognitions (not nominations)
- MEMBERSHIP: Professional organization memberships requiring achievements
- PRESS: Media coverage, news articles about beneficiary
- SALARY: Compensation data, employment contracts, salary surveys
- ADOPTION: Citations, implementations, adoptions of beneficiary's work
- ROLE: Employment positions, leadership roles
- OTHER: Any other evidence type not fitting above categories""")
    
    # Quote extraction requirements
    prompt_sections.append("""
QUOTE EXTRACTION CRITICAL REQUIREMENTS:
1. For each criterion finding, include exact AAO language like:
   - "We find that the petitioner has demonstrated..."
   - "The evidence is insufficient to establish..."
   - "We are not persuaded that..."
   - "The director correctly found that..."

2. For evidence items, quote language showing acceptance/rejection:
   - "This evidence establishes..."
   - "The letters are conclusory and lack..."
   - "This evidence post-dates the filing..."

3. For final merits, quote the dispositive language:
   - "We need not reach the question of final merits..."
   - "Even if the petitioner had established three criteria..."
   - "The totality of the evidence establishes..."

4. Include page numbers when visible in headers or footers.""")
    
    # Final merits specific guidance
    prompt_sections.append("""
FINAL MERITS (KAZARIAN STEP TWO) ANALYSIS:
- If AAO states they "need not reach final merits" or similar → final_merits: "not_reached"
- If AAO analyzes whether petitioner rises to top of field → extract that analysis
- Quote the specific language about final merits determination
- Note: Even 3+ met criteria doesn't guarantee favorable final merits""")
    
    # Authority extraction
    prompt_sections.append("""
LEGAL AUTHORITIES TO CAPTURE:
- Case law: "Kazarian v. USCIS", "Visinscaia v. Beers", circuit court decisions
- Regulations: "8 CFR 204.5(h)(3)", specific subsections
- Policy Manual: "USCIS Policy Manual Volume X Chapter Y"
- AAO precedent: Previous AAO decisions cited by name/case number
- Include quote showing how authority was applied to this case""")
    
    # Issue tagging guide
    prompt_sections.append("""
DENIAL ISSUE TAGS (when evidence rejected):
Use these specific tags when AAO/Director excludes evidence:
- "post_filing_evidence" - Evidence dated after filing
- "letters_conclusory" - Reference letters lacking detail/specificity  
- "subfield_mismatch" - Evidence from different/narrow subfield
- "not_independent" - Evidence from biased/interested parties
- "methodology_unclear" - Research lacking clear methodology
- "unverifiable" - Claims that cannot be verified
- "undated" - Evidence without clear dates
- "insufficient_corroboration" - Unsupported claims lacking corroboration""")
    
    # The actual decision text
    prompt_sections.append(f"""
AAO DECISION TEXT:
{truncate_for_prompt(full_text)}""")
    
    # Final extraction mandate
    prompt_sections.append("""
EXTRACTION REQUIREMENTS SUMMARY:
1. Populate case_number, decision_date, outcome, field_of_endeavor from decision headers/text
2. For each discussed criterion: director_finding, aao_finding, rationale, quotes[], issue_tags[]
3. Extract evidence[] items with type, description, acceptance status, exclusion reasons
4. Capture authorities[] with name, type, citation, quotes[]
5. Determine final_merits status and include final_merits_quotes[]
6. Add global_issue_tags[] for decision-level denial patterns

CRITICAL: Every finding must be supported by direct quotes from the decision text. Use exact language, not paraphrases.""")
    
    if extra_notes:
        prompt_sections.append(f"\nADDITIONAL NOTES: {extra_notes}")
    
    return "\n\n".join(prompt_sections)

# ---------- SINGLE-PASS EXTRACTION WITH ENHANCED PROMPT ----------
def structured_extraction_monolithic(text: str, filing_date: Optional[str] = None) -> DecisionExtraction:
    client = get_instructor_client()
    
    # o3 model doesn't support temperature=0, use default
    kwargs = {
        "model": settings.gpt_model,
        "response_model": DecisionExtraction,
        "messages": [
            {"role": "system", "content": DECISION_SYSTEM_PROMPT},
            {"role": "user", "content": build_decision_user_prompt(text, filing_date)},
        ],
        "max_retries": 3,
    }
    
    # Only add temperature if not using o3 model
    if settings.gpt_model != "o3":
        kwargs["temperature"] = 0
    
    return client.chat.completions.create(**kwargs)

# ---------- HELPER FUNCTIONS ----------
def summarize(text: str, beneficiary_details: List[str] = None, key_reasons: List[str] = None) -> str:
    """Generate a summary of the decision."""
    client = get_openai_client()
    
    context = ""
    if beneficiary_details:
        context += f"Beneficiary: {', '.join(beneficiary_details[:3])}\n"
    if key_reasons:
        context += f"Key issues: {', '.join(key_reasons[:3])}\n"
    
    response = client.chat.completions.create(
        model=settings.gpt_model,
        messages=[
            {"role": "system", "content": "Summarize this AAO decision in 2-3 sentences. Focus on the outcome and main reasons."},
            {"role": "user", "content": f"{context}\n\nDecision text:\n{truncate_for_prompt(text, 8000)}"}
        ],
        temperature=0.3,
        max_tokens=200,
    )
    return response.choices[0].message.content

def embed(text: str) -> List[float]:
    """Generate embeddings for text using OpenAI's embedding model."""
    client = get_openai_client()
    
    # Truncate text to fit embedding model limits
    text = text[:8000] if len(text) > 8000 else text
    
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Legacy function for backwards compatibility
def structured_extraction(text: str):
    """Legacy function - creates a basic DocumentInfo for compatibility."""
    from .models import DocumentInfo
    
    # Very basic extraction for backwards compatibility
    return DocumentInfo(
        beneficiary_details=[],
        beneficiary_status="EB1A",
        key_reasons=[],
        summary=None,
        date_of_application=None,
        footnotes=[],
        cfr_code=[],
        full_text=text,
    )



# Controlled vocabulary from  instructions
CONTROLLED_DENIAL_TAGS = {
    "post_filing_evidence": "Evidence dated after petition filing date",
    "letters_conclusory": "Reference letters lacking detail or specificity",
    "subfield_mismatch": "Evidence from different or narrow subfield",
    "not_independent": "Evidence from biased or interested parties",
    "no_single_paper_comparison": "Lack of proper comparison methodology",
    "metrics_not_reproducible": "Research methodology unclear or unreproducible",
    "salary_benchmark_weak": "Salary comparison data inadequate or flawed",
    "role_not_critical": "Leadership role lacks critical importance",
    "methodology_unclear": "Research lacking clear methodology",
    "unverifiable": "Claims that cannot be verified",
    "undated": "Evidence without clear dates",
    "insufficient_corroboration": "Unsupported claims lacking corroboration"
}

def normalize_text_for_quote_matching(text: str) -> str:
    """Normalize text for quote validation (whitespace, punctuation)."""
    import re
    # Normalize whitespace and remove excessive punctuation
    normalized = re.sub(r'\s+', ' ', text.strip())
    normalized = re.sub(r'[""''"]', '"', normalized)  # Normalize quotes
    normalized = re.sub(r'[—–]', '-', normalized)    # Normalize dashes
    return normalized.lower()

def find_quote_location(quote_text: str, full_text: str, page_offsets: List[int] = None) -> Dict[str, Any]:
    """Find quote location with page/character anchoring per instructions."""
    import difflib
    
    normalized_quote = normalize_text_for_quote_matching(quote_text)
    normalized_full = normalize_text_for_quote_matching(full_text)
    
    # Try exact match first
    start_pos = normalized_full.find(normalized_quote)
    
    if start_pos == -1:
        # Try fuzzy matching with difflib
        matcher = difflib.SequenceMatcher(None, normalized_quote, normalized_full)
        match = matcher.find_longest_match(0, len(normalized_quote), 0, len(normalized_full))
        
        if match.size >= len(normalized_quote) * 0.8:  # 80% similarity threshold
            start_pos = match.b
        else:
            return {"found": False, "similarity": match.size / len(normalized_quote) if normalized_quote else 0}
    
    end_pos = start_pos + len(normalized_quote)
    
    # Calculate page number if page_offsets provided
    page_number = None
    if page_offsets:
        for i, offset in enumerate(page_offsets):
            if start_pos >= offset:
                page_number = i + 1
            else:
                break
    
    return {
        "found": True,
        "page": page_number,
        "start_char": start_pos,
        "end_char": end_pos,
        "similarity": 1.0
    }

def create_process_compliant_prompt(criterion_type: str, rationale_chunks: List[str], 
                                   metadata: dict, filing_date: str = None) -> str:
    """Create extraction prompt following instructions exactly."""
    
    prompt = f"""You are analyzing an AAO (Administrative Appeals Office) decision for criterion: {criterion_type}.

CRITICAL EXTRACTION RULES (from instructions):
1. Extract findings only if explicitly supported by the decision text
2. When ambiguous, use UNKNOWN and explain
3. Every non-UNKNOWN finding must include at least one direct quote with page number
4. Preserve neutral tone - do not infer facts outside the record
5. Use ONLY the controlled vocabulary provided below for denial tags

CASE METADATA:
- Case Number: {metadata.get('case_number', 'Unknown')}
- Decision Date: {metadata.get('decision_date', 'Unknown')}
- Filing Date: {filing_date or 'Unknown'}
- Field of Endeavor: {metadata.get('field_of_endeavor', 'Unknown')}
- AAO Finding: {metadata.get('aao_finding', 'Unknown')}
- Director Finding: {metadata.get('director_finding', 'Unknown')}

CONTROLLED DENIAL TAG VOCABULARY (use EXACTLY these tags):
{chr(10).join(f"- {tag}: {desc}" for tag, desc in CONTROLLED_DENIAL_TAGS.items())}

RATIONALE TEXT CHUNKS TO ANALYZE:
{chr(10).join(f"[CHUNK {i+1}] {chunk}" for i, chunk in enumerate(rationale_chunks))}

REQUIRED EXTRACTIONS:

1. EVIDENCE_ANALYSIS:
   For each piece of evidence mentioned, identify:
   - evidence_type: "publication", "letter", "judging", "award", "membership", "press", "salary", "role", "other"
   - strength_assessment: "strong", "adequate", "weak", "insufficient"
   - aao_quote: Direct quote of AAO's evaluation
   - quote_page: Page number if visible
   - quote_start_char: Character offset in text
   - quote_end_char: End character offset
   - confidence_score: 0.0-1.0 based on clarity

2. REJECTION_ANALYSIS (only if AAO finding is 'not_met'):
   For each rejection reason:
   - denial_tag: MUST be from controlled vocabulary above
   - rejection_detail: Exact AAO description
   - severity_level: "minor", "moderate", "major"
   - aao_quote: Exact quote showing rejection
   - quote_page: Page number
   - quote_start_char: Character offset
   - quote_end_char: End character offset
   - confidence_score: 0.0-1.0

3. SUCCESS_ANALYSIS (only if AAO finding is 'met'):
   - success_element: Key factor that led to success
   - evidence_strength: AAO's description of strength
   - supporting_quote: Exact quote
   - impact_level: "local", "national", "international"
   - confidence_score: 0.0-1.0

4. REFILE_GUIDANCE (REQUIRED for not_met findings):
   Generate neutral, actionable guidance:
   - guidance_text: "To satisfy this criterion in a refile, provide..."
   - specific_gaps: Array of specific deficiencies identified
   - evidence_needed: Array of evidence types that would strengthen the case

5. LINGUISTIC_ANALYSIS:
   - confidence_level: "high", "medium", "low"
   - criticism_intensity: "mild", "moderate", "severe"
   - burden_language_used: true/false
   - definitive_phrases: Array with page/char anchors
   - hedging_phrases: Array with page/char anchors
   - reasoning_quotes: Key reasoning with anchors
   - quantitative_mentions: Numbers/percentages with anchors

OUTPUT FORMAT: Return structured JSON matching this schema:
{{
  "evidence_analysis": [{{
    "evidence_type": "string",
    "strength_assessment": "strong|adequate|weak|insufficient",
    "aao_quote": "exact quote",
    "quote_page": integer,
    "quote_start_char": integer,
    "quote_end_char": integer,
    "confidence_score": float
  }}],
  "rejection_analysis": [{{
    "denial_tag": "string from controlled vocabulary",
    "rejection_detail": "exact description",
    "severity_level": "minor|moderate|major",
    "aao_quote": "exact quote",
    "quote_page": integer,
    "quote_start_char": integer,
    "quote_end_char": integer,
    "confidence_score": float
  }}],
  "success_analysis": [{{
    "success_element": "string",
    "evidence_strength": "string",
    "supporting_quote": "exact quote",
    "impact_level": "local|national|international",
    "confidence_score": float
  }}],
  "refile_guidance": {{
    "guidance_text": "To satisfy this criterion in a refile, provide...",
    "specific_gaps": ["gap 1", "gap 2"],
    "evidence_needed": ["evidence type 1", "evidence type 2"]
  }},
  "linguistic_analysis": {{
    "confidence_level": "high|medium|low",
    "criticism_intensity": "mild|moderate|severe",
    "burden_language_used": boolean,
    "definitive_phrases": [{{"text": "phrase", "page": int, "start": int, "end": int}}],
    "hedging_phrases": [{{"text": "phrase", "page": int, "start": int, "end": int}}],
    "reasoning_quotes": [{{"text": "quote", "page": int, "start": int, "end": int}}],
    "quantitative_mentions": [{{"text": "number", "page": int, "start": int, "end": int}}]
  }}
}}

CRITICAL: Use only controlled vocabulary tags. Include page/character anchors for all quotes. Generate actionable refile guidance for failed criteria."""

    return prompt

def chunk_rationale_for_analysis(rationale_text: str, max_chunk_size: int = 2000) -> List[str]:
    """Chunk rationale text for better LLM processing following instructions."""
    import re
    
    # Split by sentences first
    sentences = re.split(r'(?<=[.!?])\s+', rationale_text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def extract_criterion_insights_process(criterion_record: dict) -> dict:
    """
    Extract insights following instructions exactly.
    
    Implements multi-pass extraction with retrieval/chunking, controlled vocabulary,
    quote anchoring, and actionable refile guidance.
    """
    import time
    import json
    from typing import Dict, Any
    
    start_time = time.time()
    
    try:
        client = get_openai_client()
        
        # 1. Chunk the rationale for better processing (requirement)
        rationale_chunks = chunk_rationale_for_analysis(criterion_record['rationale'])
        
        # 2. Create-compliant prompt
        prompt = create_process_compliant_prompt(
            criterion_record['criterion'],
            rationale_chunks,
            criterion_record,
            criterion_record.get('filing_date')
        )
        
        # 3. Call LLM with structured output
        kwargs = {
            "model": settings.gpt_model,
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert legal analyst. Extract findings only if explicitly supported by the decision text. When ambiguous, use UNKNOWN and explain. Every non-UNKNOWN finding must include at least one direct quote with page number. Preserve neutral tone. Use controlled vocabulary for denial tags."
                },
                {"role": "user", "content": prompt}
            ],
        }
        
        # Only add temperature if not using o1/o3 models
        if not (settings.gpt_model.startswith("o1") or settings.gpt_model.startswith("o3")):
            kwargs["temperature"] = 0  # Deterministic extraction
        
        # Use the correct token parameter based on model
        if settings.gpt_model.startswith("o1") or settings.gpt_model.startswith("o3"):
            kwargs["max_completion_tokens"] = 4000
        else:
            kwargs["max_tokens"] = 4000
        
        response = client.chat.completions.create(**kwargs)
        
        # 4. Parse and validate response
        try:
            insights = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        
        # 5. Validate controlled vocabulary compliance
        vocab_issues = validate_controlled_vocabulary(insights)
        
        # 6. Validate quote anchoring
        quote_validation = validate_quote_anchoring(insights, criterion_record['rationale'])
        
        # 7. Calculate enhanced quality metrics
        quality_metrics = calculate_enhanced_quality_metrics(
            insights, criterion_record['rationale'], vocab_issues, quote_validation
        )
        
        # 8. Add processing metadata
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "criterion_id": criterion_record['criterion_id'],
            "extraction_successful": True,
            "insights": insights,
            "quality_metrics": {
                **quality_metrics,
                "processing_time_ms": processing_time,
                "llm_model_used": settings.gpt_model,
                "extraction_version": "2.0_process_compliant"
            }
        }
        
    except Exception as e:
        return {
            "criterion_id": criterion_record['criterion_id'],
            "extraction_successful": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "processing_time_ms": int((time.time() - start_time) * 1000)
        }

def validate_controlled_vocabulary(insights: dict) -> List[str]:
    """Validate that denial tags use controlled vocabulary from instructions."""
    issues = []
    
    for rejection in insights.get("rejection_analysis", []):
        denial_tag = rejection.get("denial_tag")
        if denial_tag and denial_tag not in CONTROLLED_DENIAL_TAGS:
            issues.append(f"Invalid denial tag: {denial_tag}")
    
    return issues

def validate_quote_anchoring(insights: dict, original_text: str) -> dict:
    """Validate quote anchoring per instructions."""
    
    validation_results = {
        "total_quotes": 0,
        "valid_quotes": 0,
        "anchored_quotes": 0,
        "invalid_quotes": []
    }
    
    # Check all quote fields in the insights
    quote_sections = [
        ("evidence_analysis", ["aao_quote"]),
        ("rejection_analysis", ["aao_quote"]),
        ("success_analysis", ["supporting_quote"]),
        ("linguistic_analysis", ["definitive_phrases", "hedging_phrases", "reasoning_quotes"])
    ]
    
    for section_name, quote_fields in quote_sections:
        section = insights.get(section_name, [])
        
        if section_name == "linguistic_analysis":
            # Handle special case of linguistic analysis with arrays
            for field in quote_fields:
                phrases = section.get(field, [])
                for phrase in phrases:
                    if isinstance(phrase, dict) and "text" in phrase:
                        validation_results["total_quotes"] += 1
                        quote_loc = find_quote_location(phrase["text"], original_text)
                        
                        if quote_loc["found"]:
                            validation_results["valid_quotes"] += 1
                            if phrase.get("page") and phrase.get("start") is not None:
                                validation_results["anchored_quotes"] += 1
                        else:
                            validation_results["invalid_quotes"].append({
                                "text": phrase["text"][:100] + "...",
                                "section": section_name,
                                "field": field
                            })
        else:
            # Handle regular sections
            if isinstance(section, list):
                for item in section:
                    if isinstance(item, dict):
                        for field in quote_fields:
                            if field in item and item[field]:
                                validation_results["total_quotes"] += 1
                                quote_loc = find_quote_location(item[field], original_text)
                                
                                if quote_loc["found"]:
                                    validation_results["valid_quotes"] += 1
                                    if item.get("quote_page") and item.get("quote_start_char") is not None:
                                        validation_results["anchored_quotes"] += 1
                                else:
                                    validation_results["invalid_quotes"].append({
                                        "text": item[field][:100] + "...",
                                        "section": section_name,
                                        "field": field
                                    })
    
    return validation_results

def calculate_enhanced_quality_metrics(insights: dict, original_text: str, 
                                     vocab_issues: List[str], quote_validation: dict) -> dict:
    """Calculate enhanced quality metrics per instructions."""
    
    # Base quality score
    quality_score = 1.0
    
    # Penalize vocabulary issues
    if vocab_issues:
        quality_score -= len(vocab_issues) * 0.1
    
    # Quote accuracy score
    quote_accuracy = 0.0
    if quote_validation["total_quotes"] > 0:
        quote_accuracy = quote_validation["valid_quotes"] / quote_validation["total_quotes"]
    
    # Quote anchoring score
    quote_anchoring_score = 0.0
    if quote_validation["total_quotes"] > 0:
        quote_anchoring_score = quote_validation["anchored_quotes"] / quote_validation["total_quotes"]
    
    # Controlled vocabulary compliance
    controlled_vocab_compliance = 1.0 - (len(vocab_issues) / max(1, len(insights.get("rejection_analysis", []))))
    
    # Confidence distribution
    confidence_counts = {"high": 0, "medium": 0, "low": 0}
    
    def count_confidence(section):
        if isinstance(section, list):
            for item in section:
                if isinstance(item, dict) and "confidence_score" in item:
                    score = item["confidence_score"]
                    if score >= 0.8:
                        confidence_counts["high"] += 1
                    elif score >= 0.5:
                        confidence_counts["medium"] += 1
                    else:
                        confidence_counts["low"] += 1
    
    count_confidence(insights.get("evidence_analysis", []))
    count_confidence(insights.get("rejection_analysis", []))
    count_confidence(insights.get("success_analysis", []))
    
    # Collect issues
    issues = vocab_issues.copy()
    if quote_validation["invalid_quotes"]:
        issues.extend([f"Invalid quote: {q['text']}" for q in quote_validation["invalid_quotes"][:3]])
    
    # Final quality score adjustment
    quality_score = max(0.0, min(1.0, quality_score))
    
    return {
        "quality_score": quality_score,
        "quote_accuracy": quote_accuracy,
        "quote_anchoring_score": quote_anchoring_score,
        "controlled_vocab_compliance": controlled_vocab_compliance,
        "confidence_distribution": confidence_counts,
        "issues_found": issues,
        "quote_validation_details": quote_validation
    }

# Update the main function to use compliant version
def extract_criterion_insights(criterion_record: dict) -> dict:
    """
    Main extraction function - now uses compliant implementation.
    """
    return extract_criterion_insights_process(criterion_record)