from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .config import settings
from .models import DecisionExtraction, DocumentInfo
from .enrich import parse_decision_date  # Add missing import

def get_engine(dsn: Optional[str] = None, pool_size: int = 5, max_overflow: int = 10) -> Engine:
    """Get database engine with configurable connection pooling for parallel processing."""
    url = dsn or settings.database_url
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Pass --dsn or set it in your environment.")
    
    return create_engine(
        url, 
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False  # Set to True for SQL debugging if needed
    )

def run_sql_file(engine: Engine, path: str):
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()
    with engine.begin() as conn:
        conn.exec_driver_sql(sql)

def upsert_decision(engine: Engine, *, pdf_path: str, quick: Dict[str,Any], doc: Union[DecisionExtraction, DocumentInfo]) -> int:
    """
    Insert/update decisions with full structured extraction data.
    """
    from .enrich import parse_decision_date

    # Handle both old DocumentInfo and new DecisionExtraction models
    if isinstance(doc, DecisionExtraction):
        return upsert_decision_extraction(engine, pdf_path=pdf_path, quick=quick, doc=doc)
    else:
        return upsert_legacy_decision(engine, pdf_path=pdf_path, quick=quick, doc=doc)

def upsert_decision_extraction(engine: Engine, *, pdf_path: str, quick: Dict[str,Any], doc: DecisionExtraction) -> int:
    """Upsert a full DecisionExtraction with all related data."""
    
    # Use extracted data from LLM if available, fallback to regex
    case_number = doc.case_number or quick.get("case_number") or f"NOCASE-{pdf_path}"
    decision_date = doc.decision_date or parse_decision_date(quick.get("decision_date_text"))
    
    with engine.begin() as conn:
        # 1. Insert/update main decision record
        result = conn.execute(text("""
            INSERT INTO decisions (
                case_number, source_body, petition_type, decision_date, filing_date,
                outcome, field_of_endeavor, pdf_path, final_merits, final_merits_rationale
            )
            VALUES (:case_number, 'AAO', :petition_type, :decision_date, :filing_date, 
                    :outcome, :field_of_endeavor, :pdf_path, :final_merits, :final_merits_rationale)
            ON CONFLICT (case_number) DO UPDATE SET
                petition_type = EXCLUDED.petition_type,
                decision_date = EXCLUDED.decision_date,
                filing_date = EXCLUDED.filing_date,
                outcome = EXCLUDED.outcome,
                field_of_endeavor = EXCLUDED.field_of_endeavor,
                pdf_path = EXCLUDED.pdf_path,
                final_merits = EXCLUDED.final_merits,
                final_merits_rationale = EXCLUDED.final_merits_rationale
        """), dict(
            case_number=case_number,
            petition_type=doc.petition_type,
            decision_date=decision_date,
            filing_date=doc.filing_date,
            outcome=doc.outcome,
            field_of_endeavor=doc.field_of_endeavor,
            pdf_path=pdf_path,
            final_merits=doc.final_merits.value if doc.final_merits else None,
            final_merits_rationale=doc.final_merits_rationale,
        ))

        decision_id = conn.execute(
            text("SELECT decision_id FROM decisions WHERE case_number=:c"),
            {"c": case_number}
        ).scalar_one()

        # 2. Clear existing related data for clean upsert
        conn.execute(text("DELETE FROM claimed_criteria WHERE decision_id = :d"), {"d": decision_id})
        conn.execute(text("DELETE FROM evidence_items WHERE decision_id = :d"), {"d": decision_id})
        conn.execute(text("DELETE FROM quotes WHERE decision_id = :d"), {"d": decision_id})

        # 3. Insert criteria analysis
        for criterion in doc.criteria:
            conn.execute(text("""
                INSERT INTO claimed_criteria (decision_id, criterion, director_finding, aao_finding, rationale)
                VALUES (:decision_id, :criterion, :director_finding, :aao_finding, :rationale)
            """), dict(
                decision_id=decision_id,
                criterion=criterion.criterion.value,
                director_finding=criterion.director_finding.value,
                aao_finding=criterion.aao_finding.value,
                rationale=criterion.rationale,
            ))
            
            criterion_id = conn.execute(text("SELECT lastval()")).scalar_one()
            
            # Insert quotes for this criterion
            for quote in criterion.quotes:
                conn.execute(text("""
                    INSERT INTO quotes (decision_id, criterion_id, quote_type, text, page_number, start_char, end_char)
                    VALUES (:decision_id, :criterion_id, 'criterion', :text, :page, :start_char, :end_char)
                """), dict(
                    decision_id=decision_id,
                    criterion_id=criterion_id,
                    text=quote.text,
                    page=quote.page,
                    start_char=quote.start_char,
                    end_char=quote.end_char,
                ))

            # Insert issue tags for this criterion
            for tag in criterion.issue_tags:
                tag_id = upsert_tag(conn, tag)
                conn.execute(text("""
                    INSERT INTO criterion_issue_map (criterion_id, tag_id)
                    VALUES (:criterion_id, :tag_id) ON CONFLICT DO NOTHING
                """), {"criterion_id": criterion_id, "tag_id": tag_id})

        # 4. Insert evidence items
        for evidence in doc.evidence:
            conn.execute(text("""
                INSERT INTO evidence_items (
                    decision_id, e_type, title, description, event_date, is_pre_filing, accepted_by_uscis
                )
                VALUES (:decision_id, :e_type, :title, :description, :event_date, :is_pre_filing, :accepted_by_uscis)
            """), dict(
                decision_id=decision_id,
                e_type=evidence.e_type.value,
                title=evidence.title,
                description=evidence.description,
                event_date=evidence.event_date,
                is_pre_filing=evidence.is_pre_filing,
                accepted_by_uscis=evidence.accepted_by_uscis,
            ))
            
            evidence_id = conn.execute(text("SELECT lastval()")).scalar_one()
            
            # Insert quotes for this evidence
            for quote in evidence.quotes:
                conn.execute(text("""
                    INSERT INTO quotes (decision_id, evidence_id, quote_type, text, page_number, start_char, end_char)
                    VALUES (:decision_id, :evidence_id, 'evidence', :text, :page, :start_char, :end_char)
                """), dict(
                    decision_id=decision_id,
                    evidence_id=evidence_id,
                    text=quote.text,
                    page=quote.page,
                    start_char=quote.start_char,
                    end_char=quote.end_char,
                ))

        # 5. Insert authorities
        for authority in doc.authorities:
            conn.execute(text("""
                INSERT INTO authorities (name, type, citation)
                VALUES (:name, :type, :citation)
                ON CONFLICT (name) DO UPDATE SET type = EXCLUDED.type, citation = EXCLUDED.citation
            """), dict(name=authority.name, type=authority.type, citation=authority.citation))
            
            auth_id = conn.execute(
                text("SELECT authority_id FROM authorities WHERE name=:n"), 
                {"n": authority.name}
            ).scalar_one()
            
            conn.execute(text("""
                INSERT INTO decision_authority_map (decision_id, authority_id)
                VALUES (:d, :a) ON CONFLICT DO NOTHING
            """), {"d": decision_id, "a": auth_id})

            # Insert quotes for this authority
            for quote in authority.quotes:
                conn.execute(text("""
                    INSERT INTO quotes (decision_id, authority_id, quote_type, text, page_number, start_char, end_char)
                    VALUES (:decision_id, :authority_id, 'authority', :text, :page, :start_char, :end_char)
                """), dict(
                    decision_id=decision_id,
                    authority_id=auth_id,
                    text=quote.text,
                    page=quote.page,
                    start_char=quote.start_char,
                    end_char=quote.end_char,
                ))

        # 6. Insert final merits quotes
        for quote in doc.final_merits_quotes:
            conn.execute(text("""
                INSERT INTO quotes (decision_id, quote_type, text, page_number, start_char, end_char)
                VALUES (:decision_id, 'final_merits', :text, :page, :start_char, :end_char)
            """), dict(
                decision_id=decision_id,
                text=quote.text,
                page=quote.page,
                start_char=quote.start_char,
                end_char=quote.end_char,
            ))

        # 7. Insert global issue tags
        for tag in doc.global_issue_tags:
            tag_id = upsert_tag(conn, tag)
            conn.execute(text("""
                INSERT INTO decision_issue_map (decision_id, tag_id)
                VALUES (:decision_id, :tag_id) ON CONFLICT DO NOTHING
            """), {"decision_id": decision_id, "tag_id": tag_id})

        # 8. Insert full text blob
        full_text = getattr(doc, 'full_text', None)
        summary_text = getattr(doc, 'summary', None)
        summary_embedding = getattr(doc, 'summary_embedding', None)
        
        conn.execute(text("""
            INSERT INTO decision_text_blobs (decision_id, full_text, summary_text, summary_embedding)
            VALUES (:d, :full_text, :summary_text, :summary_embedding)
            ON CONFLICT (decision_id) DO UPDATE SET
                full_text = EXCLUDED.full_text,
                summary_text = EXCLUDED.summary_text,
                summary_embedding = EXCLUDED.summary_embedding
        """), dict(
            d=decision_id,
            full_text=full_text,
            summary_text=summary_text,
            summary_embedding=summary_embedding,
        ))

    return decision_id

# =============================================================================
# CRITERION INSIGHTS DATA FETCHING FOR LLM ANALYSIS
# =============================================================================

def get_criteria_for_analysis(engine: Engine, criterion_type: Optional[str] = None, 
                              batch_size: int = 25, offset: int = 0,
                              exclude_processed: bool = True) -> List[Dict[str, Any]]:
    """
    Fetch criteria records for LLM analysis.
    
    Args:
        engine: Database engine
        criterion_type: Specific criterion to fetch (e.g., 'AWARD', 'MEMBERSHIP')
        batch_size: Number of records to return
        offset: Starting offset for pagination
        exclude_processed: Skip records already processed by LLM
    
    Returns:
        List of criteria records with metadata
    """
    where_clauses = ["cc.rationale IS NOT NULL", "LENGTH(cc.rationale) > 50"]
    params = {"batch_size": batch_size, "offset": offset}
    
    if criterion_type:
        where_clauses.append("cc.criterion = :criterion_type")
        params["criterion_type"] = criterion_type
    
    if exclude_processed:
        where_clauses.append("""
            NOT EXISTS (
                SELECT 1 FROM extraction_quality_metrics eqm 
                WHERE eqm.criterion_id = cc.criterion_id 
                AND eqm.extraction_version = '2.0'
            )
        """)
    
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
    SELECT 
        cc.criterion_id,
        cc.decision_id,
        cc.criterion,
        cc.director_finding,
        cc.aao_finding,
        cc.rationale,
        d.case_number,
        d.decision_date,
        d.outcome,
        d.field_of_endeavor,
        d.specialization,
        LENGTH(cc.rationale) as rationale_length
    FROM claimed_criteria cc
    JOIN decisions d ON cc.decision_id = d.decision_id
    WHERE {where_clause}
    ORDER BY cc.criterion, cc.criterion_id
    LIMIT :batch_size OFFSET :offset
    """
    
    with engine.begin() as conn:
        result = conn.execute(text(query), params)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]

def get_criteria_counts_by_type(engine: Engine, exclude_processed: bool = True) -> Dict[str, int]:
    """Get count of criteria records by type for processing planning."""
    where_clause = "cc.rationale IS NOT NULL AND LENGTH(cc.rationale) > 50"
    
    if exclude_processed:
        where_clause += """
            AND NOT EXISTS (
                SELECT 1 FROM extraction_quality_metrics eqm 
                WHERE eqm.criterion_id = cc.criterion_id 
                AND eqm.extraction_version = '2.0'
            )
        """
    
    query = f"""
    SELECT 
        cc.criterion,
        COUNT(*) as count
    FROM claimed_criteria cc
    WHERE {where_clause}
    GROUP BY cc.criterion
    ORDER BY cc.criterion
    """
    
    with engine.begin() as conn:
        result = conn.execute(text(query))
        return {row[0]: row[1] for row in result.fetchall()}

def create_extraction_batch(engine: Engine, criterion_type: str, batch_size: int) -> int:
    """Create a new extraction batch record for tracking."""
    with engine.begin() as conn:
        result = conn.execute(text("""
            INSERT INTO criterion_extraction_batches (criterion_type, batch_size)
            VALUES (:criterion_type, :batch_size)
            RETURNING batch_id
        """), {"criterion_type": criterion_type, "batch_size": batch_size})
        return result.scalar_one()

def update_extraction_batch(engine: Engine, batch_id: int, 
                           processed_count: int = None, success_count: int = None,
                           error_count: int = None, status: str = None,
                           error_log_path: str = None) -> None:
    """Update extraction batch progress."""
    updates = []
    params = {"batch_id": batch_id}
    
    if processed_count is not None:
        updates.append("processed_count = :processed_count")
        params["processed_count"] = processed_count
    
    if success_count is not None:
        updates.append("success_count = :success_count") 
        params["success_count"] = success_count
        
    if error_count is not None:
        updates.append("error_count = :error_count")
        params["error_count"] = error_count
    
    if status is not None:
        updates.append("status = :status")
        params["status"] = status
        if status in ('completed', 'failed'):
            updates.append("completed_at = NOW()")
    
    if error_log_path is not None:
        updates.append("error_log_path = :error_log_path")
        params["error_log_path"] = error_log_path
    
    if updates:
        query = f"UPDATE criterion_extraction_batches SET {', '.join(updates)} WHERE batch_id = :batch_id"
        with engine.begin() as conn:
            conn.execute(text(query), params)

def _sanitize_insights_data(insights: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize LLM-generated insights data to match database constraints."""
    
    # Valid values for strength_assessment (from your database constraints)
    VALID_STRENGTH_ASSESSMENTS = {
        'strong', 'moderate', 'weak', 'insufficient', 'missing'
    }
    
    # Valid values for severity_level
    VALID_SEVERITY_LEVELS = {
        'minor', 'moderate', 'major', 'critical'
    }
    
    def sanitize_quote_page(page_value):
        """Convert quote_page to valid integer or None."""
        if page_value in [None, 'UNKNOWN', 'unknown', '']:
            return None
        try:
            return int(page_value)
        except (ValueError, TypeError):
            return None
    
    def sanitize_strength_assessment(value):
        """Ensure strength_assessment is valid or convert to default."""
        if not value or str(value).upper() == 'UNKNOWN':
            return 'insufficient'  # Default fallback
        
        value_lower = str(value).lower()
        if value_lower in VALID_STRENGTH_ASSESSMENTS:
            return value_lower
        
        # Try to map common variations
        mapping = {
            'very_strong': 'strong',
            'very_weak': 'weak',
            'not_provided': 'missing',
            'none': 'missing',
            'unclear': 'insufficient'
        }
        return mapping.get(value_lower, 'insufficient')
    
    def sanitize_severity_level(value):
        """Ensure severity_level is valid or convert to default."""
        if not value or str(value).upper() == 'UNKNOWN':
            return 'moderate'  # Default fallback
        
        value_lower = str(value).lower()
        if value_lower in VALID_SEVERITY_LEVELS:
            return value_lower
        
        # Try to map common variations
        mapping = {
            'high': 'major',
            'low': 'minor',
            'severe': 'critical'
        }
        return mapping.get(value_lower, 'moderate')
    
    # Create a deep copy to avoid modifying the original
    import copy
    sanitized = copy.deepcopy(insights)
    
    # Sanitize evidence analysis
    if 'evidence_analysis' in sanitized:
        for evidence in sanitized['evidence_analysis']:
            evidence['strength_assessment'] = sanitize_strength_assessment(
                evidence.get('strength_assessment')
            )
            evidence['quote_page'] = sanitize_quote_page(
                evidence.get('quote_page')
            )
    
    # Sanitize rejection analysis
    if 'rejection_analysis' in sanitized:
        for rejection in sanitized['rejection_analysis']:
            rejection['severity_level'] = sanitize_severity_level(
                rejection.get('severity_level')
            )
            rejection['quote_page'] = sanitize_quote_page(
                rejection.get('quote_page')
            )
    
    # Sanitize success analysis
    if 'success_analysis' in sanitized:
        for success in sanitized['success_analysis']:
            success['quote_page'] = sanitize_quote_page(
                success.get('quote_page')
            )
    
    return sanitized

def store_criterion_insights(engine: Engine, criterion_id: int, insights: Dict[str, Any],
                            extraction_version: str = "2.0") -> None:
    """Store LLM-extracted insights for a criterion with proper data validation."""
    
    # Sanitize the insights data before processing
    sanitized_insights = _sanitize_insights_data(insights)
    
    with engine.begin() as conn:
        # Get criterion metadata
        criterion_meta = conn.execute(text("""
            SELECT cc.criterion, cc.aao_finding, d.filing_date
            FROM claimed_criteria cc 
            JOIN decisions d ON cc.decision_id = d.decision_id
            WHERE cc.criterion_id = :criterion_id
        """), {"criterion_id": criterion_id}).fetchone()
        
        if not criterion_meta:
            raise ValueError(f"Criterion {criterion_id} not found")
        
        criterion_type = criterion_meta[0]
        aao_finding = criterion_meta[1]
        filing_date = criterion_meta[2]
        
        # Store evidence insights with quote anchoring
        for evidence in sanitized_insights.get("evidence_analysis", []):
            conn.execute(text("""
                INSERT INTO criterion_evidence_insights (
                    criterion_id, criterion_type, evidence_type, strength_assessment,
                    specific_deficiency, aao_quote, quote_page, quote_start_char, quote_end_char,
                    confidence_score
                )
                VALUES (:criterion_id, :criterion_type, :evidence_type, :strength_assessment,
                        :specific_deficiency, :aao_quote, :quote_page, :quote_start_char, :quote_end_char,
                        :confidence_score)
                ON CONFLICT (criterion_id, evidence_type) DO UPDATE SET
                    strength_assessment = EXCLUDED.strength_assessment,
                    specific_deficiency = EXCLUDED.specific_deficiency,
                    aao_quote = EXCLUDED.aao_quote,
                    quote_page = EXCLUDED.quote_page,
                    quote_start_char = EXCLUDED.quote_start_char,
                    quote_end_char = EXCLUDED.quote_end_char,
                    confidence_score = EXCLUDED.confidence_score
            """), {
                "criterion_id": criterion_id,
                "criterion_type": criterion_type,
                "evidence_type": evidence.get("evidence_type"),
                "strength_assessment": evidence.get("strength_assessment"),
                "specific_deficiency": evidence.get("specific_deficiency"),
                "aao_quote": evidence.get("aao_quote"),
                "quote_page": evidence.get("quote_page"),
                "quote_start_char": evidence.get("quote_start_char"),
                "quote_end_char": evidence.get("quote_end_char"),
                "confidence_score": evidence.get("confidence_score", 0.0)
            })
        
        # Store rejection patterns using controlled vocabulary
        if aao_finding == 'not_met':
            for rejection in sanitized_insights.get("rejection_analysis", []):
                # Get tag_id from controlled vocabulary
                tag_name = rejection.get("denial_tag")
                if tag_name:
                    tag_result = conn.execute(text("""
                        SELECT tag_id FROM denial_issue_taxonomy WHERE tag_name = :tag_name
                    """), {"tag_name": tag_name}).fetchone()
                    
                    if tag_result:
                        conn.execute(text("""
                            INSERT INTO criterion_rejection_patterns (
                                criterion_id, criterion_type, denial_tag_id, rejection_detail,
                                severity_level, aao_quote, quote_page, quote_start_char, quote_end_char,
                                confidence_score
                            )
                            VALUES (:criterion_id, :criterion_type, :denial_tag_id, :rejection_detail,
                                    :severity_level, :aao_quote, :quote_page, :quote_start_char, :quote_end_char,
                                    :confidence_score)
                        """), {
                            "criterion_id": criterion_id,
                            "criterion_type": criterion_type,
                            "denial_tag_id": tag_result[0],
                            "rejection_detail": rejection.get("rejection_detail"),
                            "severity_level": rejection.get("severity_level"),
                            "aao_quote": rejection.get("aao_quote"),
                            "quote_page": rejection.get("quote_page"),
                            "quote_start_char": rejection.get("quote_start_char"),
                            "quote_end_char": rejection.get("quote_end_char"),
                            "confidence_score": rejection.get("confidence_score", 0.0)
                        })
        
        # Store success factors (only for met criteria)
        if aao_finding == 'met':
            for success in sanitized_insights.get("success_analysis", []):
                # Handle UNKNOWN impact_level by setting to NULL
                impact_level = success.get("impact_level")
                if impact_level and impact_level.upper() == 'UNKNOWN':
                    impact_level = None
                
                conn.execute(text("""
                    INSERT INTO criterion_success_factors (
                        criterion_id, criterion_type, success_element, evidence_strength,
                        supporting_quote, quote_page, quote_start_char, quote_end_char,
                        impact_level, confidence_score
                    )
                    VALUES (:criterion_id, :criterion_type, :success_element, :evidence_strength,
                            :supporting_quote, :quote_page, :quote_start_char, :quote_end_char,
                            :impact_level, :confidence_score)
                """), {
                    "criterion_id": criterion_id,
                    "criterion_type": criterion_type,
                    "success_element": success.get("success_element"),
                    "evidence_strength": success.get("evidence_strength"),
                    "supporting_quote": success.get("supporting_quote"),
                    "quote_page": success.get("quote_page"),
                    "quote_start_char": success.get("quote_start_char"),
                    "quote_end_char": success.get("quote_end_char"),
                    "impact_level": impact_level,
                    "confidence_score": success.get("confidence_score", 0.0)
                })
        
        # Store actionable refile guidance (new requirement)
        if aao_finding == 'not_met' and sanitized_insights.get("refile_guidance"):
            guidance = sanitized_insights["refile_guidance"]
            
            # Convert Python lists to PostgreSQL arrays (not JSON strings)
            specific_gaps = guidance.get("specific_gaps", [])
            evidence_needed = guidance.get("evidence_needed", [])
            
            conn.execute(text("""
                INSERT INTO criterion_refile_guidance (
                    criterion_id, criterion_type, guidance_text, specific_gaps, evidence_needed
                )
                VALUES (:criterion_id, :criterion_type, :guidance_text, :specific_gaps, :evidence_needed)
                ON CONFLICT (criterion_id) DO UPDATE SET
                    guidance_text = EXCLUDED.guidance_text,
                    specific_gaps = EXCLUDED.specific_gaps,
                    evidence_needed = EXCLUDED.evidence_needed
            """), {
                "criterion_id": criterion_id,
                "criterion_type": criterion_type,
                "guidance_text": guidance.get("guidance_text"),
                "specific_gaps": specific_gaps,  # Pass Python list directly for TEXT[] column
                "evidence_needed": evidence_needed  # Pass Python list directly for TEXT[] column
            })
        
        # Store enhanced linguistic analysis with quote anchoring
        linguistic = sanitized_insights.get("linguistic_analysis", {})
        
        # Convert Python objects to JSON strings for JSONB columns
        import json
        
        conn.execute(text("""
            INSERT INTO aao_linguistic_analysis (
                criterion_id, confidence_level, criticism_intensity, burden_language_used,
                definitive_phrases, hedging_phrases, reasoning_quotes, quantitative_mentions
            )
            VALUES (:criterion_id, :confidence_level, :criticism_intensity, :burden_language_used,
                    :definitive_phrases, :hedging_phrases, :reasoning_quotes, :quantitative_mentions)
            ON CONFLICT (criterion_id) DO UPDATE SET
                confidence_level = EXCLUDED.confidence_level,
                criticism_intensity = EXCLUDED.criticism_intensity,
                burden_language_used = EXCLUDED.burden_language_used,
                definitive_phrases = EXCLUDED.definitive_phrases,
                hedging_phrases = EXCLUDED.hedging_phrases,
                reasoning_quotes = EXCLUDED.reasoning_quotes,
                quantitative_mentions = EXCLUDED.quantitative_mentions
        """), {
            "criterion_id": criterion_id,
            "confidence_level": linguistic.get("confidence_level"),
            "criticism_intensity": linguistic.get("criticism_intensity"),
            "burden_language_used": linguistic.get("burden_language_used", False),
            "definitive_phrases": json.dumps(linguistic.get("definitive_phrases", [])),
            "hedging_phrases": json.dumps(linguistic.get("hedging_phrases", [])),
            "reasoning_quotes": json.dumps(linguistic.get("reasoning_quotes", [])),
            "quantitative_mentions": json.dumps(linguistic.get("quantitative_mentions", []))
        })
        
        # Store enhanced quality metrics
        quality = sanitized_insights.get("quality_metrics", {})
        conn.execute(text("""
            INSERT INTO extraction_quality_metrics (
                criterion_id, extraction_version, quality_score, quote_accuracy,
                quote_anchoring_score, controlled_vocab_compliance, confidence_distribution,
                issues_found, processing_time_ms, llm_model_used
            )
            VALUES (:criterion_id, :extraction_version, :quality_score, :quote_accuracy,
                    :quote_anchoring_score, :controlled_vocab_compliance, :confidence_distribution,
                    :issues_found, :processing_time_ms, :llm_model_used)
            ON CONFLICT (criterion_id, extraction_version) DO UPDATE SET
                quality_score = EXCLUDED.quality_score,
                quote_accuracy = EXCLUDED.quote_accuracy,
                quote_anchoring_score = EXCLUDED.quote_anchoring_score,
                controlled_vocab_compliance = EXCLUDED.controlled_vocab_compliance,
                confidence_distribution = EXCLUDED.confidence_distribution,
                issues_found = EXCLUDED.issues_found,
                processing_time_ms = EXCLUDED.processing_time_ms,
                llm_model_used = EXCLUDED.llm_model_used
        """), {
            "criterion_id": criterion_id,
            "extraction_version": extraction_version,
            "quality_score": quality.get("quality_score", 0.0),
            "quote_accuracy": quality.get("quote_accuracy", 0.0),
            "quote_anchoring_score": quality.get("quote_anchoring_score", 0.0),
            "controlled_vocab_compliance": quality.get("controlled_vocab_compliance", 0.0),
            "confidence_distribution": json.dumps(quality.get("confidence_distribution", {})),
            "issues_found": quality.get("issues_found", []),
            "processing_time_ms": quality.get("processing_time_ms"),
            "llm_model_used": quality.get("llm_model_used", "unknown")
        })

def validate_evidence_calendar(engine: Engine, decision_id: int) -> Dict[str, Any]:
    """Validate evidence dates against filing date per instructions."""
    
    with engine.begin() as conn:
        # Get filing date
        filing_result = conn.execute(text("""
            SELECT filing_date FROM decisions WHERE decision_id = :decision_id
        """), {"decision_id": decision_id}).fetchone()
        
        if not filing_result or not filing_result[0]:
            return {"status": "no_filing_date", "evidence_checked": 0}
        
        filing_date = filing_result[0]
        
        # Get evidence items with dates
        evidence_items = conn.execute(text("""
            SELECT evidence_id, event_date FROM evidence_items 
            WHERE decision_id = :decision_id AND event_date IS NOT NULL
        """), {"decision_id": decision_id}).fetchall()
        
        results = {
            "status": "completed",
            "filing_date": filing_date.isoformat(),
            "evidence_checked": len(evidence_items),
            "post_filing_count": 0,
            "auto_tagged": []
        }
        
        for evidence_id, event_date in evidence_items:
            is_pre_filing = event_date <= filing_date
            post_filing_gap_days = (event_date - filing_date).days if not is_pre_filing else 0
            
            # Store calendar check result
            conn.execute(text("""
                INSERT INTO evidence_calendar_check (
                    evidence_id, filing_date, evidence_date, is_pre_filing, post_filing_gap_days
                )
                VALUES (:evidence_id, :filing_date, :evidence_date, :is_pre_filing, :post_filing_gap_days)
                ON CONFLICT (evidence_id) DO UPDATE SET
                    filing_date = EXCLUDED.filing_date,
                    evidence_date = EXCLUDED.evidence_date,
                    is_pre_filing = EXCLUDED.is_pre_filing,
                    post_filing_gap_days = EXCLUDED.post_filing_gap_days
            """), {
                "evidence_id": evidence_id,
                "filing_date": filing_date,
                "evidence_date": event_date,
                "is_pre_filing": is_pre_filing,
                "post_filing_gap_days": post_filing_gap_days
            })
            
            # Auto-tag post-filing evidence
            if not is_pre_filing:
                results["post_filing_count"] += 1
                
                # Check if post_filing_evidence tag exists
                post_filing_tag_result = conn.execute(text("""
                    SELECT tag_id FROM denial_issue_taxonomy WHERE tag_name = 'post_filing_evidence'
                """)).fetchone()
                
                if post_filing_tag_result:
                    # Add to issue tags if not already present
                    conn.execute(text("""
                        INSERT INTO evidence_exclusions (evidence_id, reason_id)
                        SELECT :evidence_id, reason_id FROM exclusion_reasons 
                        WHERE reason_name = 'post_filing'
                        ON CONFLICT DO NOTHING
                    """), {"evidence_id": evidence_id})
                    
                    results["auto_tagged"].append({
                        "evidence_id": evidence_id,
                        "gap_days": post_filing_gap_days
                    })
                    
                    # Mark as auto-tagged
                    conn.execute(text("""
                        UPDATE evidence_calendar_check 
                        SET auto_tagged = TRUE 
                        WHERE evidence_id = :evidence_id
                    """), {"evidence_id": evidence_id})
        
        return results

def get_processing_progress(engine: Engine, criterion_type: Optional[str] = None) -> Dict[str, Any]:
    """Get processing progress statistics."""
    where_clause = ""
    params = {}
    
    if criterion_type:
        where_clause = "WHERE criterion_type = :criterion_type"
        params["criterion_type"] = criterion_type
    
    query = f"""
    SELECT 
        criterion_type,
        COUNT(*) as total_batches,
        SUM(batch_size) as total_records,
        SUM(processed_count) as processed_records,
        SUM(success_count) as success_records,
        SUM(error_count) as error_records,
        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_batches,
        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_batches,
        COUNT(CASE WHEN status = 'running' THEN 1 END) as running_batches
    FROM criterion_extraction_batches
    {where_clause}
    GROUP BY criterion_type
    ORDER BY criterion_type
    """
    
    with engine.begin() as conn:
        result = conn.execute(text(query), params)
        columns = result.keys()
        return [dict(zip(columns, row)) for row in result.fetchall()]

def upsert_tag(conn, tag_name: str) -> int:
    """Insert tag if not exists and return tag_id."""
    conn.execute(text("""
        INSERT INTO issue_tags (tag_name) VALUES (:tag) ON CONFLICT (tag_name) DO NOTHING
    """), {"tag": tag_name})
    return conn.execute(
        text("SELECT tag_id FROM issue_tags WHERE tag_name = :tag"), 
        {"tag": tag_name}
    ).scalar_one()

def upsert_legacy_decision(engine: Engine, *, pdf_path: str, quick: Dict[str,Any], doc: DocumentInfo) -> int:
    """Legacy upsert for DocumentInfo model (backwards compatibility)."""
    from .enrich import parse_decision_date

    petition_type = None
    if doc.beneficiary_status:
        s = doc.beneficiary_status.lower()
        if "eb-1" in s or "eb1" in s or "eb-1a" in s or "extraordinary" in s:
            petition_type = "EB1A"

    decision_date = parse_decision_date(quick.get("decision_date_text"))
    outcome_map = {"dismissed":"dismissed","sustained":"sustained","rejected":"denied"}
    outcome = outcome_map.get(quick.get("outcome_guess") or "", None)

    case_number = quick.get("case_number") or f"NOCASE-{pdf_path}"

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO decisions (case_number, source_body, petition_type, decision_date, outcome, pdf_path, summary)
            VALUES (:case_number, 'AAO', :petition_type, :decision_date, :outcome, :pdf_path, :summary)
            ON CONFLICT (case_number) DO UPDATE
              SET petition_type = EXCLUDED.petition_type,
                  decision_date = EXCLUDED.decision_date,
                  outcome       = EXCLUDED.outcome,
                  pdf_path      = EXCLUDED.pdf_path,
                  summary       = EXCLUDED.summary
        """), dict(
            case_number=case_number,
            petition_type=petition_type,
            decision_date=decision_date,
            outcome=outcome,
            pdf_path=pdf_path,
            summary=(doc.summary or None),
        ))

        decision_id = conn.execute(
            text("SELECT decision_id FROM decisions WHERE case_number=:c"),
            {"c": case_number}
        ).scalar_one()

        # authorities (e.g., Kazarian)
        for name in (quick.get("authorities") or []):
            conn.execute(text("""
              INSERT INTO authorities (name, type) VALUES (:name, 'Circuit case')
              ON CONFLICT DO NOTHING
            """), {"name": name})
            auth_id = conn.execute(text("SELECT authority_id FROM authorities WHERE name=:n"), {"n": name}).scalar_one()
            conn.execute(text("""
              INSERT INTO decision_authority_map (decision_id, authority_id)
              VALUES (:d,:a) ON CONFLICT DO NOTHING
            """), {"d": decision_id, "a": auth_id})

        # text + embeddings
        conn.execute(text("""
          INSERT INTO decision_text_blobs (decision_id, full_text, summary_text, summary_embedding)
          VALUES (:d, :full_text, :summary_text, :summary_embedding)
          ON CONFLICT (decision_id) DO UPDATE
            SET full_text = EXCLUDED.full_text,
                summary_text = EXCLUDED.summary_text,
                summary_embedding = EXCLUDED.summary_embedding
        """), dict(
            d=decision_id,
            full_text=(doc.full_text or None),
            summary_text=(doc.summary or None),
            summary_embedding=(doc.summary_embedding if doc.summary_embedding else None),
        ))

    return decision_id
