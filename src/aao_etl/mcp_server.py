from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from typing import Optional

from mcp.server.fastmcp import FastMCPServer
from pydantic import Field
from sqlalchemy import text

from .db import get_engine

# Lazily initialised globals so the MCP server picks up runtime configuration
_ENGINE = None
_DEFAULT_DSN: Optional[str] = None

server = FastMCPServer("aao-etl-db")

SCHEMA_OVERVIEW = (
    "decisions(decision_id, case_number, source_body, service_center, petition_type,"
    " decision_date, filing_date, outcome, field_of_endeavor, specialization, pdf_path,"
    " source_url, summary, final_merits, final_merits_rationale);\n"
    "claimed_criteria(criterion_id, decision_id→decisions, criterion, director_finding,"
    " aao_finding, rationale);\n"
    "evidence_items(evidence_id, decision_id→decisions, e_type, title, description,"
    " event_date, is_pre_filing, accepted_by_uscis);\n"
    "quotes(quote_id, decision_id→decisions, criterion_id→claimed_criteria,"
    " evidence_id→evidence_items, authority_id→authorities, quote_type, text,"
    " page_number, start_char, end_char);\n"
    "authorities(authority_id, name, type, citation); decision_authority_map(decision_id,"
    " authority_id, notes);\n"
    "issue_tags(tag_id, tag_name); decision_issue_map(decision_id, tag_id);"
    " criterion_issue_map(criterion_id, tag_id);\n"
    "exclusion_reasons(reason_id, reason_name); evidence_exclusions(evidence_id, reason_id);\n"
    "decision_text_blobs(decision_id, full_text, summary_text, summary_embedding vector(1536));\n"
    "denial_issue_taxonomy(tag_id, tag_name, category, description, created_at);\n"
    "criterion_evidence_insights(insight_id, criterion_id, criterion_type, evidence_type,"
    " strength_assessment, specific_deficiency, aao_quote, quote_page, quote_start_char,"
    " quote_end_char, confidence_score, created_at);\n"
    "criterion_rejection_patterns(pattern_id, criterion_id, criterion_type, denial_tag_id,"
    " rejection_detail, severity_level, aao_quote, quote_page, quote_start_char,"
    " quote_end_char, confidence_score, created_at);\n"
    "criterion_success_factors(factor_id, criterion_id, criterion_type, success_element,"
    " evidence_strength, supporting_quote, quote_page, quote_start_char, quote_end_char,"
    " impact_level, confidence_score, created_at);\n"
    "criterion_refile_guidance(guidance_id, criterion_id, criterion_type, guidance_text,"
    " specific_gaps[], evidence_needed[], created_at);\n"
    "aao_linguistic_analysis(analysis_id, criterion_id, confidence_level,"
    " criticism_intensity, burden_language_used, definitive_phrases jsonb,"
    " hedging_phrases jsonb, reasoning_quotes jsonb, quantitative_mentions jsonb,"
    " created_at);\n"
    "evidence_calendar_check(check_id, evidence_id, filing_date, evidence_date,"
    " is_pre_filing, post_filing_gap_days, auto_tagged, created_at);\n"
    "extraction_quality_metrics(metric_id, criterion_id, extraction_version, quality_score,"
    " quote_accuracy, quote_anchoring_score, controlled_vocab_compliance,"
    " confidence_distribution jsonb, issues_found[], processing_time_ms,"
    " llm_model_used, created_at);\n"
    "criterion_extraction_batches(batch_id, criterion_type, batch_size, processed_count,"
    " success_count, error_count, started_at, completed_at, error_log_path, status)."
)


def configure(dsn: Optional[str]) -> None:
    """Store DSN so the server can lazily create a shared engine."""
    global _DEFAULT_DSN
    _DEFAULT_DSN = dsn


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = get_engine(_DEFAULT_DSN)
    return _ENGINE


@contextmanager
def _db_connection():
    engine = _get_engine()
    with engine.connect() as conn:
        yield conn


@server.tool(description="Show key AAO ETL schema tables and relationships to guide query planning.")
def describe_schema() -> str:
    return SCHEMA_OVERVIEW


@server.tool(description="Search AAO decisions by text, outcome, criterion, or date range.")
def search_decisions(
    text_filter: str | None = Field(
        default=None,
        description="Case-insensitive substring to match against summaries, rationales, or field of endeavor.",
    ),
    outcome: str | None = Field(
        default=None,
        description="Filter by outcome enum (approved, denied, dismissed, sustained, remanded, withdrawn).",
    ),
    criterion: str | None = Field(
        default=None,
        description="Restrict to decisions that include this criterion code in claimed_criteria (e.g. AWARD).",
    ),
    start_date: date | None = Field(default=None, description="Only include decisions on/after this date."),
    end_date: date | None = Field(default=None, description="Only include decisions on/before this date."),
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of decisions to return."),
) -> str:
    where = ["1=1"]
    params: dict[str, object] = {"limit": limit}
    join_clause = ""

    if text_filter:
        params["text_a"] = f"%{text_filter}%"
        params["text_b"] = f"%{text_filter}%"
        params["text_c"] = f"%{text_filter}%"
        where.append(
            "(d.summary ILIKE :text_a OR d.final_merits_rationale ILIKE :text_b OR d.field_of_endeavor ILIKE :text_c)"
        )

    if outcome:
        params["outcome"] = outcome
        where.append("d.outcome = :outcome")

    if criterion:
        join_clause = "JOIN claimed_criteria c ON c.decision_id = d.decision_id"
        params["criterion"] = criterion
        where.append("c.criterion = :criterion")

    if start_date:
        params["start_date"] = start_date
        where.append("d.decision_date >= :start_date")

    if end_date:
        params["end_date"] = end_date
        where.append("d.decision_date <= :end_date")

    sql = text(
        f"""
        SELECT DISTINCT d.decision_id, d.case_number, d.decision_date, d.outcome, d.field_of_endeavor,
               COALESCE(NULLIF(d.summary, ''), LEFT(d.final_merits_rationale, 280)) AS synopsis
        FROM decisions d
        {join_clause}
        WHERE {' AND '.join(where)}
        ORDER BY d.decision_date DESC NULLS LAST
        LIMIT :limit
        """
    )

    with _db_connection() as conn:
        rows = conn.execute(sql, params).mappings().all()

    if not rows:
        return "No decisions matched the provided filters."

    lines = ["Found decisions:"]
    for row in rows:
        summary = (row["synopsis"] or "").replace("\n", " ")
        if len(summary) > 240:
            summary = summary[:237] + "..."
        lines.append(
            f"• #{row['decision_id']} {row['case_number'] or 'case TBD'} "
            f"({row['decision_date'] or 'date n/a'}) outcome={row['outcome'] or 'unknown'}"
        )
        if summary:
            lines.append(f"  ↳ {summary}")

    return "\n".join(lines)


@server.tool(description="Retrieve detailed breakdown of a single decision, including criteria and evidence counts.")
def get_decision_details(
    decision_id: int | None = Field(default=None, description="Primary key of the decision to inspect."),
    case_number: str | None = Field(default=None, description="Case number if the primary key is unknown."),
    include_quotes: bool = Field(default=False, description="Include up to 5 representative quotes."),
) -> str:
    if decision_id is None and not case_number:
        return "Provide either decision_id or case_number to identify the decision."

    filters = []
    params: dict[str, object] = {}
    if decision_id is not None:
        filters.append("decision_id = :decision_id")
        params["decision_id"] = decision_id
    if case_number:
        filters.append("case_number = :case_number")
        params["case_number"] = case_number

    base_sql = text(
        f"""
        SELECT decision_id, case_number, decision_date, filing_date, outcome, field_of_endeavor,
               final_merits, final_merits_rationale
        FROM decisions
        WHERE {' OR '.join(filters)}
        LIMIT 1
        """
    )

    with _db_connection() as conn:
        decision = conn.execute(base_sql, params).mappings().first()
        if not decision:
            return "No decision found with the provided identifier."

        crit_rows = conn.execute(
            text(
                """
                SELECT criterion, director_finding, aao_finding,
                       LEFT(COALESCE(rationale,''), 600) AS rationale
                FROM claimed_criteria
                WHERE decision_id = :decision_id
                ORDER BY criterion
                """
            ),
            {"decision_id": decision["decision_id"]},
        ).mappings().all()

        evidence_rows = conn.execute(
            text(
                """
                SELECT e_type, COUNT(*) as count, SUM(CASE WHEN accepted_by_uscis THEN 1 ELSE 0 END) AS accepted
                FROM evidence_items
                WHERE decision_id = :decision_id
                GROUP BY e_type
                ORDER BY count DESC
                """
            ),
            {"decision_id": decision["decision_id"]},
        ).mappings().all()

        quotes = []
        if include_quotes:
            quotes = conn.execute(
                text(
                    """
                    SELECT quote_type, LEFT(text, 500) AS text, page_number
                    FROM quotes
                    WHERE decision_id = :decision_id
                    ORDER BY quote_type, page_number NULLS LAST
                    LIMIT 5
                    """
                ),
                {"decision_id": decision["decision_id"]},
            ).mappings().all()

    lines = [
        f"Decision #{decision['decision_id']} | case {decision['case_number'] or 'n/a'}",
        f"Outcome: {decision['outcome'] or 'unknown'} | Decision date: {decision['decision_date'] or 'n/a'}",
        f"Field of endeavor: {decision['field_of_endeavor'] or 'n/a'}",
    ]

    if decision.get("final_merits"):
        lines.append(f"Final merits: {decision['final_merits']}" )
    if decision.get("final_merits_rationale"):
        rationale = decision["final_merits_rationale"].strip().replace("\n", " ")
        if len(rationale) > 400:
            rationale = rationale[:397] + "..."
        lines.append(f"Final merits rationale: {rationale}")

    if crit_rows:
        lines.append("\nCriteria findings:")
        for row in crit_rows:
            snippet = row["rationale"].replace("\n", " ") if row["rationale"] else ""
            if len(snippet) > 160:
                snippet = snippet[:157] + "..."
            lines.append(
                f"• {row['criterion']}: director={row['director_finding'] or 'n/a'}, "
                f"AAO={row['aao_finding'] or 'n/a'}"
            )
            if snippet:
                lines.append(f"  ↳ {snippet}")

    if evidence_rows:
        lines.append("\nEvidence summary (accepted/total):")
        for row in evidence_rows:
            lines.append(f"• {row['e_type']}: {row['accepted'] or 0}/{row['count']}")

    if quotes:
        lines.append("\nRepresentative quotes:")
        for row in quotes:
            text_snippet = row["text"].replace("\n", " ")
            if len(text_snippet) > 200:
                text_snippet = text_snippet[:197] + "..."
            lines.append(f"• {row['quote_type']} (page {row['page_number'] or 'n/a'}): {text_snippet}")

    return "\n".join(lines)


@server.tool(description="Analyze denial patterns or success factors for a given criterion code.")
def summarize_criterion_insights(
    criterion: str = Field(description="Criterion code to summarise, e.g. AWARD, MEMBERSHIP."),
    include_denials: bool = Field(default=True, description="Include aggregated denial pattern statistics."),
    include_success: bool = Field(default=True, description="Include aggregated success factor statistics."),
    include_guidance: bool = Field(default=False, description="Include refile guidance highlights."),
    limit: int = Field(default=5, ge=1, le=20, description="Maximum distinct insights to surface per category."),
) -> str:
    params = {"criterion": criterion, "limit": limit}
    lines = [f"Insights for criterion {criterion}:"]

    with _db_connection() as conn:
        if include_denials:
            denial_rows = conn.execute(
                text(
                    """
                    SELECT t.tag_name, t.category, COALESCE(r.rejection_detail,'') AS detail,
                           COUNT(*) AS occurrences
                    FROM criterion_rejection_patterns r
                    LEFT JOIN denial_issue_taxonomy t ON r.denial_tag_id = t.tag_id
                    WHERE r.criterion_type = :criterion
                    GROUP BY t.tag_name, t.category, COALESCE(r.rejection_detail,'')
                    ORDER BY occurrences DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()

            if denial_rows:
                lines.append("\nTop denial patterns:")
                for row in denial_rows:
                    detail = row["detail"].replace("\n", " ") if row["detail"] else ""
                    if len(detail) > 160:
                        detail = detail[:157] + "..."
                    label = row["tag_name"] or "unspecified"
                    lines.append(
                        f"• {label} (category={row['category'] or 'n/a'}, occurrences={row['occurrences']})"
                    )
                    if detail:
                        lines.append(f"  ↳ {detail}")
            else:
                lines.append("\nNo denial patterns recorded for this criterion.")

        if include_success:
            success_rows = conn.execute(
                text(
                    """
                    SELECT success_element, COALESCE(evidence_strength,'n/a') AS strength,
                           COUNT(*) AS occurrences
                    FROM criterion_success_factors
                    WHERE criterion_type = :criterion
                    GROUP BY success_element, COALESCE(evidence_strength,'n/a')
                    ORDER BY occurrences DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()

            if success_rows:
                lines.append("\nSuccess factors:")
                for row in success_rows:
                    lines.append(
                        f"• {row['success_element']} (strength={row['strength']}, occurrences={row['occurrences']})"
                    )
            else:
                lines.append("\nNo success factor records for this criterion.")

        if include_guidance:
            guidance_rows = conn.execute(
                text(
                    """
                    SELECT guidance_text
                    FROM criterion_refile_guidance
                    WHERE criterion_type = :criterion
                    ORDER BY created_at DESC
                    LIMIT :limit
                    """
                ),
                params,
            ).scalars().all()

            if guidance_rows:
                lines.append("\nRefile guidance highlights:")
                for idx, text_block in enumerate(guidance_rows, start=1):
                    snippet = text_block.replace("\n", " ")
                    if len(snippet) > 220:
                        snippet = snippet[:217] + "..."
                    lines.append(f"{idx}. {snippet}")
            else:
                lines.append("\nNo refile guidance entries recorded for this criterion.")

    return "\n".join(lines)


def run_server(dsn: Optional[str] = None) -> None:
    configure(dsn)
    server.run()
