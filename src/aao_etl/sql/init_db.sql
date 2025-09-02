-- Enable pgvector (requires superuser once per DB)
CREATE EXTENSION IF NOT EXISTS vector;

-- Outcome enum
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'decision_outcome') THEN
    CREATE TYPE decision_outcome AS ENUM ('approved','denied','dismissed','sustained','remanded','withdrawn');
  END IF;
END$$;

-- Finding enum for criteria
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'finding_result') THEN
    CREATE TYPE finding_result AS ENUM ('met','not_met','not_analyzed','unknown');
  END IF;
END$$;

-- Evidence type enum
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'evidence_type') THEN
    CREATE TYPE evidence_type AS ENUM ('PUBLICATION','LETTER','JUDGING','AWARD','MEMBERSHIP','PRESS','SOFTWARE','PATENT','ADOPTION','SALARY','ROLE','OTHER');
  END IF;
END$$;

-- Final merits enum
DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'final_merits_result') THEN
    CREATE TYPE final_merits_result AS ENUM ('favorable','unfavorable','not_reached');
  END IF;
END$$;

-- Core table
CREATE TABLE IF NOT EXISTS decisions (
  decision_id       BIGSERIAL PRIMARY KEY,
  case_number       TEXT UNIQUE,
  source_body       TEXT NOT NULL DEFAULT 'AAO',
  service_center    TEXT,
  petition_type     TEXT,
  decision_date     DATE,
  filing_date       DATE,
  outcome           decision_outcome,
  field_of_endeavor TEXT,
  specialization    TEXT,
  pdf_path          TEXT,
  source_url        TEXT,
  summary           TEXT,
  final_merits      final_merits_result,
  final_merits_rationale TEXT
);

-- Criteria analysis
CREATE TABLE IF NOT EXISTS claimed_criteria (
  criterion_id      BIGSERIAL PRIMARY KEY,
  decision_id       BIGINT REFERENCES decisions(decision_id) ON DELETE CASCADE,
  criterion         TEXT NOT NULL, -- AWARD, MEMBERSHIP, etc.
  director_finding  finding_result,
  aao_finding       finding_result,
  rationale         TEXT,
  UNIQUE(decision_id, criterion)
);

-- Evidence items
CREATE TABLE IF NOT EXISTS evidence_items (
  evidence_id       BIGSERIAL PRIMARY KEY,
  decision_id       BIGINT REFERENCES decisions(decision_id) ON DELETE CASCADE,
  e_type            evidence_type,
  title             TEXT,
  description       TEXT,
  event_date        DATE,
  is_pre_filing     BOOLEAN,
  accepted_by_uscis BOOLEAN
);

-- Quotes (for criteria, evidence, authorities, final merits)
CREATE TABLE IF NOT EXISTS quotes (
  quote_id          BIGSERIAL PRIMARY KEY,
  decision_id       BIGINT REFERENCES decisions(decision_id) ON DELETE CASCADE,
  criterion_id      BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  evidence_id       BIGINT REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
  authority_id      BIGINT REFERENCES authorities(authority_id) ON DELETE CASCADE,
  quote_type        TEXT, -- 'criterion', 'evidence', 'authority', 'final_merits'
  text              TEXT NOT NULL,
  page_number       INTEGER,
  start_char        INTEGER,
  end_char          INTEGER
);

-- Authorities & map
CREATE TABLE IF NOT EXISTS authorities (
  authority_id BIGSERIAL PRIMARY KEY,
  name         TEXT NOT NULL,
  type         TEXT,
  citation     TEXT
);

CREATE TABLE IF NOT EXISTS decision_authority_map (
  decision_id  BIGINT REFERENCES decisions(decision_id) ON DELETE CASCADE,
  authority_id BIGINT REFERENCES authorities(authority_id) ON DELETE CASCADE,
  notes        TEXT,
  PRIMARY KEY(decision_id, authority_id)
);

-- Issue tags
CREATE TABLE IF NOT EXISTS issue_tags (
  tag_id        BIGSERIAL PRIMARY KEY,
  tag_name      TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS decision_issue_map (
  decision_id   BIGINT REFERENCES decisions(decision_id) ON DELETE CASCADE,
  tag_id        BIGINT REFERENCES issue_tags(tag_id) ON DELETE CASCADE,
  PRIMARY KEY(decision_id, tag_id)
);

CREATE TABLE IF NOT EXISTS criterion_issue_map (
  criterion_id  BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  tag_id        BIGINT REFERENCES issue_tags(tag_id) ON DELETE CASCADE,
  PRIMARY KEY(criterion_id, tag_id)
);

-- Exclusion reasons
CREATE TABLE IF NOT EXISTS exclusion_reasons (
  reason_id     BIGSERIAL PRIMARY KEY,
  reason_name   TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence_exclusions (
  evidence_id   BIGINT REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
  reason_id     BIGINT REFERENCES exclusion_reasons(reason_id) ON DELETE CASCADE,
  PRIMARY KEY(evidence_id, reason_id)
);

-- Text + embeddings
CREATE TABLE IF NOT EXISTS decision_text_blobs (
  decision_id        BIGINT PRIMARY KEY REFERENCES decisions(decision_id) ON DELETE CASCADE,
  full_text          TEXT,
  summary_text       TEXT,
  summary_embedding  vector(1536)  -- text-embedding-3-small
);

-- =============================================================================
-- LLM CRITERION INSIGHTS TABLES 
-- =============================================================================

-- Controlled vocabulary for denial issue tags (from  instructions)
CREATE TABLE IF NOT EXISTS denial_issue_taxonomy (
  tag_id            BIGSERIAL PRIMARY KEY,
  tag_name          TEXT UNIQUE NOT NULL,
  category          TEXT NOT NULL, -- 'evidence', 'methodology', 'scope', 'timing'
  description       TEXT,
  created_at        TIMESTAMP DEFAULT NOW()
);

-- Insert controlled vocabulary tags from instructions
INSERT INTO denial_issue_taxonomy (tag_name, category, description) VALUES
  ('post_filing_evidence', 'timing', 'Evidence dated after petition filing date'),
  ('letters_conclusory', 'evidence', 'Reference letters lacking detail or specificity'),
  ('subfield_mismatch', 'scope', 'Evidence from different or narrow subfield'),
  ('not_independent', 'evidence', 'Evidence from biased or interested parties'),
  ('no_single_paper_comparison', 'methodology', 'Lack of proper comparison methodology'),
  ('metrics_not_reproducible', 'methodology', 'Research methodology unclear or unreproducible'),
  ('salary_benchmark_weak', 'methodology', 'Salary comparison data inadequate or flawed'),
  ('role_not_critical', 'scope', 'Leadership role lacks critical importance'),
  ('methodology_unclear', 'methodology', 'Research lacking clear methodology'),
  ('unverifiable', 'evidence', 'Claims that cannot be verified'),
  ('undated', 'timing', 'Evidence without clear dates'),
  ('insufficient_corroboration', 'evidence', 'Unsupported claims lacking corroboration')
ON CONFLICT (tag_name) DO NOTHING;

-- Enhanced evidence insights with quote anchoring
CREATE TABLE IF NOT EXISTS criterion_evidence_insights (
  insight_id         BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  criterion_type     TEXT NOT NULL,
  evidence_type      TEXT NOT NULL,
  strength_assessment TEXT CHECK (strength_assessment IN ('strong', 'adequate', 'weak', 'insufficient')),
  specific_deficiency TEXT,
  aao_quote          TEXT,
  quote_page         INTEGER,
  quote_start_char   INTEGER,
  quote_end_char     INTEGER,
  confidence_score   FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
  created_at         TIMESTAMP DEFAULT NOW(),
  UNIQUE(criterion_id, evidence_type)
);

-- Rejection patterns using controlled vocabulary
CREATE TABLE IF NOT EXISTS criterion_rejection_patterns (
  pattern_id         BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  criterion_type     TEXT NOT NULL,
  denial_tag_id      BIGINT REFERENCES denial_issue_taxonomy(tag_id),
  rejection_detail   TEXT,
  severity_level     TEXT CHECK (severity_level IN ('minor', 'moderate', 'major')),
  aao_quote          TEXT,
  quote_page         INTEGER,
  quote_start_char   INTEGER,
  quote_end_char     INTEGER,
  confidence_score   FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
  created_at         TIMESTAMP DEFAULT NOW()
);

-- Success factors for met criteria  
CREATE TABLE IF NOT EXISTS criterion_success_factors (
  factor_id          BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  criterion_type     TEXT NOT NULL,
  success_element    TEXT NOT NULL,
  evidence_strength  TEXT,
  supporting_quote   TEXT,
  quote_page         INTEGER,
  quote_start_char   INTEGER,
  quote_end_char     INTEGER,
  impact_level       TEXT CHECK (impact_level IN ('local', 'national', 'international')),
  confidence_score   FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
  created_at         TIMESTAMP DEFAULT NOW()
);

-- Actionable refile guidance (new requirement from  instructions)
CREATE TABLE IF NOT EXISTS criterion_refile_guidance (
  guidance_id        BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  criterion_type     TEXT NOT NULL,
  guidance_text      TEXT NOT NULL, -- "To satisfy this criterion in a refile, provide..."
  specific_gaps      TEXT[], -- array of specific deficiencies identified
  evidence_needed    TEXT[], -- array of evidence types needed
  created_at         TIMESTAMP DEFAULT NOW(),
  UNIQUE(criterion_id)
);

-- Enhanced linguistic analysis with quote anchoring
CREATE TABLE IF NOT EXISTS aao_linguistic_analysis (
  analysis_id        BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  confidence_level   TEXT CHECK (confidence_level IN ('high', 'medium', 'low')),
  criticism_intensity TEXT CHECK (criticism_intensity IN ('mild', 'moderate', 'severe')),
  burden_language_used BOOLEAN DEFAULT FALSE,
  definitive_phrases JSONB, -- array with page/char anchors: [{"text": "clearly", "page": 3, "start": 100, "end": 107}]
  hedging_phrases    JSONB, -- array with page/char anchors
  reasoning_quotes   JSONB, -- key reasoning statements with anchors
  quantitative_mentions JSONB, -- numbers/percentages with anchors
  created_at         TIMESTAMP DEFAULT NOW(),
  UNIQUE(criterion_id)
);

-- Evidence calendar validation (from instructions)
CREATE TABLE IF NOT EXISTS evidence_calendar_check (
  check_id           BIGSERIAL PRIMARY KEY,
  evidence_id        BIGINT REFERENCES evidence_items(evidence_id) ON DELETE CASCADE,
  filing_date        DATE,
  evidence_date      DATE,
  is_pre_filing      BOOLEAN,
  post_filing_gap_days INTEGER,
  auto_tagged        BOOLEAN DEFAULT FALSE, -- if post_filing tag was auto-added
  created_at         TIMESTAMP DEFAULT NOW(),
  UNIQUE(evidence_id)
);

-- Enhanced quality metrics with quote validation
CREATE TABLE IF NOT EXISTS extraction_quality_metrics (
  metric_id          BIGSERIAL PRIMARY KEY,
  criterion_id       BIGINT REFERENCES claimed_criteria(criterion_id) ON DELETE CASCADE,
  extraction_version TEXT DEFAULT '2.0', -- updated version for  compliance
  quality_score      FLOAT CHECK (quality_score >= 0.0 AND quality_score <= 1.0),
  quote_accuracy     FLOAT CHECK (quote_accuracy >= 0.0 AND quote_accuracy <= 1.0),
  quote_anchoring_score FLOAT CHECK (quote_anchoring_score >= 0.0 AND quote_anchoring_score <= 1.0),
  controlled_vocab_compliance FLOAT CHECK (controlled_vocab_compliance >= 0.0 AND controlled_vocab_compliance <= 1.0),
  confidence_distribution JSONB,
  issues_found       TEXT[],
  processing_time_ms INTEGER,
  llm_model_used     TEXT,
  created_at         TIMESTAMP DEFAULT NOW(),
  UNIQUE(criterion_id, extraction_version)
);

-- Batch processing tracking
CREATE TABLE IF NOT EXISTS criterion_extraction_batches (
  batch_id           BIGSERIAL PRIMARY KEY,
  criterion_type     TEXT NOT NULL,
  batch_size         INTEGER NOT NULL,
  processed_count    INTEGER DEFAULT 0,
  success_count      INTEGER DEFAULT 0,
  error_count        INTEGER DEFAULT 0,
  started_at         TIMESTAMP DEFAULT NOW(),
  completed_at       TIMESTAMP,
  error_log_path     TEXT, -- path to error log file
  status             TEXT DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed', 'paused'))
);

-- =============================================================================
-- INDEXES FOR PERFORMANCE
-- =============================================================================

-- Evidence insights indexes
CREATE INDEX IF NOT EXISTS idx_evidence_insights_criterion_id ON criterion_evidence_insights(criterion_id);
CREATE INDEX IF NOT EXISTS idx_evidence_insights_type ON criterion_evidence_insights(criterion_type);
CREATE INDEX IF NOT EXISTS idx_evidence_insights_evidence_type ON criterion_evidence_insights(evidence_type);
CREATE INDEX IF NOT EXISTS idx_evidence_insights_strength ON criterion_evidence_insights(strength_assessment);

-- Rejection patterns indexes
CREATE INDEX IF NOT EXISTS idx_rejection_patterns_criterion_id ON criterion_rejection_patterns(criterion_id);
CREATE INDEX IF NOT EXISTS idx_rejection_patterns_type ON criterion_rejection_patterns(criterion_type);
CREATE INDEX IF NOT EXISTS idx_rejection_patterns_severity ON criterion_rejection_patterns(severity_level);

-- Success factors indexes
CREATE INDEX IF NOT EXISTS idx_success_factors_criterion_id ON criterion_success_factors(criterion_id);
CREATE INDEX IF NOT EXISTS idx_success_factors_type ON criterion_success_factors(criterion_type);
CREATE INDEX IF NOT EXISTS idx_success_factors_element ON criterion_success_factors(success_element);
CREATE INDEX IF NOT EXISTS idx_success_factors_impact ON criterion_success_factors(impact_level);

-- Linguistic analysis indexes
CREATE INDEX IF NOT EXISTS idx_linguistic_analysis_criterion_id ON aao_linguistic_analysis(criterion_id);
CREATE INDEX IF NOT EXISTS idx_linguistic_analysis_confidence ON aao_linguistic_analysis(confidence_level);
CREATE INDEX IF NOT EXISTS idx_linguistic_analysis_criticism ON aao_linguistic_analysis(criticism_intensity);

-- Quality metrics indexes
CREATE INDEX IF NOT EXISTS idx_quality_metrics_criterion_id ON extraction_quality_metrics(criterion_id);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_score ON extraction_quality_metrics(quality_score);
CREATE INDEX IF NOT EXISTS idx_quality_metrics_version ON extraction_quality_metrics(extraction_version);

-- Batch tracking indexes
CREATE INDEX IF NOT EXISTS idx_extraction_batches_type ON criterion_extraction_batches(criterion_type);
CREATE INDEX IF NOT EXISTS idx_extraction_batches_status ON criterion_extraction_batches(status);
CREATE INDEX IF NOT EXISTS idx_extraction_batches_started ON criterion_extraction_batches(started_at);
