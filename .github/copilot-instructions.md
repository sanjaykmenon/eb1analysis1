# AAO ETL Copilot Instructions

## Project Overview
AAO ETL is a specialized legal data processing pipeline that extracts structured insights from AAO (Administrative Appeals Office) EB-1A immigration decision PDFs using AI. The system combines PDF text extraction, regex enrichment, OpenAI LLM analysis, and PostgreSQL storage with pgvector for semantic search.

## Architecture & Data Flow

### Core Pipeline (`src/aao_etl/pipeline.py`)
1. **PDF Extraction** → **Regex Enrichment** → **LLM Structured Extraction** → **Database Storage**
2. Two modes: `--dry-run` (regex only) and full LLM extraction
3. Processing always requires OpenAI API key unless in dry-run mode

### Key Components
- **CLI** (`cli.py`): Typer-based with parallel processing using ThreadPoolExecutor
- **Models** (`models.py`): Pydantic models for 10 EB-1A criteria with strict enums (`Criterion`, `Finding`, `EvidenceType`)
- **Database** (`db.py`): Complex schema with ~15 tables for legal data relationships
- **LLM Integration** (`llm.py`): OpenAI + Instructor for structured extraction with criterion-specific prompts

### Database Schema Patterns
- **Core**: `decisions` table with metadata, linked to `claimed_criteria`, `evidence_items`, `quotes`
- **Analytics**: Specialized tables for insights (`criterion_evidence_insights`, `aao_linguistic_analysis`)
- **pgvector**: Semantic search on decision summaries via `summary_embedding` column

## Development Workflows

### Essential Commands
```bash
# Database initialization (required first step)
uv run aao init-db

# Processing PDFs (always test with --dry-run first)
uv run aao process ./pdfs --dry-run
uv run aao process ./pdfs --max-workers 4

# Parallel insights analysis (for large datasets)
uv run aao insights-parallel --max-workers 10
```

### Environment Setup
- Uses `uv` package manager (preferred) with `pyproject.toml`
- Required env vars: `DATABASE_URL`, `OPENAI_API_KEY`
- Optional: `GPT_MODEL` (defaults to `gpt-4.1-mini`)
- PostgreSQL with pgvector extension required
- **Important**: Uses `mcp>=1.0` (not `modelcontextprotocol`) to avoid fireworks-ai dependency conflicts

## Project-Specific Patterns

### Error Handling & Resilience
- **Thread-local database connections** for parallel processing (`get_thread_engine()`)
- **Resume capability**: `--resume` flag skips already-processed PDFs
- **Error logging**: Failures logged to `analysis_errors/` directory
- **Rate limiting**: Built into OpenAI client for API compliance

### Legal Domain Specifics
- **10 EB-1A Criteria**: Hardcoded enum in `models.py` following 8 CFR 204.5(h)(3)
- **Quote Anchoring**: Every finding includes source text + page numbers for legal accuracy
- **Timeline Validation**: Pre-filing vs post-filing evidence detection critical for immigration law
- **Citation Extraction**: CFR, BIA, federal case citations parsed and stored

### Code Conventions
- **Pydantic Models**: Strict validation for legal data integrity
- **Enum-First Design**: Outcomes, findings, evidence types use strict enums
- **SQL-Heavy**: Raw SQL for complex analytics queries, not ORM
- **Thread Safety**: Database operations designed for parallel processing

## MCP Server Integration
- Exposes database via Model Context Protocol (`mcp_server.py`)
- Uses `FastMCP` from `mcp.server.fastmcp` (v1.18+)
- **Schema Overview** includes all table relationships for AI agents
- Enables semantic search and complex legal data queries
- **Troubleshooting**: If seeing protobuf errors, ensure using `mcp>=1.0` not `modelcontextprotocol` (which pulls in fireworks-ai)

## Testing & Debugging
- **Dry Run Mode**: Always test regex extraction before LLM processing
- **Progress Tracking**: Batch processing with detailed progress reporting
- **Cost Awareness**: Model choice affects processing cost ($0.10-0.50/decision)
- **Validation**: LLM extraction failures raise exceptions, don't silently continue

## Key Files for Understanding Context
- `src/aao_etl/models.py`: Core legal data structures and enums
- `src/aao_etl/sql/init_db.sql`: Complete database schema
- `src/aao_etl/llm.py`: LLM prompts and criterion-specific extraction logic
- `README.md`: Comprehensive setup and usage documentation
- `QUICK_START_MCP.md`: Guide for using LLMs to query the database via MCP
- `MCP_SETUP.md`: Detailed MCP server setup and troubleshooting