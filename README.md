# AAO ETL: EB-1A Decision Analysis Platform

> 🏛️ **Extract, Transform, and Analyze** U.S. Administrative Appeals Office (AAO) EB-1A petition decisions using AI-powered text extraction and PostgreSQL analytics.

## 📋 Table of Contents

- [What This Project Does](#what-this-project-does)
- [Key Features](#key-features)
- [Prerequisites](#prerequisites)
- [Quick Start Guide](#quick-start-guide)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Commands Reference](#commands-reference)
- [Database Schema](#database-schema)
- [AI-Powered Insights](#ai-powered-insights)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## 🎯 What This Project Does

This project automatically processes AAO EB-1A (Extraordinary Ability) petition decision PDFs and extracts structured legal insights into a PostgreSQL database. It uses OpenAI's GPT models to understand complex legal language and extract:

- **Decision outcomes** and case metadata
- **Criterion-by-criterion analysis** (all 10 EB-1A criteria)
- **Evidence assessment** (what worked, what didn't, and why)
- **AAO reasoning patterns** and linguistic analysis
- **Success/failure factors** for each criterion type
- **Legal authorities** cited in decisions

Perfect for immigration attorneys, researchers, and anyone analyzing EB-1A petition patterns.

## ✨ Key Features

### 🤖 AI-Powered Extraction
- **Multi-pass LLM analysis** using OpenAI GPT models
- **Structured data extraction** with validation
- **Quote anchoring** - every finding includes source text and page numbers
- **Parallel processing** for faster analysis

### 📊 Comprehensive Analytics
- **Success rate analysis** by criterion and evidence type
- **Rejection pattern identification** across thousands of cases
- **Linguistic analysis** of AAO decision language
- **Timeline validation** (pre/post-filing evidence detection)

### 🗄️ Robust Database
- **PostgreSQL** with pgvector for semantic search
- **Full-text search** capabilities
- **Structured schema** for complex legal data
- **Faceted search** API ready

### ⚡ Production Ready
- **Parallel processing** with configurable workers
- **Rate limiting** for API compliance
- **Error handling** and retry logic
- **Resume capability** for large batches

## 🛠️ Prerequisites

### System Requirements
- **Python 3.10+** (required)
- **PostgreSQL 12+** with pgvector extension
- **OpenAI API key** (for AI extraction features)
- **4GB+ RAM** (for processing large PDFs)

### Installation Dependencies
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager (recommended)
- OR standard Python pip/venv

## 🚀 Quick Start Guide

### Step 1: Install uv (Recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or use pip
pip install uv
```

### Step 2: Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url> aao-etl
cd aao-etl

# Create and sync Python environment
uv venv
uv sync

# Alternative with pip/venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### Step 3: Database Setup

```bash
# Install PostgreSQL (if not already installed)
# macOS with Homebrew:
brew install postgresql pgvector

# Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL and create database
createdb aao_decisions
```

### Step 4: Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
```

**Required `.env` settings:**
```bash
# Database connection
DATABASE_URL=postgresql+psycopg://username:password@localhost:5432/aao_decisions

# OpenAI API (required for AI features)
OPENAI_API_KEY=sk-your-api-key-here

# Model selection (optional)
GPT_MODEL=gpt-4o-mini  # or gpt-4, o1-mini, etc.
```

### Step 5: Initialize Database

```bash
# Create tables and enable extensions
uv run aao init-db
```

This creates ~15 tables including:
- `decisions` - Case metadata and outcomes
- `claimed_criteria` - Criterion-by-criterion analysis
- `evidence_items` - Evidence assessment
- `authorities` - Legal citations
- Plus analytics tables for insights

### Step 6: Process Your First PDFs

```bash
# Test with regex-only extraction (no API calls)
uv run aao process ./your-pdf-folder --dry-run

# Full AI-powered extraction
uv run aao process ./your-pdf-folder

# Resume interrupted processing
uv run aao process ./your-pdf-folder --resume

# Parallel processing (faster)
uv run aao process ./your-pdf-folder --max-workers 8
```

## ⚙️ Configuration

### Database Configuration

The project supports any PostgreSQL database with pgvector. Configure via `DATABASE_URL`:

```bash
# Local PostgreSQL
DATABASE_URL=postgresql+psycopg://postgres:password@localhost:5432/aao

# Cloud PostgreSQL (AWS RDS, Google Cloud SQL, etc.)
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/dbname

# Connection pooling for production
DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db?pool_size=20
```

### OpenAI Configuration

```bash
# Model selection (cost vs. quality tradeoff)
GPT_MODEL=gpt-4o-mini      # Fastest, cheapest
GPT_MODEL=gpt-4o           # Balanced
GPT_MODEL=o1-mini          # Highest quality, most expensive

# API key from OpenAI dashboard
OPENAI_API_KEY=sk-proj-...
```

### Performance Tuning

```bash
# Processing optimization
--max-workers 8            # Parallel PDF processing
--batch-size 50            # Database batch size
--requests-per-minute 3000 # API rate limiting
```

## 📖 Usage Examples

### Basic PDF Processing

```bash
# Process a single folder
uv run aao process ./aao_pdfs

# Process with specific settings
uv run aao process ./pdfs \
  --max-workers 4 \
  --batch-size 25 \
  --resume
```

### Advanced AI Insights Extraction

```bash
# Extract detailed insights for all criteria
uv run aao insights

# Process specific criterion type
uv run aao insights --criterion-type AWARD

# Parallel insights processing (much faster)
uv run aao insights-parallel \
  --max-workers 10 \
  --criterion-workers 3 \
  --requests-per-minute 3000
```

### Analytics and Statistics

```bash
# View success rates and patterns
uv run aao insights-stats

# Filter by criterion type
uv run aao insights-stats --criterion-type MEMBERSHIP

# Export as JSON or CSV
uv run aao insights-stats --format json > results.json
uv run aao insights-stats --format csv > results.csv
```

### Evidence Timeline Validation

```bash
# Validate all evidence dates against filing dates
uv run aao validate-evidence

# Check specific decision
uv run aao validate-evidence --decision-id 1234
```

## 📚 Commands Reference

### Core Commands

| Command | Purpose | Example |
|---------|---------|---------|
| `init-db` | Initialize database schema | `uv run aao init-db` |
| `process` | Extract data from PDFs | `uv run aao process ./pdfs` |
| `insights` | Generate AI insights | `uv run aao insights` |
| `insights-parallel` | Parallel insights processing | `uv run aao insights-parallel` |
| `insights-stats` | View analytics | `uv run aao insights-stats` |
| `validate-evidence` | Check evidence dates | `uv run aao validate-evidence` |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-workers N` | Parallel workers | 4 |
| `--batch-size N` | Database batch size | 100 |
| `--dry-run` | Skip AI, regex only | False |
| `--resume` | Skip processed files | False |
| `--requests-per-minute N` | API rate limit | 3000 |

## 🗃️ Database Schema

### Core Tables

- **`decisions`** - Case metadata, outcomes, dates
- **`claimed_criteria`** - AAO findings per criterion
- **`evidence_items`** - Evidence analysis and assessment
- **`authorities`** - Legal citations and precedents
- **`quotes`** - Extracted text with page anchors

### Analytics Tables

- **`criterion_evidence_insights`** - Evidence effectiveness analysis
- **`criterion_rejection_patterns`** - Common rejection reasons
- **`criterion_success_factors`** - Success pattern analysis
- **`aao_linguistic_analysis`** - Language pattern analysis

### Sample Queries

```sql
-- Success rates by criterion
SELECT criterion, 
       AVG(CASE WHEN aao_finding = 'met' THEN 1.0 ELSE 0.0 END) as success_rate,
       COUNT(*) as total_cases
FROM claimed_criteria 
GROUP BY criterion 
ORDER BY success_rate DESC;

-- Most common rejection reasons
SELECT rejection_category, COUNT(*) as frequency
FROM criterion_rejection_patterns 
GROUP BY rejection_category 
ORDER BY frequency DESC;
```

## 🧠 AI-Powered Insights

### What Gets Analyzed

For each criterion in every decision, the AI extracts:

1. **Evidence Types** - Letters, publications, awards, etc.
2. **AAO Assessment** - Strength evaluation and specific criticisms
3. **Rejection Patterns** - Why criteria failed (methodology, independence, etc.)
4. **Success Factors** - What made successful criteria work
5. **Linguistic Patterns** - AAO confidence level and reasoning style
6. **Quote Anchoring** - Every finding tied to specific text and page

### Example Insights Output

```json
{
  "evidence_analysis": [
    {
      "evidence_type": "letter",
      "aao_assessment": "lacks independence and specificity",
      "strength_indicator": "weak",
      "supporting_quote": "The letters are conclusory and fail to explain...",
      "confidence": "high"
    }
  ],
  "rejection_patterns": [
    {
      "rejection_category": "insufficient_independence",
      "specific_issue": "Letters from collaborators and colleagues",
      "aao_language": "not independent evidence of recognition",
      "severity": "major"
    }
  ]
}
```

## 🔧 Troubleshooting

### Common Issues

#### Database Connection Errors
```bash
# Check PostgreSQL is running
brew services start postgresql  # macOS
sudo systemctl start postgresql  # Linux

# Test connection
psql "postgresql://localhost:5432/aao_decisions"
```

#### OpenAI API Issues
```bash
# Verify API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Check rate limits in processing logs
```

#### Memory Issues with Large PDFs
```bash
# Reduce parallel workers
uv run aao process ./pdfs --max-workers 2

# Process in smaller batches
uv run aao process ./pdfs --batch-size 10
```

### Performance Optimization

#### For Large Datasets (1000+ PDFs)
```bash
# Use parallel processing with rate limiting
uv run aao process ./pdfs \
  --max-workers 8 \
  --requests-per-minute 2000 \
  --batch-size 50
```

#### For Insights Processing
```bash
# Parallel insights for speed
uv run aao insights-parallel \
  --max-workers 10 \
  --criterion-workers 3 \
  --batch-workers 2
```

### Debug Mode

```bash
# Enable verbose logging
export LOG_LEVEL=DEBUG
uv run aao process ./pdfs

# Check detailed error logs
cat analysis_errors/*.log
```

## 📊 Expected Results

After processing AAO decisions, you can expect:

- **Extraction Rate**: ~90-95% successful PDF processing
- **Processing Speed**: 1-3 decisions per minute (depending on model)
- **Data Quality**: Every finding backed by quotes and page numbers
- **Cost**: ~$0.10-0.50 per decision (depending on model choice)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with tests
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
uv sync --dev

``

## 📄 License

MIT License - see LICENSE file for details.

## 🙋‍♀️ Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Documentation**: See `/docs` folder for detailed guides
- **Examples**: Check `/examples` for sample usage

---

> 💡 **Pro Tip**: Start with `--dry-run` to test your setup before using API credits, then use `insights-parallel` for fastest processing of large datasets.
