# MCP Server - Case Number Display Updates

## Overview
Enhanced the MCP server to prominently display `case_number` and `decision_id` in all query results, making it easy to verify LLM responses against the actual database.

## Changes Made

### 1. `search_decisions` Tool Enhancement
**Location**: `src/aao_etl/mcp_server.py` lines ~165-180

**Before:**
```
Found decisions:
• #123 case TBD (2023-01-15) outcome=denied
  ↳ Summary text...
```

**After:**
```
Found 3 decision(s):

📄 Case: WAC-12-345-67890 | ID: 123 | Date: 2023-01-15 | Outcome: denied
   Field: Computer Science
   Summary: Summary text...

📄 Case: WAC-12-345-67891 | ID: 124 | Date: 2023-02-20 | Outcome: approved
   Field: Electrical Engineering
   Summary: Summary text...
```

**Key Improvements:**
- ✅ Case number displayed first (most important for DB lookup)
- ✅ Decision ID clearly labeled
- ✅ Better formatting with blank lines between results
- ✅ Field of endeavor on separate line
- ✅ Count of results at top

### 2. `get_decision_details` Tool Enhancement
**Location**: `src/aao_etl/mcp_server.py` lines ~270-280

**Before:**
```
Decision #123 | case WAC-12-345-67890
Outcome: denied | Decision date: 2023-01-15
Field of endeavor: Computer Science
```

**After:**
```
======================================================================
📄 CASE NUMBER: WAC-12-345-67890
   Decision ID: 123
======================================================================
Outcome: denied
Decision date: 2023-01-15 | Filing date: 2022-06-30
Field of endeavor: Computer Science
```

**Key Improvements:**
- ✅ Case number prominently displayed at top with visual separator
- ✅ Decision ID clearly labeled below case number
- ✅ Filing date added for timeline context
- ✅ Visual separators make results easier to scan

### 3. `summarize_criterion_insights` Tool Enhancement
**Location**: `src/aao_etl/mcp_server.py` lines ~330-380

**Before (Denial Patterns):**
```
Top denial patterns:
• post_filing_evidence (category=timing, occurrences=5)
  ↳ Evidence dated after petition filing date
```

**After (Denial Patterns):**
```
======================================================================
🚫 TOP DENIAL PATTERNS
======================================================================
• post_filing_evidence (category=timing, occurrences=5)
  Case: WAC-12-345-67890 (ID: 123)
  Detail: Evidence dated after petition filing date

• letters_conclusory (category=evidence, occurrences=3)
  Case: WAC-12-345-67891 (ID: 124)
  Detail: Reference letters lacking detail or specificity
```

**Before (Success Factors):**
```
Success factors:
• Strong citation metrics (strength=adequate, occurrences=10)
```

**After (Success Factors):**
```
======================================================================
✅ SUCCESS FACTORS
======================================================================
• Strong citation metrics (strength=adequate, occurrences=10)
  Case: WAC-12-345-67892 (ID: 125)

• Independent expert letters (strength=strong, occurrences=8)
  Case: WAC-12-345-67893 (ID: 126)
```

**Key Improvements:**
- ✅ Each insight shows specific case_number and decision_id
- ✅ Enables verification of LLM claims against actual DB records
- ✅ Visual section headers with emoji for clarity
- ✅ Blank lines between entries for readability
- ✅ SQL joins added to fetch case numbers from related tables

## Database Verification Workflow

### Step 1: Get Results from Chat
```bash
uv run python chat_with_db.py
```

Query: "Find denied cases about machine learning"

Response:
```
Found 2 decision(s):

📄 Case: WAC-12-345-67890 | ID: 123 | Date: 2023-01-15 | Outcome: denied
   Field: Machine Learning
   Summary: Petition denied due to insufficient evidence...

📄 Case: WAC-12-345-67891 | ID: 124 | Date: 2023-03-20 | Outcome: denied
   Field: Artificial Intelligence  
   Summary: AAO found evidence lacked independent corroboration...
```

### Step 2: Verify in Database
```bash
# Using psql
psql $DATABASE_URL

-- Verify case exists
SELECT case_number, decision_id, outcome, field_of_endeavor, decision_date
FROM decisions
WHERE case_number = 'WAC-12-345-67890';

-- Get full details
SELECT * FROM decisions WHERE decision_id = 123;

-- Check criteria findings
SELECT criterion, director_finding, aao_finding
FROM claimed_criteria
WHERE decision_id = 123;

-- Check evidence items
SELECT e_type, title, accepted_by_uscis
FROM evidence_items
WHERE decision_id = 123;
```

### Step 3: Verify Insights
Query: "What are common ORIGINAL_CONTRIBUTION denial patterns?"

Response:
```
======================================================================
🚫 TOP DENIAL PATTERNS
======================================================================
• letters_conclusory (category=evidence, occurrences=5)
  Case: WAC-12-345-67890 (ID: 123)
  Detail: Reference letters lacking specific examples of original contribution
```

Verify in DB:
```sql
-- Check the pattern exists
SELECT r.rejection_detail, t.tag_name, t.category,
       d.case_number, d.decision_id
FROM criterion_rejection_patterns r
LEFT JOIN denial_issue_taxonomy t ON r.denial_tag_id = t.tag_id
LEFT JOIN claimed_criteria c ON r.criterion_id = c.criterion_id
LEFT JOIN decisions d ON c.decision_id = d.decision_id
WHERE r.criterion_type = 'ORIGINAL_CONTRIBUTION'
  AND t.tag_name = 'letters_conclusory';
```

## SQL Schema Updates

### New Joins in `summarize_criterion_insights`

#### Denial Patterns Query
```sql
SELECT t.tag_name, t.category, 
       COALESCE(r.rejection_detail,'') AS detail,
       d.case_number,                    -- NEW: Case number for verification
       d.decision_id,                    -- NEW: Decision ID for verification
       COUNT(*) OVER (PARTITION BY t.tag_name, t.category) AS occurrences
FROM criterion_rejection_patterns r
LEFT JOIN denial_issue_taxonomy t ON r.denial_tag_id = t.tag_id
LEFT JOIN claimed_criteria c ON r.criterion_id = c.criterion_id  -- NEW: Join to get decision_id
LEFT JOIN decisions d ON c.decision_id = d.decision_id          -- NEW: Join to get case_number
WHERE r.criterion_type = :criterion
ORDER BY occurrences DESC, t.tag_name
LIMIT :limit
```

#### Success Factors Query
```sql
SELECT s.success_element, 
       COALESCE(s.evidence_strength,'n/a') AS strength,
       d.case_number,                    -- NEW: Case number for verification
       d.decision_id,                    -- NEW: Decision ID for verification
       COUNT(*) OVER (PARTITION BY s.success_element) AS occurrences
FROM criterion_success_factors s
LEFT JOIN claimed_criteria c ON s.criterion_id = c.criterion_id  -- NEW: Join to get decision_id
LEFT JOIN decisions d ON c.decision_id = d.decision_id          -- NEW: Join to get case_number
WHERE s.criterion_type = :criterion
ORDER BY occurrences DESC, s.success_element
LIMIT :limit
```

## Benefits

### 1. Hallucination Detection
- Every LLM response includes case_number and decision_id
- Quick spot-check: Copy case number → query database → verify facts
- Example: LLM says "10 approved cases in engineering" → Verify each case number exists

### 2. Data Integrity
- Ensures MCP server returns real data from actual DB rows
- No synthetic or made-up case numbers
- Easy to trace any response back to source PDF

### 3. Debugging & Development
- When LLM gives unexpected answer, check the case_number in DB
- Verify if issue is with:
  - Data extraction (PDF → DB)
  - Query logic (SQL)
  - LLM interpretation
  - Missing/incomplete data

### 4. Legal Use Case
- Case numbers are official identifiers (e.g., "WAC-12-345-67890")
- Can reference specific AAO decisions in legal briefs
- Traceable to original PDF documents

## Testing

### Test 1: Empty Database
```bash
uv run python chat_with_db.py
> "Find all denied cases"
```

**Expected**: "No decisions matched the provided filters."  
**Actual**: LLM will see empty result set from MCP tool

### Test 2: With Real Data
```bash
# First populate database
uv run aao process ./pdfs --dry-run  # Test extraction
uv run aao process ./pdfs             # Full LLM extraction

# Then query
uv run python chat_with_db.py
> "Find 5 denied cases"
```

**Expected**: List of 5 cases with real case_numbers and decision_ids
```
📄 Case: WAC-21-123-45678 | ID: 1 | Date: 2021-05-15 | Outcome: denied
   Field: Computer Science
```

**Verify**: 
```sql
SELECT case_number FROM decisions WHERE outcome = 'denied' LIMIT 5;
```

### Test 3: Criterion Insights
```bash
uv run python chat_with_db.py
> "What are common AWARD denial patterns?"
```

**Expected**: Each pattern shows case_number
```
🚫 TOP DENIAL PATTERNS
• insufficient_national_significance (category=scope, occurrences=3)
  Case: WAC-21-123-45678 (ID: 1)
  Detail: Award limited to regional recognition
```

**Verify**:
```sql
SELECT d.case_number, r.rejection_detail, t.tag_name
FROM criterion_rejection_patterns r
JOIN denial_issue_taxonomy t ON r.denial_tag_id = t.tag_id
JOIN claimed_criteria c ON r.criterion_id = c.criterion_id
JOIN decisions d ON c.decision_id = d.decision_id
WHERE r.criterion_type = 'AWARD';
```

## Files Modified

1. **`src/aao_etl/mcp_server.py`**
   - Line ~165-180: `search_decisions` output formatting
   - Line ~270-280: `get_decision_details` header formatting
   - Line ~330-370: `summarize_criterion_insights` denial patterns with joins
   - Line ~370-390: `summarize_criterion_insights` success factors with joins

## Backward Compatibility

✅ **No breaking changes**
- All existing queries still work
- Only output formatting changed
- SQL queries enhanced with additional joins (no performance impact for small datasets)
- Tool parameters unchanged

## Performance Considerations

- Added LEFT JOINs to fetch case_numbers in insights queries
- For large datasets (10,000+ decisions), consider adding indexes:

```sql
-- Recommended indexes for insights queries
CREATE INDEX IF NOT EXISTS idx_criterion_patterns_type 
  ON criterion_rejection_patterns(criterion_type);
  
CREATE INDEX IF NOT EXISTS idx_success_factors_type 
  ON criterion_success_factors(criterion_type);
  
CREATE INDEX IF NOT EXISTS idx_claimed_criteria_decision 
  ON claimed_criteria(decision_id);
```

## Next Steps

1. **Populate Database**: Run `uv run aao process ./pdfs` with real AAO decision PDFs
2. **Test Queries**: Use chat interface to query and verify case numbers appear
3. **Spot Check**: For any interesting result, verify case_number in database
4. **Report Issues**: If case numbers don't match or are missing, check:
   - PDF extraction (`aao_etl/pdf.py`)
   - Regex enrichment (`aao_etl/enrich.py`)
   - Database insertion (`aao_etl/pipeline.py`)

## Example Session

```bash
$ uv run python chat_with_db.py
Choose provider [openai]: 
Choose model [1]: 

👤 You: Find 3 denied ORIGINAL_CONTRIBUTION cases

🔧 Calling search_decisions(...)

🤖 Assistant:
Found 3 decision(s):

📄 Case: WAC-21-123-45678 | ID: 15 | Date: 2021-08-10 | Outcome: denied
   Field: Computer Science
   Summary: AAO found letters lacked specific examples...

📄 Case: WAC-21-234-56789 | ID: 23 | Date: 2021-09-15 | Outcome: denied
   Field: Biomedical Engineering
   Summary: Evidence did not establish original contributions...

📄 Case: WAC-21-345-67890 | ID: 31 | Date: 2021-11-20 | Outcome: denied
   Field: Physics
   Summary: Citations insufficient to demonstrate impact...

👤 You: exit
```

**Verification:**
```sql
SELECT case_number, decision_id, outcome, field_of_endeavor
FROM decisions
WHERE decision_id IN (15, 23, 31);
```

Should return the exact case numbers shown by the LLM.

---

## Troubleshooting

### LLM Returns "NO_CASE_NUMBER"
**Cause**: PDF extraction didn't capture case number from document  
**Solution**: Check regex patterns in `src/aao_etl/enrich.py`

### Decision IDs Don't Match
**Cause**: Database was recreated, IDs changed  
**Solution**: Use `case_number` for lookup (stable identifier)

### Empty Results
**Cause**: No data in database  
**Solution**: Run `uv run aao process ./pdfs` to populate

### Hallucinated Case Numbers
**Cause**: LLM making up data when MCP returns empty  
**Solution**: Check actual MCP tool output in terminal logs (shows real SQL results)

---

**Last Updated**: October 18, 2025  
**Version**: 2.0 (Case Number Verification Update)
