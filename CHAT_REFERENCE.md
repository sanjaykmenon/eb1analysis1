# 🎯 AAO ETL Database Chat - Quick Reference

## Start the Chat
```bash
uv run python chat_with_db.py
```

## Example Queries

### Basic Searches
```
Find the last 10 denied cases
Show me 5 approved cases from 2023
Search for cases with Computer Science field
Find decisions between 2020 and 2023
```

### Specific Criteria Analysis
```
What are common ORIGINAL_CONTRIBUTION denial reasons?
Show me AWARD criterion success patterns
Analyze MEMBERSHIP criterion denials
What makes PUBLISHED_MATERIAL criterion succeed?
```

### Field-Specific Searches
```
Find denied cases in software engineering
Show approved cases in machine learning
Search for cases about artificial intelligence research
Find decisions in biotechnology field
```

### Database Exploration
```
What's in the database schema?
How many decisions are in the database?
What fields of endeavor are most common?
Show me the structure of the evidence table
```

### Complex Queries
```
Find denied cases where ORIGINAL_CONTRIBUTION was not met
Show me cases with high citation counts that were approved
What evidence types are most commonly excluded?
Compare success rates between AWARD and MEMBERSHIP criteria
```

## Available MCP Tools

The chat uses these tools automatically:

1. **describe_schema** - Shows database structure
2. **search_decisions** - Searches cases by:
   - Text (field of endeavor, summary, rationale)
   - Outcome (approved, denied, etc.)
   - Date range
   - Criterion type
   - Limit (max results)

3. **get_decision_details** - Gets full details for a specific case
4. **summarize_criterion_insights** - Analyzes patterns for specific criteria

## Tips for Best Results

✅ **Do:**
- Be specific about what you want to find
- Use criterion codes (AWARD, MEMBERSHIP, etc.) when asking about criteria
- Specify result limits (e.g., "find 5 cases" instead of "find cases")
- Ask follow-up questions to dig deeper

❌ **Avoid:**
- Very broad queries without filters ("show me everything")
- Asking for too many results at once (stick to 5-20)
- Using unclear field names (say "field of endeavor" not "field")

## Query Patterns That Work Well

### Pattern 1: Filtered Search
```
Find [NUMBER] [OUTCOME] cases [with FILTERS]
```
Examples:
- Find 10 denied cases in computer science
- Find 5 approved cases from 2022

### Pattern 2: Criterion Analysis  
```
What are common [CRITERION] [denial/success] [patterns/reasons]?
```
Examples:
- What are common AWARD denial reasons?
- Show me ORIGINAL_CONTRIBUTION success patterns

### Pattern 3: Evidence Analysis
```
[Question] about evidence [TYPE/PATTERN]
```
Examples:
- What evidence types are most accepted?
- Why do letters of recommendation get excluded?

### Pattern 4: Semantic Search
```
Search for decisions about [TOPIC]
```
Examples:
- Search for decisions about peer review
- Find cases mentioning citation metrics

## Troubleshooting

### "None" Response
- Tool was called but returned no results
- Try broader search terms or different filters

### "Input validation error"
- Together AI has some function calling limitations
- Try rephrasing the question more simply
- Use more specific parameters

### Slow Response
- Complex queries take time
- Database searches with text filters are slower
- Be patient, especially for semantic searches

## Environment Variables

```bash
# Required
export DATABASE_URL="postgresql+psycopg://user:pass@host/db"
export TOGETHER_API_KEY="your-key"  # or OPENAI_API_KEY, ANTHROPIC_API_KEY

# Optional
export DEBUG=1  # Show full error tracebacks
```

## Keyboard Shortcuts

- `Ctrl+C` - Cancel current operation (press twice to exit)
- `exit`, `quit`, `bye` - Exit the chat
- `Enter` on empty line - Skip (will re-prompt)

## Getting More Help

- See `QUICK_START_MCP.md` for setup instructions
- See `MCP_SETUP.md` for technical details
- Check `README.md` for full project documentation
