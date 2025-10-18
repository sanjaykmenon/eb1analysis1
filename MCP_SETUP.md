# MCP Server Setup Guide

This document explains how to use the AAO ETL MCP server to query your database using AI assistants.

## What is MCP?

The Model Context Protocol (MCP) allows AI assistants like Claude to connect directly to your database and query it intelligently. The AAO ETL MCP server exposes tools for:

- **Querying decisions** by case number, outcome, date range, field of endeavor
- **Searching criteria** with semantic search on rationales
- **Analyzing patterns** across denial reasons, success factors, and evidence types
- **Getting insights** on specific EB-1A criteria (AWARD, MEMBERSHIP, etc.)

## Setup Instructions

### Option 1: Claude Desktop (Recommended)

1. **Install Claude Desktop** from https://claude.ai/download

2. **Configure the MCP Server**:
   
   Copy the example config and update with your database credentials:
   
   ```bash
   cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```

3. **Edit the config** to set your actual DATABASE_URL:
   
   ```bash
   nano ~/Library/Application\ Support/Claude/claude_desktop_config.json
   ```
   
   Update the DATABASE_URL to match your PostgreSQL connection string:
   ```json
   {
     "mcpServers": {
       "aao-etl": {
         "command": "uv",
         "args": [
           "--directory",
           "/Users/sanjaykrishna/Documents/github/eb1analysis1",
           "run",
           "aao",
           "mcp-server"
         ],
         "env": {
           "DATABASE_URL": "postgresql+psycopg://user:pass@localhost:5432/aao_decisions"
         }
       }
     }
   }
   ```

4. **Restart Claude Desktop**

5. **Verify Connection**: Look for a 🔌 icon in Claude Desktop showing "aao-etl" is connected

### Option 2: Python MCP Client (Programmatic Access)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Connect to the MCP server
server_params = StdioServerParameters(
    command="uv",
    args=["--directory", "/path/to/eb1analysis1", "run", "aao", "mcp-server"],
    env={"DATABASE_URL": "postgresql+psycopg://..."}
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        # Initialize
        await session.initialize()
        
        # List available tools
        tools = await session.list_tools()
        print("Available tools:", [t.name for t in tools.tools])
        
        # Call a tool
        result = await session.call_tool(
            "query_decisions",
            arguments={"outcome": "denied", "limit": 5}
        )
        print(result.content)
```

### Option 3: Cline/Continue.dev VS Code Extension

If using Cline or Continue.dev in VS Code:

1. Open settings for the extension
2. Add MCP server configuration:
   ```json
   {
     "mcpServers": {
       "aao-etl": {
         "command": "uv",
         "args": ["--directory", "/path/to/eb1analysis1", "run", "aao", "mcp-server"],
         "env": {"DATABASE_URL": "postgresql+psycopg://..."}
       }
     }
   }
   ```

## Available MCP Tools

Once connected, you can ask the AI to:

### 1. Query Decisions
```
"Show me the last 10 denied EB-1A cases"
"Find all cases in the field of Computer Science decided after 2020"
"Search for cases mentioning 'machine learning' in the summary"
```

### 2. Analyze Criteria
```
"What are the common denial patterns for the ORIGINAL_CONTRIBUTION criterion?"
"Show me success factors for the AWARD criterion"
"Get insights on why MEMBERSHIP claims get rejected"
```

### 3. Search Evidence
```
"Find all evidence items of type PUBLICATION that were accepted"
"What evidence types have the highest acceptance rate?"
```

### 4. Get Statistics
```
"What's the approval rate by field of endeavor?"
"Show me denial patterns across all criteria"
```

## Example Queries to Try

1. **Basic Search**:
   - "Find all approved cases from 2023"
   - "Show me cases with the case number starting with WAC"

2. **Pattern Analysis**:
   - "What are the top 3 reasons for ORIGINAL_CONTRIBUTION denials?"
   - "Compare success factors between AWARD and MEMBERSHIP criteria"

3. **Evidence Analysis**:
   - "Which evidence types are most commonly excluded and why?"
   - "Show me examples of effective LETTER evidence"

4. **Semantic Search**:
   - "Find decisions discussing citation counts"
   - "Search for cases mentioning peer review"

## Troubleshooting

### MCP Server Won't Connect

1. **Check database connection**:
   ```bash
   psql "postgresql://user:pass@localhost:5432/aao_decisions" -c "SELECT COUNT(*) FROM decisions;"
   ```

2. **Test MCP server manually**:
   ```bash
   uv run aao mcp-server
   # Should show: "🚀 Starting AAO ETL MCP server (STDIN/STDOUT)..."
   ```

3. **Check logs**: Claude Desktop logs are in `~/Library/Logs/Claude/`

### No Data Returned

Ensure you've processed some PDFs first:
```bash
uv run aao init-db
uv run aao process ./path/to/pdfs
```

### Performance Issues

For large databases, consider:
- Adding indexes on frequently queried columns
- Limiting result sets with `limit` parameter
- Using date range filters to narrow searches

## Security Notes

- The MCP server has **read-only** access (no INSERT/UPDATE/DELETE)
- Database credentials are stored in the config file - keep it secure
- Consider using connection strings with read-only database users
- The server only runs locally and doesn't expose any network ports
