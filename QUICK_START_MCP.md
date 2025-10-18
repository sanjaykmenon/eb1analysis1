# Quick Start: Using LLM with AAO ETL Database via MCP

## 🎯 Goal
Query your AAO EB-1A decision database using natural language through Claude, OpenAI, Together AI, or other AI assistants.

## ✅ Prerequisites
- Database initialized with `uv run aao init-db`
- Some PDFs processed with `uv run aao process ./pdfs`
- Database credentials in `.env` file or `DATABASE_URL` environment variable
- API key for your chosen LLM provider

## 🚀 Quick Setup Options

### Option 1: Python Chat Script (Easiest - Works with Any Provider!)

Use the included chat script with OpenAI, Together AI, or Anthropic:

```bash
# Set your API key (choose one)
export OPENAI_API_KEY="sk-..."           # For OpenAI
export TOGETHER_API_KEY="..."           # For Together AI  
export ANTHROPIC_API_KEY="sk-ant-..."  # For Anthropic

# Make sure DATABASE_URL is set
export DATABASE_URL="postgresql+psycopg://user:pass@localhost/aao_decisions"

# Start chatting!
uv run python chat_with_db.py
```

**That's it!** The script will:
- ✅ Auto-detect which provider you have API keys for
- ✅ Connect to your MCP server
- ✅ Let you query your database in natural language
- ✅ Handle all the tool calling automatically

**Example queries:**
```
You: What's in the database schema?
You: Find the last 5 denied cases
You: What are common ORIGINAL_CONTRIBUTION denial reasons?
You: Search for decisions about machine learning
```

**Using Together AI specifically:**
```bash
# Get API key from https://api.together.xyz/settings/api-keys
export TOGETHER_API_KEY="your-key-here"
export DATABASE_URL="postgresql+psycopg://user:pass@localhost/aao_decisions"

uv run python chat_with_db.py
# Choose "together" when prompted (or it will auto-detect)
```

**Supported models:**
- **OpenAI**: `gpt-4o-mini` (default), `gpt-4o`, `gpt-4-turbo`
- **Together AI**: `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` (default), any Together model
- **Anthropic**: `claude-3-5-sonnet-20241022` (default), other Claude models

### Option 2: Claude Desktop (Great for Mac Users)

#### Step 1: Update the Config Template

Edit `claude_desktop_config.json` in this repo and update your DATABASE_URL:

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
        "DATABASE_URL": "YOUR_ACTUAL_DATABASE_URL_HERE"
      }
    }
  }
}
```

#### Step 2: Copy to Claude Desktop

```bash
# Create the Claude config directory if needed
mkdir -p ~/Library/Application\ Support/Claude

# Copy the config
cp claude_desktop_config.json ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

#### Step 3: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Reopen Claude Desktop
3. Look for the 🔌 icon showing "aao-etl" is connected

#### Step 4: Test It!

Open Claude and try queries like:

```
"Use the aao-etl MCP server to show me the database schema"

"Find the last 5 denied EB-1A cases"

"What are common denial patterns for the ORIGINAL_CONTRIBUTION criterion?"

"Search for decisions mentioning machine learning"

"Show me statistics on approval rates by field of endeavor"
```

### Option 3: VS Code Extensions (Cline/Continue)

See the `MCP_SETUP.md` file for detailed VS Code extension setup instructions.

## 📊 Available MCP Tools

Your MCP server exposes these tools to the LLM:

1. **`describe_schema`** - Shows database tables and relationships
2. **`search_decisions`** - Query decisions by text, outcome, dates, criteria
3. **`get_decision_detail`** - Get full details for a specific case
4. **`analyze_criterion_patterns`** - Analyze denial/success patterns for criteria

## 🧪 Test Without LLM

You can test the MCP server manually:

```bash
# Run the test script
python test_mcp.py

# Or manually test the server (it will wait for JSON-RPC input)
uv run aao mcp-server
```

## 🎨 Example Natural Language Queries

### Basic Queries
- "How many total decisions are in the database?"
- "Show me all approved cases from 2023"
- "Find cases in the Computer Science field"

### Pattern Analysis
- "What are the top reasons MEMBERSHIP criterion gets denied?"
- "Compare success factors between AWARD and ORIGINAL_CONTRIBUTION"
- "Show me linguistic patterns in AAO denials"

### Evidence Analysis
- "Which evidence types have the highest acceptance rate?"
- "Find examples of strong PUBLICATION evidence"
- "Why do letters of recommendation get excluded?"

### Semantic Search
- "Find decisions discussing citation metrics"
- "Search for cases mentioning 'peer review process'"
- "Show decisions about machine learning research"

## 🔍 Debugging

### Test the Python chat script:
```bash
# Make sure your environment is set up
export DATABASE_URL="postgresql+psycopg://user:pass@localhost/aao_decisions"
export TOGETHER_API_KEY="your-key"  # or OPENAI_API_KEY, ANTHROPIC_API_KEY

# Run the chat
python chat_with_db.py
```

### Check if MCP server starts:
```bash
uv run aao mcp-server
# Should show: "🚀 Starting AAO ETL MCP server (STDIN/STDOUT)..."
# Press Ctrl+C to exit
```

### Verify database connection:
```bash
# Test your DATABASE_URL
psql "$DATABASE_URL" -c "SELECT COUNT(*) FROM decisions;"
```

### Check Claude Desktop logs:
```bash
tail -f ~/Library/Logs/Claude/mcp*.log
```

## 💡 Tips

1. **Be specific**: "Show me denials for ORIGINAL_CONTRIBUTION" works better than "show me some denials"

2. **Use criterion codes**: The database uses codes like `AWARD`, `MEMBERSHIP`, `ORIGINAL_CONTRIBUTION`, etc.

3. **Limit results**: Start with small result sets (limit 5-10) then increase if needed

4. **Ask for schema first**: "Describe the database schema" helps the LLM understand what's available

## 🛠️ Alternative: Use with VS Code Extensions

If using **Cline** or **Continue.dev** in VS Code, add to their settings:

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
        "DATABASE_URL": "postgresql+psycopg://user:pass@localhost/aao_decisions"
      }
    }
  }
}
```

## 📚 Next Steps

- Process more PDFs to build up your database
- Run `uv run aao insights-parallel` to extract criterion-level insights
- Ask the LLM to help you discover patterns in denial reasons
- Use semantic search to find similar cases
