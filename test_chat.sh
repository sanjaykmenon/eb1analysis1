#!/bin/bash
# Quick test script for AAO ETL MCP + LLM chat

echo "🧪 AAO ETL Database Chat - Quick Test"
echo "======================================"
echo

# Check for database URL
if [ -z "$DATABASE_URL" ]; then
    echo "❌ DATABASE_URL not set!"
    echo "   Set it with: export DATABASE_URL='postgresql+psycopg://user:pass@host/db'"
    exit 1
fi

echo "✅ DATABASE_URL is set"

# Check for API keys
has_key=0
if [ -n "$OPENAI_API_KEY" ]; then
    echo "✅ OPENAI_API_KEY is set (will use OpenAI)"
    has_key=1
fi

if [ -n "$TOGETHER_API_KEY" ]; then
    echo "✅ TOGETHER_API_KEY is set (will use Together AI)"
    has_key=1
fi

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "✅ ANTHROPIC_API_KEY is set (will use Anthropic)"
    has_key=1
fi

if [ $has_key -eq 0 ]; then
    echo "❌ No API keys found!"
    echo "   Set one of:"
    echo "     export OPENAI_API_KEY='sk-...'"
    echo "     export TOGETHER_API_KEY='...'"
    echo "     export ANTHROPIC_API_KEY='sk-ant-...'"
    exit 1
fi

echo
echo "🚀 Starting interactive chat..."
echo "   Type 'exit' to quit"
echo

uv run python chat_with_db.py
