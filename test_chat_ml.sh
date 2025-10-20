#!/bin/bash
# Test script for chat with real queries

export OPENAI_API_KEY=$(cat .env | grep OPENAI_API_KEY | cut -d'=' -f2)

echo "Testing chat with OpenAI gpt-4o-mini..."
echo ""
echo "Provider: openai"
echo "Model: 1 (gpt-4o-mini)"
echo "Query: find 2 machine learning cases and show me the evidence items for each case"
echo ""

uv run python chat_with_db.py <<EOF


1
find 2 machine learning cases and show me the evidence items for each case
exit
EOF
