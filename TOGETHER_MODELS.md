# Together AI Models for AAO ETL Database Chat

## Current Default Model

**`meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`**

- **Size**: 70 billion parameters
- **Strengths**: 
  - Good balance of speed, cost, and capability
  - Solid function/tool calling support
  - Strong reasoning for legal text analysis
  - Fast inference on Together AI infrastructure
- **Cost**: ~$0.88/M input tokens, ~$0.88/M output tokens
- **Context**: 128K tokens

## Recommended Alternative Models

### For Better Reasoning: DeepSeek-V3
**`deepseek-ai/DeepSeek-V3`**
- **Best for**: Complex analysis, multi-step reasoning
- **Strengths**: 
  - Exceptional reasoning capabilities
  - Great at understanding complex legal language
  - Strong tool calling support
- **Cost**: ~$0.27/M input tokens, ~$1.10/M output tokens
- **Context**: 64K tokens
- **When to use**: Complex criterion analysis, pattern detection

### For Maximum Capability: Llama 3.1 405B
**`meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo`**
- **Best for**: Most challenging queries, detailed analysis
- **Strengths**:
  - Highest capability Llama model
  - Superior reasoning and analysis
  - Excellent instruction following
- **Cost**: ~$3.50/M input tokens, ~$3.50/M output tokens
- **Context**: 128K tokens
- **When to use**: When you need the best possible analysis

### For Latest Features: Llama 3.3 70B
**`meta-llama/Llama-3.3-70B-Instruct-Turbo`**
- **Best for**: Updated model with improvements over 3.1
- **Strengths**:
  - Newer architecture improvements
  - Better instruction following
  - Similar cost to 3.1 70B
- **Cost**: ~$0.88/M input tokens, ~$0.88/M output tokens
- **Context**: 128K tokens

### Strong Alternative: Qwen 2.5 72B
**`Qwen/Qwen2.5-72B-Instruct-Turbo`**
- **Best for**: Alternative perspective, technical analysis
- **Strengths**:
  - Strong coding and analytical capabilities
  - Good multilingual support
  - Competitive reasoning
- **Cost**: ~$1.20/M input tokens, ~$1.20/M output tokens
- **Context**: 32K tokens

## How to Choose a Model

### Option 1: Interactive Selection

When you start the chat, you'll be prompted:

```bash
uv run python chat_with_db.py

📋 Available Together AI models:
  1. meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo (default)
  2. meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
  3. meta-llama/Llama-3.3-70B-Instruct-Turbo
  4. Qwen/Qwen2.5-72B-Instruct-Turbo
  5. deepseek-ai/DeepSeek-V3

Choose model [1]: 5  ← Enter number for DeepSeek-V3
```

### Option 2: Environment Variable

Set the MODEL environment variable:

```bash
export MODEL="deepseek-ai/DeepSeek-V3"
uv run python chat_with_db.py
```

### Option 3: Modify Default in Code

Edit `chat_with_db.py` line 62:

```python
defaults = {
    "openai": "gpt-4o-mini",
    "together": "deepseek-ai/DeepSeek-V3",  # Change this
    "anthropic": "claude-3-5-sonnet-20241022"
}
```

## Performance Comparison for AAO Database Queries

### Simple Queries ("Find 10 denied cases")
- **All models**: Fast, accurate, no significant difference
- **Recommendation**: Stick with default Llama 3.1 70B

### Complex Pattern Analysis ("What factors lead to ORIGINAL_CONTRIBUTION denials?")
- **Best**: DeepSeek-V3, Llama 3.1 405B
- **Good**: Llama 3.3 70B, Qwen 2.5 72B
- **Adequate**: Llama 3.1 70B (default)

### Legal Text Understanding
- **Best**: DeepSeek-V3, Llama 3.1 405B
- **Good**: All others
- **Note**: All models handle legal language well

### Multi-step Reasoning ("Compare 3 criteria and identify patterns")
- **Best**: DeepSeek-V3, Llama 3.1 405B
- **Good**: Llama 3.3 70B
- **Adequate**: Llama 3.1 70B (default), Qwen 2.5 72B

## Cost Considerations

### Budget-Friendly
- Llama 3.1 70B: **$0.88/M tokens**
- Llama 3.3 70B: **$0.88/M tokens**
- DeepSeek-V3: **$0.27/M input, $1.10/M output**

### Premium
- Llama 3.1 405B: **$3.50/M tokens** (4x more expensive)

### Example Costs (per 100 queries)
- Simple queries (1K tokens each): **$0.09 - $0.35**
- Complex queries (5K tokens each): **$0.44 - $1.75**

## Recommendations by Use Case

### Legal Research & Analysis
**Recommended**: DeepSeek-V3 or Llama 3.1 405B
- Better at understanding complex legal reasoning
- Identifies subtle patterns in AAO decisions
- Worth the extra cost for accuracy

### Pattern Discovery
**Recommended**: DeepSeek-V3
- Excellent at finding non-obvious patterns
- Strong analytical capabilities
- Best cost/performance for analysis

### General Database Queries
**Recommended**: Llama 3.1 70B (default) or Llama 3.3 70B
- Fast and cost-effective
- More than sufficient for straightforward queries
- Save premium models for complex tasks

### Production Use
**Recommended**: DeepSeek-V3
- Best balance of cost and capability
- Reliable reasoning
- Good error handling

## Testing Different Models

Try this to compare models on the same query:

```bash
# Test with default (Llama 3.1 70B)
uv run python chat_with_db.py
> What are common ORIGINAL_CONTRIBUTION denial reasons?

# Test with DeepSeek-V3
export MODEL="deepseek-ai/DeepSeek-V3"
uv run python chat_with_db.py
> What are common ORIGINAL_CONTRIBUTION denial reasons?

# Compare the quality and detail of responses
```

## Current Limitations

All Together AI models have some quirks with function calling:
- ✅ Can call tools successfully
- ⚠️ Don't support `tool_choice` parameter (we handle this)
- ⚠️ Conversation history needs special handling (we handle this)
- ✅ Work well for single tool calls
- ⚠️ May need guidance for complex multi-tool scenarios

Our chat script handles these limitations automatically.

## Summary

**For most users**: Start with the default **Llama 3.1 70B** - it works well and is cost-effective.

**For serious analysis**: Upgrade to **DeepSeek-V3** - best bang for buck with excellent reasoning.

**For maximum quality**: Use **Llama 3.1 405B** when analyzing critical decisions or patterns.
