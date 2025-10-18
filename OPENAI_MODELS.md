# OpenAI Models for AAO ETL Database Chat

## Available OpenAI Models

### Standard Models

#### **gpt-4o-mini** (Default)
- **Best for**: Most queries, cost-effective
- **Strengths**: 
  - Fast and cheap
  - Good balance of capability
  - Excellent function calling
  - Solid reasoning for most tasks
- **Cost**: ~$0.15/M input tokens, ~$0.60/M output tokens
- **Context**: 128K tokens
- **When to use**: General database queries, simple analysis

#### **gpt-4o**
- **Best for**: Complex analysis, high accuracy needed
- **Strengths**:
  - Superior reasoning
  - Better at complex legal text
  - Excellent function calling
  - More accurate than mini
- **Cost**: ~$2.50/M input tokens, ~$10.00/M output tokens
- **Context**: 128K tokens
- **When to use**: Important analysis, complex pattern detection

#### **gpt-4-turbo**
- **Best for**: Previous generation, still very capable
- **Strengths**:
  - Strong reasoning
  - Good function calling
  - Well-tested and reliable
- **Cost**: ~$10.00/M input tokens, ~$30.00/M output tokens
- **Context**: 128K tokens
- **When to use**: When you prefer proven performance

### Reasoning Models (o-series)

#### **o1**
- **Best for**: Deep analysis, complex reasoning tasks
- **Strengths**:
  - Chain-of-thought reasoning
  - Excellent for pattern detection
  - Superior at complex legal analysis
  - Can think through problems step-by-step
- **Cost**: ~$15.00/M input tokens, ~$60.00/M output tokens
- **Context**: 200K tokens
- **When to use**: Complex criterion analysis, multi-step reasoning
- **Note**: Slower than standard models (thinks before responding)

#### **o1-mini**
- **Best for**: Faster reasoning at lower cost
- **Strengths**:
  - Chain-of-thought reasoning
  - Faster than o1
  - More affordable
  - Good at structured analysis
- **Cost**: ~$3.00/M input tokens, ~$12.00/M output tokens
- **Context**: 128K tokens
- **When to use**: Reasoning tasks where speed matters

#### **o3-mini** (If Available)
- **Best for**: Latest reasoning capabilities
- **Strengths**:
  - Next-generation reasoning
  - Improved efficiency
  - Better at complex tasks
- **Cost**: TBD (check OpenAI pricing)
- **Context**: TBD
- **When to use**: Cutting-edge reasoning needs
- **Note**: May not be available yet - check OpenAI API

## How to Use

### Option 1: Interactive Selection

```bash
uv run python chat_with_db.py

Available providers: openai, together
Choose provider [openai]: openai

📋 Available OpenAI models:
  1. gpt-4o-mini (default)
  2. gpt-4o
  3. gpt-4-turbo
  4. o1 [reasoning model]
  5. o1-mini [reasoning model]
  6. o3-mini [reasoning model]

Choose model [1]: 4  ← Select o1 for deep reasoning
```

### Option 2: Environment Variable

```bash
# Use GPT-4o for better accuracy
export MODEL="gpt-4o"
uv run python chat_with_db.py

# Use o1 for deep reasoning
export MODEL="o1"
uv run python chat_with_db.py

# Use o1-mini for fast reasoning
export MODEL="o1-mini"
uv run python chat_with_db.py
```

### Option 3: One-liner

```bash
# GPT-4o
MODEL="gpt-4o" uv run python chat_with_db.py

# o1 reasoning model
MODEL="o1" uv run python chat_with_db.py

# o3-mini (if available)
MODEL="o3-mini" uv run python chat_with_db.py
```

## Model Comparison for AAO Queries

### Simple Queries ("Find 10 denied cases")
- **Best**: gpt-4o-mini (fast, cheap, sufficient)
- **Alternative**: gpt-4o (if you want higher accuracy)
- **Overkill**: o1/o1-mini (too slow for simple queries)

### Complex Pattern Analysis ("Why do ORIGINAL_CONTRIBUTION cases fail?")
- **Best**: o1 (superior reasoning)
- **Good**: o1-mini (fast reasoning)
- **Adequate**: gpt-4o (strong but less reasoning depth)

### Legal Text Understanding
- **Best**: o1, gpt-4o
- **Good**: gpt-4-turbo, o1-mini
- **Adequate**: gpt-4o-mini

### Multi-step Reasoning
- **Best**: o1 (designed for this)
- **Good**: o1-mini (faster but still strong)
- **Adequate**: gpt-4o

### Cost-Effective Analysis
- **Best**: gpt-4o-mini (cheapest, still very capable)
- **Good**: o1-mini (best reasoning per dollar)
- **Premium**: o1, gpt-4o (worth it for critical work)

## Cost Comparison (per 100 queries)

### Simple queries (~1K tokens input/output each)
- **gpt-4o-mini**: ~$0.08
- **gpt-4o**: ~$1.25
- **o1-mini**: ~$1.50
- **o1**: ~$7.50

### Complex queries (~5K tokens input/output each)
- **gpt-4o-mini**: ~$0.38
- **gpt-4o**: ~$6.25
- **o1-mini**: ~$7.50
- **o1**: ~$37.50

## Recommendations by Use Case

### Budget-Conscious Usage
**Use**: gpt-4o-mini (default)
- Handles 90% of queries well
- Very affordable
- Fast responses

### Balanced Usage
**Use**: gpt-4o
- Better accuracy than mini
- Good reasoning
- Worth the cost for important work

### Deep Analysis
**Use**: o1-mini or o1
- Chain-of-thought reasoning
- Best for understanding complex legal patterns
- Takes time but delivers insights

### Production/Research
**Use**: o1 for analysis, gpt-4o-mini for simple queries
- Use o1 when you need the best reasoning
- Use gpt-4o-mini for routine queries
- Best balance of cost and capability

## Special Features

### o-series Models (o1, o1-mini, o3-mini)
These models use **chain-of-thought reasoning**:
- ✅ Think through problems step-by-step
- ✅ Show their reasoning process (if requested)
- ✅ Better at complex, multi-step problems
- ⚠️ Slower than standard models
- ⚠️ More expensive
- ⚠️ May not support all standard model features

**Best for**:
- Analyzing why certain evidence patterns fail
- Identifying non-obvious denial reasons
- Understanding complex criterion interactions
- Deep pattern analysis across many cases

### When to Use Reasoning Models

Use o1/o1-mini/o3-mini when asking:
- "Why do cases with X evidence still get denied?"
- "What's the relationship between Y and Z in denials?"
- "Analyze the pattern across these 50 cases and identify the key factors"
- "What makes the difference between approved and denied cases with similar evidence?"

Don't use reasoning models for:
- Simple database lookups
- Straightforward counts or lists
- Basic filtering queries

## Examples

### Using gpt-4o-mini (Default)
```bash
uv run python chat_with_db.py
> Find 10 denied cases in computer science
> Show me the database schema
> Search for decisions from 2023
```
**Fast, cheap, works great for these queries.**

### Using gpt-4o (Better Accuracy)
```bash
MODEL="gpt-4o" uv run python chat_with_db.py
> What are the most common reasons for ORIGINAL_CONTRIBUTION denials?
> Compare success patterns between AWARD and MEMBERSHIP criteria
> Analyze the quality of evidence in approved vs denied cases
```
**More accurate analysis, worth the cost.**

### Using o1 (Deep Reasoning)
```bash
MODEL="o1" uv run python chat_with_db.py
> Analyze why some cases with strong publications still get denied
> What underlying patterns distinguish successful ORIGINAL_CONTRIBUTION claims?
> Examine the relationship between evidence timing and case outcomes
```
**Best reasoning, takes longer but provides deeper insights.**

### Using o1-mini (Fast Reasoning)
```bash
MODEL="o1-mini" uv run python chat_with_db.py
> What factors correlate with MEMBERSHIP criterion success?
> Identify patterns in cases where AAO overturned director's decision
> Analyze the impact of evidence types on approval rates
```
**Good reasoning, faster and cheaper than o1.**

## Summary

**Default (gpt-4o-mini)**: Great for 90% of queries
**Upgrade to gpt-4o**: When accuracy matters more than cost
**Use o1**: For deep analysis and complex reasoning
**Use o1-mini**: For reasoning on a budget
**Try o3-mini**: When available, for latest capabilities
