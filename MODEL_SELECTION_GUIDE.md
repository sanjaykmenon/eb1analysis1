# Model Selection Guide

## Quick Reference

The chat interface now supports **OpenAI** (7 models) and **Together AI** (5 models).

### Usage Methods

#### 1. Interactive Selection (Default)
```bash
uv run python chat_with_db.py
```
You'll be prompted to:
1. Choose provider (openai or together)
2. Choose model from available list

#### 2. Environment Variable
```bash
# OpenAI models
MODEL="o1" uv run python chat_with_db.py
MODEL="gpt-4o" uv run python chat_with_db.py
MODEL="gpt-4o-mini" uv run python chat_with_db.py

# Together AI models
MODEL="deepseek-ai/DeepSeek-V3" uv run python chat_with_db.py
MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo" uv run python chat_with_db.py
```

#### 3. One-liner with API Key
```bash
OPENAI_API_KEY="sk-..." MODEL="o1" uv run python chat_with_db.py
TOGETHER_API_KEY="..." MODEL="deepseek-ai/DeepSeek-V3" uv run python chat_with_db.py
```

---

## OpenAI Models (7 options)

### Standard Models (Fast, Cost-Effective)
1. **gpt-4o-mini** ⭐ *Default*
   - Best for: Budget-conscious queries, high volume
   - Cost: ~$0.08/100 simple queries
   - Speed: Very fast
   
2. **gpt-4o**
   - Best for: Balanced performance and cost
   - Cost: ~$0.70/100 simple queries
   - Speed: Fast

3. **gpt-4-turbo**
   - Best for: Legacy compatibility
   - Cost: ~$0.25/100 simple queries
   - Speed: Fast

4. **gpt-5-mini-2025-08-07**
   - Best for: Latest lightweight model
   - Cost: Similar to gpt-4o-mini
   - Speed: Very fast

### Reasoning Models (Deep Analysis)
5. **o1** 🧠
   - Best for: Complex legal analysis, multi-step reasoning
   - Cost: ~$7.50/100 simple queries (10x more than standard)
   - Speed: Slower (chain-of-thought reasoning)
   - Use when: Analyzing complex denial patterns, comparing multiple cases

6. **o1-mini** 🧠
   - Best for: Faster reasoning tasks
   - Cost: ~$1.50/100 simple queries
   - Speed: Medium (faster than o1)
   - Use when: Need reasoning but want faster responses

7. **o3-mini** 🧠
   - Best for: Latest reasoning model (if available)
   - Cost: TBD (likely similar to o1-mini)
   - Speed: Medium
   - Use when: Want cutting-edge reasoning capabilities

---

## Together AI Models (5 options)

### Meta Llama Models
1. **Meta-Llama-3.1-70B-Instruct-Turbo** ⭐ *Default*
   - Best for: General queries, good balance
   - Cost: Very affordable (~$0.30/M tokens)
   - Speed: Fast

2. **Meta-Llama-3.1-405B-Instruct-Turbo**
   - Best for: Complex reasoning, better than 70B
   - Cost: Moderate (~$1.20/M tokens)
   - Speed: Slower but very capable

3. **Llama-3.3-70B-Instruct-Turbo**
   - Best for: Latest Llama improvements
   - Cost: Similar to 3.1-70B
   - Speed: Fast

### Alternative Models
4. **Qwen/Qwen2.5-72B-Instruct-Turbo**
   - Best for: Strong alternative to Llama
   - Cost: Very affordable (~$0.30/M tokens)
   - Speed: Fast

5. **deepseek-ai/DeepSeek-V3**
   - Best for: Excellent reasoning capabilities
   - Cost: Very affordable (~$0.30/M tokens)
   - Speed: Fast
   - Special: Known for strong analytical abilities

---

## When to Use Each Model

### Budget-Conscious (High Volume)
- **OpenAI**: `gpt-4o-mini` or `gpt-5-mini`
- **Together AI**: Any model (all very affordable)

### Balanced Performance
- **OpenAI**: `gpt-4o`
- **Together AI**: `Meta-Llama-3.1-70B-Instruct-Turbo` (default)

### Complex Legal Analysis
- **OpenAI**: `o1` or `o1-mini` (reasoning models)
- **Together AI**: `DeepSeek-V3` or `Meta-Llama-3.1-405B`

### Fast Prototyping/Testing
- **OpenAI**: `gpt-4o-mini`
- **Together AI**: Any 70B model

### Deep Multi-Step Analysis
- **OpenAI**: `o1` (best reasoning)
- **Together AI**: `DeepSeek-V3` or `405B` model

---

## Interactive Menu Examples

### OpenAI Selection
```
Available providers: openai, together
Choose provider [openai]: ← press Enter

📋 Available OpenAI models:
  1. gpt-4o-mini (default)
  2. gpt-4o
  3. gpt-4-turbo
  4. o1 [reasoning model]
  5. o1-mini [reasoning model]
  6. o3-mini [reasoning model]
  7. gpt-5-mini-2025-08-07

Choose model [1]: 4 ← select o1 for deep analysis

🚀 Starting chat with openai...
   Model: o1
```

### Together AI Selection
```
Available providers: openai, together
Choose provider [openai]: together

📋 Available Together AI models:
  1. meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo (default)
  2. meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
  3. meta-llama/Llama-3.3-70B-Instruct-Turbo
  4. Qwen/Qwen2.5-72B-Instruct-Turbo
  5. deepseek-ai/DeepSeek-V3

Choose model [1]: 5 ← select DeepSeek for reasoning

🚀 Starting chat with together...
   Model: deepseek-ai/DeepSeek-V3
```

---

## Cost Comparison (Per 100 Queries)

### Simple Queries (1K input + 500 output)
| Model | Cost | Provider |
|-------|------|----------|
| gpt-4o-mini | $0.08 | OpenAI |
| gpt-5-mini | ~$0.08 | OpenAI |
| gpt-4-turbo | $0.25 | OpenAI |
| gpt-4o | $0.70 | OpenAI |
| o1-mini | $1.50 | OpenAI |
| o1 | $7.50 | OpenAI |
| Any Together AI | ~$0.05-0.20 | Together AI |

### Complex Queries (5K input + 2K output)
| Model | Cost | Provider |
|-------|------|----------|
| gpt-4o-mini | $0.38 | OpenAI |
| gpt-4-turbo | $1.25 | OpenAI |
| gpt-4o | $3.50 | OpenAI |
| o1-mini | $7.50 | OpenAI |
| o1 | $37.50 | OpenAI |
| Any Together AI | ~$0.25-1.00 | Together AI |

💡 **Tip**: For high-volume analysis, Together AI models are 5-10x cheaper than OpenAI while still providing excellent quality.

---

## Example Workflows

### Testing New Features (Fast & Cheap)
```bash
# Use default together AI
uv run python chat_with_db.py
# Select: together → [Enter for default]
```

### Production Analysis (Balanced)
```bash
MODEL="gpt-4o" uv run python chat_with_db.py
# Good balance of quality and cost
```

### Deep Legal Insights (Complex)
```bash
MODEL="o1" uv run python chat_with_db.py
# Use when: "Compare denial patterns across 50 cases"
# Use when: "What are subtle differences in AAO reasoning?"
```

### Budget-Conscious Exploration
```bash
MODEL="deepseek-ai/DeepSeek-V3" uv run python chat_with_db.py
# Excellent reasoning at Together AI pricing
```

---

## Troubleshooting

### "Invalid provider '1'" Error
- Don't enter numbers for provider selection
- Type `openai` or `together` (or press Enter for default)

### "Invalid API key" Error
- Check environment variable: `echo $OPENAI_API_KEY`
- Get key from: https://platform.openai.com/api-keys (OpenAI)
- Get key from: https://api.together.ai/settings/api-keys (Together AI)

### Model Not Working
- Some models (o3-mini) may not be available yet
- Check OpenAI API status: https://status.openai.com
- Try a different model from the list

---

For detailed model specifications:
- OpenAI models: See `OPENAI_MODELS.md`
- Together AI models: See `TOGETHER_MODELS.md`
