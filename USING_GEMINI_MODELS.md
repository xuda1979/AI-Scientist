# Using Gemini Models with AI-Scientist

## ✅ Gemini Support Confirmed

The AI-Scientist software **fully supports Gemini models**, including the latest Gemini 2.5!

## 🚀 How to Use Gemini 2.5

### Basic Command

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5
```

### With All-Code Mode

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5 \
  --all-code
```

### For New Paper Generation

```bash
python main.py \
  --topic "Your Research Topic" \
  --field "Research Field" \
  --model gemini-2.5-pro \
  --max-iterations 5
```

## 📋 Supported Gemini Models

You can use any of these Gemini model names:

| Model Name | Description |
|------------|-------------|
| `gemini-2.5-pro` | Latest Gemini 2.5 Pro (recommended) |
| `gemini-2.5-flash` | Faster Gemini 2.5 Flash |
| `gemini-2.0-flash-exp` | Experimental Gemini 2.0 Flash |
| `gemini-1.5-pro` | Gemini 1.5 Pro |
| `gemini-1.5-flash` | Gemini 1.5 Flash |
| `gemini-exp-1206` | Experimental Gemini (Dec 2024) |

Or use the full model path format:
- `models/gemini-2.5-pro`
- `models/gemini-2.5-flash`
- etc.

## 🎯 Quick Examples

### Example 1: Modify ACSC Paper with Gemini 2.5 Pro

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5
```

### Example 2: Generate New Paper with Gemini 2.5 Flash (Faster)

```bash
python main.py \
  --topic "Transformer Attention Mechanisms" \
  --field "Machine Learning" \
  --model gemini-2.5-flash \
  --max-iterations 5 \
  --output-dir output/transformer_paper
```

### Example 3: All-Code Mode with Gemini 2.5

```bash
python main.py \
  --all-code \
  --topic "PyTorch CNN for CIFAR-10" \
  --model gemini-2.5-pro \
  --max-iterations 6
```

### Example 4: With Custom Prompt and Gemini

```bash
python main.py \
  --modify-existing \
  --output-dir papers/my_paper \
  --model gemini-2.5-pro \
  --user-prompt "Focus on improving the experimental methodology section" \
  --max-iterations 3
```

### Example 5: Test-Time Scaling with Gemini

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --use-test-time-scaling \
  --revision-candidates 5 \
  --max-iterations 5
```

## 🔧 Advanced Features with Gemini

### PDF Review (Gemini can process PDFs!)

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --enable-pdf-review \
  --max-iterations 5
```

**Note**: Gemini models can directly process PDF files, allowing the LLM to see the compiled paper!

### Quality Validation with Gemini

```bash
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --quality-threshold 1.0 \
  --max-iterations 10
```

### Experimental Code Generation with Gemini

```bash
python main.py \
  --all-code \
  --topic "Reinforcement Learning Algorithm" \
  --model gemini-2.5-pro \
  --max-iterations 8 \
  --code-output-dir src
```

## 🌐 Proxy Configuration (If Needed)

If you're behind a proxy, Gemini API calls will automatically use the `HTTPS_PROXY` environment variable:

```bash
# PowerShell
$env:HTTPS_PROXY = "http://your-proxy:port"

# Then run your command
python main.py --model gemini-2.5-pro --modify-existing --output-dir output/acsc
```

The software automatically detects Gemini models and configures the proxy appropriately.

## 💰 Cost Comparison

| Model | Speed | Cost | Best For |
|-------|-------|------|----------|
| `gemini-2.5-pro` | Medium | Medium | High-quality papers, complex reasoning |
| `gemini-2.5-flash` | Fast | Low | Quick iterations, drafts |
| `gpt-5-pro` | Medium | High | Maximum quality |
| `gpt-4o` | Fast | Medium | Balanced performance |

**Recommendation**: Use `gemini-2.5-pro` for excellent quality at lower cost than GPT-5!

## 🆚 Gemini vs GPT for Paper Modification

### Gemini 2.5 Pro Advantages
- ✅ **Lower cost** than GPT-5
- ✅ **Native PDF processing** capability
- ✅ **Large context window** (1M+ tokens)
- ✅ **Multimodal** (can analyze figures directly)
- ✅ **Fast inference** speed

### GPT-5 Pro Advantages
- ✅ **Highest reasoning** quality
- ✅ **Most reliable** for complex tasks
- ✅ **Better at following** precise instructions

**Recommendation**: Try Gemini 2.5 Pro first - it's excellent for most paper modifications!

## 📊 Comparison Table

| Feature | Gemini 2.5 Pro | GPT-5 Pro | GPT-4o |
|---------|----------------|-----------|--------|
| **Quality** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Speed** | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡⚡⚡⚡ |
| **Cost** | 💰💰 | 💰💰💰💰 | 💰💰💰 |
| **PDF Support** | ✅ Native | ❌ No | ❌ No |
| **Context Window** | 1M+ tokens | 128K tokens | 128K tokens |
| **Multimodal** | ✅ Yes | ⚠️ Limited | ✅ Yes |

## 🎯 Recommended Commands for Your Use Case

### To Modify ACSC Paper with Gemini 2.5 Pro

```bash
# Basic modification
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5

# With all-code mode (for generating experimental code)
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5 \
  --all-code

# With PDF review enabled
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 5 \
  --enable-pdf-review

# With quality validation
python main.py \
  --modify-existing \
  --output-dir output/acsc \
  --model gemini-2.5-pro \
  --max-iterations 8 \
  --quality-threshold 1.0
```

## 🔑 API Key Setup

Make sure you have your Gemini API key set up:

```bash
# PowerShell
$env:GEMINI_API_KEY = "your-api-key-here"

# Or add to your environment variables permanently
```

If you're using OpenRouter or another provider, set:

```bash
$env:OPENROUTER_API_KEY = "your-key-here"
```

## ✅ Verification

To verify Gemini support is working, you can run:

```bash
python main.py \
  --test-scaling \
  --model gemini-2.5-pro \
  --scaling-prompt "Test Gemini API connection" \
  --output-dir output/test
```

## 📚 Summary

**YES, you can use Gemini 2.5 to modify papers!**

✅ Full Gemini support already built-in  
✅ Works with all features (all-code, quality validation, etc.)  
✅ Native PDF processing capability  
✅ Lower cost than GPT-5  
✅ Excellent quality for paper modification  

**Just use `--model gemini-2.5-pro` in any command!**

---

*For more information, see:*
- Main workflow: `sciresearch_workflow.py`
- Gemini integration: Lines 1118-1240
- All features documentation: Various `*_DOCUMENTATION.md` files
