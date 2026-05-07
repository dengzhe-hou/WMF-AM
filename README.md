# WMF-AM: Probing LLM Working Memory via Depth-Parameterized Cumulative State Tracking

## Overview

WMF-AM is a calibrated no-scratchpad probe of cumulative arithmetic state tracking for evaluating LLM process-level capabilities. Given K sequential additive/subtractive operations on an initial numeric state, the model must report the final state without external scratchpad.

**Key features:**
- **K-calibrated difficulty**: Adjustable depth parameter K maintains discriminability across the full capability spectrum (0.5B to frontier models)
- **Construct isolation**: Four ablation controls (K=1, non-arithmetic ceiling, yoked cancellation, prompt paraphrase) isolate cumulative arithmetic tracking as the difficulty source
- **Downstream prediction**: WMF-AM is associated with agent performance on a 10-task deterministic battery (τ=0.595, p<0.001, N=28)
- **Cross-domain extension**: Non-arithmetic cumulative probes (permissions, schedule, inventory) show strong rank consistency (τ=0.728, N=28)
- **Extended K-sweep**: K=3 to 100 reveals sigmoid-cliff collapse dynamics with model-specific critical depths
- **Load-shift intervention**: History removal diagnostic separates models that internalize state from those that rely on external context

## Repository Structure

```
code/
├── config.py                    # Model registry + call_model() utility
├── wmf_am_multiseed_expansion.py # Core WMF-AM probe (multi-seed)
├── wmf_am_extended_k.py         # Extended K-sweep (K=3..100)
├── wmf_am_control.py            # K=1 single-step control
├── wmf_am_nonarithmetic.py      # Non-arithmetic ceiling ablation
├── wmf_am_yoked_control.py      # Yoked cancellation control
├── wmf_am_template_harmonization.py # Template stability check
├── wmf_am_paraphrase.py         # Paraphrase robustness
├── oos_validation.py            # Agent battery + WMF-AM validation
├── api_held_out_validation.py   # API model held-out validation
├── agent_load_shift.py          # Load-shift intervention
├── baseline_mmlu_gsm8k.py       # MMLU/GSM8K baseline comparison
├── wmf_am_extended_k.py         # Extended K-sweep
└── analysis/
    ├── beta1_slope_analysis.py  # Manipulation slope analysis
    └── gen_figures.py           # Figure generation

data/
├── master_28models.csv          # All 28 models × all metrics
├── wmf_am_item_level_consolidated.csv  # 1500 item-level WMF-AM trials
├── cef_agent_validation_all.json       # 10-task agent battery (20 models)
├── cef_phase1_full.json               # Phase 1 item-level data (7 models)
├── nexp/                              # Per-model expansion data (8 models)
└── results/
    ├── extended_k_*.json              # Extended K-sweep results
    ├── load_shift_*.json              # Load-shift intervention results
    ├── baseline_*.json                # MMLU/GSM8K results
    ├── api_held_out_*.json            # API model validation results
    └── wmf_am_*_control_*.json        # Ablation results

PREREGISTRATION.md   # Pre-registered analysis plan (GitHub-committed)
```

## Quick Start

### Install dependencies
```bash
pip install openai python-dotenv scipy scikit-learn
```

### Configure API keys (for closed models)
```bash
cp .env.example .env
# Edit .env with your OpenRouter API key and Google AI Studio key
```

### Run WMF-AM probe on a model
```bash
# Open-weight model via Ollama
python code/wmf_am_multiseed_expansion.py --models ollama:qwen2.5:7b

# API model via OpenRouter
python code/api_held_out_validation.py --models openrouter:gpt-4o --wmf-only
```

### Run agent battery
```bash
python code/oos_validation.py  # Runs WMF-AM + 10-task agent battery
```

### Run extended K-sweep
```bash
python code/wmf_am_extended_k.py --models ollama:qwen2.5:7b --k-values 3 5 7 10 15 20 30 50
```

### Run load-shift intervention
```bash
python code/agent_load_shift.py --models openrouter:gpt-4o
```

### Run ablation controls
```bash
python code/wmf_am_control.py --model ollama:qwen2.5:7b          # K=1 control
python code/wmf_am_nonarithmetic.py --models ollama:qwen2.5:7b   # Non-arithmetic
python code/wmf_am_yoked_control.py --model ollama:qwen2.5:7b    # Yoked cancellation
```

### Generate figures
```bash
python code/analysis/gen_figures.py
```

## Models Evaluated (N=28)

**Open-weight (21 models, 12 families):** Qwen 2.5 (0.5B–32B), Llama 3.x (1B–70B), Gemma 2 (2B–27B), DeepSeek-R1 distill (7B/14B), Mistral 7B, Mixtral 8×7B, Phi-3 14B, Command-R 35B, Yi 34B, TinyLlama 1.1B

**Closed API (7 models):** GPT-4o, GPT-4o-mini, o3-mini (LRM), Claude Sonnet 4, Gemini 2.5 Flash, DeepSeek-V3, DeepSeek-R1 full (LRM)

## Key Results

| Analysis | N | Key Statistic |
|----------|---|--------------|
| WMF-AM → Agent | 28 | τ=0.595, CI [0.374, 0.785] |
| Partial τ(WMF-AM \| MMLU) | 28 | 0.302 (p=0.029) |
| Cross-domain (non-arith) | 28 | τ=0.728 |
| K=1 control | 28 | 22/28 ceiling |
| Non-arithmetic | 28 | mean 0.92 |
| Yoked cancellation | 28 | τ=0.381 |
| K-sweep K_crit | 28 | 1.3–55.3 (does NOT predict agent) |
| Load-shift | 28 | mean Δ=0.30 |

## License

CC-BY-4.0
