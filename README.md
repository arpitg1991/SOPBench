# SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints

## Overview

<p align="center"><img width="100%" src="assets/overview-v8.png" /></p>

This repository contains the data and code for the paper: "SOPBench: Evaluating Language Agents at Following Standard Operating Procedures and Constraints".

## Installation

```bash
# Clone the repository
git clone https://github.com/Leezekun/SOPBench.git
cd SOPBench

# Create and activate conda environment
conda create -n agent python=3.10
conda activate agent

# Install dependencies
pip install -r requirements.txt
```

## Configuration

### API Keys Setup

Create a `.env` file in the root directory with your API keys:

```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_gemini_api_key
FIREWORKS_API_KEY=your_fireworks_api_key
```

### Supported Language Models

The framework supports a wide range of language models through unified interfaces for both multi-turn inference and function calling:

#### API-based Models

- **OpenAI Models**
  - GPT-4o Series: `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`
  - "o" Series: `o1`, `o3`, `o3-mini`, `o4-mini`
- **Anthropic Models**
  - Claude 3.5: `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022`
  - Claude 3.7: `claude-3-7-sonnet-20250219`, `claude-3-7-sonnet-20250219-thinking`
- **Google Gemini Models**
  - Gemini 1.5: `gemini-1.5-flash`, `gemini-1.5-pro`
  - Gemini 2.0: `gemini-2.0-flash-001`, `gemini-2.0-flash`, `gemini-2.0-flash-thinking-exp`
  - Gemini 2.5: `gemini-2.5-pro-preview-03-25`, `gemini-2.5-flash-preview-04-17`
- **Fireworks Models**
  - Various models hosted on the Fireworks AI platform

#### Local Inference
- **OSS Models via vLLM**: Run open-source models locally with vLLM for efficient inference

All models use a unified format for multi-turn inference and function calling, with backend-specific implementations that convert responses to a standardized format compatible with OpenAI's API.

#### Adding Custom Models

You can add or customize supported models by modifying the model lists in `swarm/constants.py`.

## Usage

### Key Parameters

The following command line arguments control the simulation and evaluation:

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--domain` | Test domain | bank, online_market, dmv, healthcare, library, hotel, university |
| `--user_model` | Model for user agent | Any supported model name, "human" for interactive mode, or None (default) |
| `--assistant_model` | Model for assistant agent | Any supported model name |
| `--env_mode` | Environment mode | "prompt" (without code constraint checking), "program" (with code constraint checking) |
| `--tool_list` | Available tools | "full" (all tools), "oracle" (only the oracle-used tools for each case) |
| `--tool_call_mode` | Tool call mode | "fc" (function calling), "react", "act-only" |

### Data Preparation

Pre-generated data is provided in the `data` folder. You can also download the data from [Huggingface](https://huggingface.co/datasets/Zekunli/SOPBench).

To generate new data (note that generating each task using GPT-4o costs approximately $0.015 USD):

```bash
python run_generation.py
```

### Running Simulations

```bash
python run_simulation.py \
  --domain [domain] \
  --user_model [user_model] \
  --assistant_model [assistant_model] \
  --env_mode [env_mode] \
  --tool_list [tool_list] \
  --tool_call_mode [tool_call_mode]
```

### Running Evaluations

```bash
python run_evaluation.py \
  --domain [domain] \
  --user_model [user_model] \
  --assistant_model [assistant_model] \
  --tool_list [tool_list] \
  --tool_call_mode [tool_call_mode]
```

### Reviewing Agent Trajectories

To view agent trajectories and evaluation results:

```bash
python run_checking.py \
  --output_dir [output_dir] \
  --domain [domain] \
  --assistant_model [assistant_model] \
  --tool_call_mode [tool_call_mode] \
  --default_constraint_option [default_constraint_option] \
  --constraint_descr_format [constraint_descr_format] \
  --tool_list [tool_list]
```

Over 24,000 agent trajectories are provided in the `output/` directory for reference.

## Project Structure

```
SOPBench/
├── swarm/                  # Framework code for agent interaction
│   ├── core.py             # Core agent and swarm classes
│   ├── llm_handler.py      # Unified LLM backend handler
│   ├── types.py            # Type definitions
│   ├── util.py             # Utility functions
│   ├── claude.py           # Claude-specific utilities
│   ├── gemini.py           # Gemini-specific utilities
│   └── constants.py        # Project constants and configurations
├── env/                    # Environment for different domains
│   ├── dependencies.py     # Core program code for constraint checking
│   ├── helpers.py          # Helper functions for environment
│   ├── dep_eval.py         # Evaluation utilities
│   └── domains/            # Domain implementations
│       ├── bank/
│       ├── online_market/
│       ├── dmv/
│       ├── healthcare/
│       ├── library/
│       ├── hotel/
│       └── university/
├── data/                   # Task data for simulation and evaluation
├── scripts/                # Scripts for simulation and evaluation
├── output/                 # Simulation and evaluation results
├── plotting/               # Data visualization utilities
├── run_generation.py       # Task generation script
├── run_simulation.py       # Simulation script
├── run_evaluation.py       # Evaluation script
├── run_checking.py         # Validation script
└── run_operation.py        # Operations script
```
