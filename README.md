# Spacewar! AI

A faithful recreation of the classic 1962 game Spacewar! with modern reinforcement learning agents.

## Overview

Two AI-controlled spaceships compete around a central star with gravitational pull. Ships can thrust, rotate, and fire torpedoes while managing limited fuel and ammunition.

## Features

- Faithful recreation of original Spacewar! physics
- Inverse-square gravity from central star
- Newtonian physics with momentum
- Screen wrap-around (toroidal topology)
- Limited fuel and torpedoes with cooldown

### Reinforcement Learning

- **DQN**: Double DQN with Dueling architecture
- **PPO**: Proximal Policy Optimization with GAE
- Action masking (prevents invalid actions when out of ammo/fuel)
- Self-play with league training (maintains pool of past policies)

### LLM Integration (Optional)

- API support: OpenAI, Anthropic, Google (Gemini), xAI (Grok), Groq, Together, Mistral, DeepSeek, OpenRouter
- Local LLM via llama-cpp-python (auto-downloads best model for your hardware)
- Training-time guidance (not real-time) for sample-efficient exploration
- Iterative strategy refinement (Eureka-style) with automatic provider failover

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai_space_war

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Demo Mode

Watch heuristic agents play:

```bash
python spacewar_ai/main.py --demo
```

### Training

```bash
# Train with PPO (recommended)
python spacewar_ai/main.py --algorithm ppo --episodes 10000 --render

# Train with DQN
python spacewar_ai/main.py --algorithm dqn --episodes 5000

# Train without rendering (faster)
python spacewar_ai/main.py --algorithm ppo --episodes 10000
```

### With LLM-Guided Exploration

```bash
# Create API key file
echo "sk-your-key-here" > openai.key.txt

# Or use .env file
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Train with LLM guidance (uses all available providers)
python spacewar_ai/main.py --algorithm ppo --use-llm

# Or specify a provider
python spacewar_ai/main.py --algorithm ppo --use-llm --llm-provider anthropic
```

### With Local LLM (llama.cpp)

```bash
# Install llama-cpp-python for your hardware:
# CPU only:
pip install llama-cpp-python huggingface_hub

# NVIDIA GPU:
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# AMD GPU (ROCm):
CMAKE_ARGS="-DLLAMA_HIPBLAS=on" pip install llama-cpp-python

# Train with local LLM (auto-downloads best model for your hardware)
python spacewar_ai/main.py --algorithm ppo --use-llm --llm-local

# Or specify a model path
python spacewar_ai/main.py --algorithm ppo --use-llm --llm-local --llm-local-model ./models/my-model.gguf
```

The system auto-detects your hardware and downloads the best fitting model:
- 86GB VRAM: Qwen 2.5 72B Q5 (quality score: 96)
- 24GB VRAM: Qwen 2.5 32B Q6 (quality score: 90)
- 8GB VRAM: Qwen 2.5 7B Q8 (quality score: 78)
- CPU only: Qwen 2.5 3B (quality score: 65)

### Evaluation

```bash
# Auto-detect best model (both players use trained model)
python spacewar_ai/main.py --evaluate --render

# Or specify a checkpoint
python spacewar_ai/main.py --evaluate --checkpoint checkpoints/ppo_best_model.pt --render
```

Checkpoints are named by algorithm: `ppo_best_model.pt`, `dqn_best_model.pt`, etc.

### Tournament Mode (Best of the Best)

Pit trained agents against each other to find the champion:

```bash
# Train multiple models first
python spacewar_ai/main.py --algorithm ppo --episodes 5000
python spacewar_ai/main.py --algorithm dqn --episodes 5000

# Run tournament between all trained agents
python spacewar_ai/main.py --tournament --render

# Different tournament formats
python spacewar_ai/main.py --tournament --tournament-mode round_robin   # Everyone plays everyone
python spacewar_ai/main.py --tournament --tournament-mode swiss         # Similar ratings play
python spacewar_ai/main.py --tournament --tournament-mode elimination   # Single elimination bracket

# Adjust matches per pairing
python spacewar_ai/main.py --tournament --matches-per-pairing 10
```

### LLM Arena (LLM vs LLM)

Have different LLMs compete against each other:

```bash
# All available LLMs compete (needs API keys)
python spacewar_ai/main.py --llm-arena --render

# Include local LLM in the competition
python spacewar_ai/main.py --llm-arena --include-local-llm --render

# Swiss format with more matches
python spacewar_ai/main.py --llm-arena --tournament-mode swiss --matches-per-pairing 10
```

Each LLM plays as a different player, and an ELO rating system tracks performance.

### Iterative Strategy Training (Eureka-style)

LLM generates and refines strategies based on training feedback:

```bash
# Use local LLM for iterative training
python spacewar_ai/main.py --iterative --local-llm-only --algorithm ppo --episodes 10000

# Use cloud APIs (automatically fails over to working provider)
python spacewar_ai/main.py --iterative --algorithm ppo --episodes 10000

# Customize iterations
python spacewar_ai/main.py --iterative --iterations 20 --episodes-per-iteration 1000
```

The iterative trainer:
- Generates initial strategy from LLM
- Trains for N episodes, measures performance
- Asks LLM to refine strategy based on results
- Repeats until target win rate or max iterations

### Champion Training

Train against the strongest opponents from previous tournaments:

```bash
# Train against all available opponents (trained agents + LLMs)
python spacewar_ai/main.py --champion-train --use-llm --episodes 5000

# Include local LLM as opponent
python spacewar_ai/main.py --champion-train --use-llm --include-local-llm --episodes 5000
```

The champion trainer:
- Loads ELO ratings from previous tournaments
- Prioritizes training against stronger opponents
- Produces models that beat the best of the best

### Complete Training Pipeline

For the absolute best model:

```bash
# Step 1: Train PPO and DQN models
python spacewar_ai/main.py --algorithm ppo --episodes 10000
python spacewar_ai/main.py --algorithm dqn --episodes 10000

# Step 2: Run tournament to find the best algorithm
python spacewar_ai/main.py --tournament

# Step 3: Run LLM arena to benchmark against LLMs
python spacewar_ai/main.py --llm-arena

# Step 4: Champion training against all top performers
python spacewar_ai/main.py --champion-train --use-llm --include-local-llm --episodes 10000

# Step 5: Final tournament with champion
python spacewar_ai/main.py --tournament
```

## Project Structure

```
ai_space_war/
├── spacewar_ai/
│   ├── config.py           # All hyperparameters
│   ├── environment.py      # Game physics and Gymnasium environment
│   ├── agents.py           # DQN and PPO implementations
│   ├── llm_integration.py  # API key management, LLM clients
│   ├── tournament.py       # Tournament system, ELO ratings, arenas
│   └── main.py             # Training loop and CLI
├── checkpoints/            # Saved models and tournament ratings
├── models/                 # Downloaded local LLM models (auto-created)
├── requirements.txt
├── .gitignore
├── ARCHITECTURE.md         # Detailed architecture with diagrams
├── CLAUDE.md               # Quick reference documentation
└── VIBE_HISTORY.md         # Development history and decisions
```

## Controls (for reference)

| Action | Description |
|--------|-------------|
| 0 | No action (drift) |
| 1 | Rotate left |
| 2 | Rotate right |
| 3 | Thrust forward |
| 4 | Fire torpedo |
| 5 | Thrust + Fire |

## API Key Setup

The system discovers API keys from multiple sources (in priority order):

1. `*.key.txt` files (e.g., `openai.key.txt`, `xai.key.txt`)
2. `.env` file
3. Environment variables

Supported providers and their key file names:
- `anthropic.key.txt` or `claude.key.txt` - Anthropic (Claude)
- `openai.key.txt` or `gpt.key.txt` - OpenAI (GPT)
- `google.key.txt` or `gemini.key.txt` - Google (Gemini)
- `xai.key.txt` or `grok.key.txt` - xAI (Grok)
- `groq.key.txt` - Groq
- `together.key.txt` - Together AI
- `mistral.key.txt` - Mistral
- `deepseek.key.txt` - DeepSeek
- `openrouter.key.txt` - OpenRouter

For `--iterative` mode, the system automatically tries each provider until one works (handles billing/quota failures gracefully).

## License

MIT
