# Finetune Personas

Fine-tune an LLM to embody a specific persona. Uses Yoda as the example — generates synthetic training data with Claude, then fine-tunes Qwen 2.5 1.7B into a Yoda-speaking model.

## How It Works

1. **Generate training data** — Claude creates 100 user-assistant conversation pairs (5 variations across 20 topics) where the assistant responds as Yoda
2. **Split data** — 90/10 train/validation split
3. **Fine-tune** — Train Qwen 2.5 1.7B on the generated data
4. **Export** — Output as a 1.7GB GGUF for local inference

### The Yoda Persona

The system prompt instructs Claude to generate responses with:
- Inverted sentence structure ("Strong in the Force, you are")
- Force/Jedi references woven naturally into answers
- Verbal tics ("Hmmm", "Yes, yes")
- Wisdom-based guidance style

Topics span technical (ML, databases, APIs, recursion) and everyday (cooking, learning, motivation) to build a well-rounded persona.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key.

```bash
uv sync
export ANTHROPIC_API_KEY=your_key
```

## Usage

```bash
# Generate synthetic training data via Claude
uv run python generate_data.py

# Split into train/validation sets
uv run python split_data.py
```

Fine-tuning is done externally (e.g., via MLX or another training framework). The output `yoda.gguf` can be loaded into LM Studio or any GGUF-compatible runtime.

## Project Structure

```
├── generate_data.py      # Claude-powered synthetic data generation
├── split_data.py         # Train/validation split
├── training_data.jsonl   # Generated training dataset (100 examples)
├── yoda.gguf             # Fine-tuned model (1.7GB, GGUF format)
└── pyproject.toml        # Dependencies
```

## Tech Stack

- **Anthropic SDK** — Claude API for synthetic data generation
- **MLX-LM** — inference with fine-tuned model
- **GGUF** — quantized model format
- **SentencePiece** — tokenization
- **PyTorch** — model training

## Related

See [yoda_evals](../yoda_evals) for the evaluation framework that measures how well this fine-tune actually works.
