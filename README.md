# PyTorch Coding Agent

An autonomous AI coding agent powered by DeepSeek-R1-Distill-Qwen-7B, built with PyTorch and Hugging Face transformers. This project enables code generation, file manipulation, and shell command execution through a conversational interface.

## Features

- 🤖 **Autonomous Coding Agent**: Solves coding tasks using reasoning and tool usage
- 🛠️ **Built-in Tools**: Read/write files, execute shell commands
- 📊 **Context Tracking**: Real-time monitoring of token usage and VRAM consumption
- 🔒 **Safe Mode**: Optional restriction on shell command execution
- 🎯 **Fine-tuning Support**: Complete SFT (Supervised Fine-Tuning) pipeline with QLoRA
- 💾 **4-bit Quantization**: Efficient memory usage with BitsAndBytes integration

## Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support (required for inference and training)
- At least 8GB VRAM recommended for 4-bit inference

## Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch>=2.1.0 transformers>=4.40.0 accelerate>=0.30.0 \
    bitsandbytes>=0.43.0 huggingface-hub>=0.22.0 trl>=0.9.0 \
    datasets>=2.18.0 peft>=0.11.0 typer>=0.12.0 rich>=13.7.0
```

## Usage

### Interactive Mode

Start an interactive session with the coding agent:

```bash
codeagent interactive --model "./models/DeepSeek-R1-Distill-Qwen-7B"
```

Options:
- `--model`: Path to model directory (default: `./models/DeepSeek-R1-Distill-Qwen-7B`)
- `--safe`: Enable safe mode (disables shell commands)
- `--fp16`: Use full precision instead of 4-bit quantization

### Single Query Mode

Run a single task:

```bash
codeagent ask "Write a Python script to sort a list of dictionaries by key" --model "./models/DeepSeek-R1-Distill-Qwen-7B"
```

Options:
- `--max-turns`: Maximum conversation turns (default: 10)
- `--max-tokens`: Maximum tokens per response (default: 4096)

## Available Tools

The agent has access to the following tools:

| Tool | Description |
|------|-------------|
| `read_file` | Read content from a file |
| `write_file` | Write content to a file |
| `run_shell` | Execute shell commands (disabled in safe mode) |

## Model Setup

The model is downloaded to the `./models` directory using the download script:

```bash
python scripts/download_model.py deepseek-ai/DeepSeek-R1-Distill-Qwen-7B DeepSeek-R1-Distill-Qwen-7B
```

This will download the model to `./models/DeepSeek-R1-Distill-Qwen-7B`.

## Fine-tuning

### Dataset Preparation

Format your dataset as JSONL with chat messages:

```json
{"messages": [{"role": "user", "content": "Task"}, {"role": "assistant", "content": "Solution"}]}
```

Use the formatting script:

```bash
python scripts/format_dataset.py --input raw_data.jsonl --output formatted_data.jsonl
```

### Training

Run SFT training with QLoRA:

```bash
python scripts/train.py train \
    --model-path "./models/DeepSeek-R1-Distill-Qwen-7B" \
    --train-data "data/train/sft_train.jsonl" \
    --val-data "data/val/sft_val.jsonl" \
    --output-dir "outputs/checkpoints" \
    --merge-dir "outputs/final_models" \
    --epochs 2 \
    --batch-size 2 \
    --grad-accum 8 \
    --lr 1e-4 \
    --qlora
```

Training options:
- `--qlora/--no-qlora`: Enable 4-bit QLoRA training
- `--lora-r`: LoRA rank (default: 32)
- `--lora-alpha`: LoRA alpha (default: 64)
- `--max-seq-len`: Maximum sequence length (default: 4096)
- `--eval-steps`: Evaluation frequency (default: 100)

## Project Structure

```
├── src/
│   └── agent_cli/
│       ├── main.py          # CLI entry point
│       ├── agent.py         # Core agent logic
│       ├── inference.py     # Model inference engine
│       └── tools.py         # Tool definitions and registry
├── scripts/
│   ├── download_model.py    # Model download utility
│   ├── download_dataset.py  # Dataset download utility
│   ├── format_dataset.py    # Dataset formatting
│   ├── split_dataset.py     # Train/val splitting
│   └── train.py            # SFT training script
├── pyproject.toml
└── README.md
```

## Architecture

The agent follows a think-act loop:

1. **Think**: The model reasons about the task in `<think>` tags
2. **Act**: The model outputs either a final answer or a JSON tool call
3. **Execute**: Tools are executed and results are fed back to the model
4. **Repeat**: Continue until the task is complete or max turns reached

## Safety Considerations

- Enable `--safe` mode to prevent shell command execution
- Review tool outputs before trusting agent responses
- Run in isolated environments for untrusted tasks

## License

MIT License

## Contributing

Contributions are welcome! Please open issues or submit pull requests.
