# PyTorch Coding Agent Elite v2.0

An **enterprise-grade**, **security-hardened** autonomous AI coding agent powered by state-of-the-art language models. Built with PyTorch and Hugging Face transformers, this agent represents a significant evolution over basic coding assistants with advanced security, self-reflection, and comprehensive tooling capabilities.

## 🚀 Key Improvements Over Standard Agents

### Security Enhancements (Critical)
- **Path Validation**: All file operations validated against allowed directories
- **Command Filtering**: Dangerous shell commands automatically blocked
- **Symlink Protection**: Prevents symlink-based directory traversal attacks
- **Audit Logging**: Complete audit trail of all tool executions
- **Sandboxed Python Execution**: Restricted Python interpreter with safe imports only
- **File Size Limits**: Prevents memory exhaustion from large files
- **Timeout Controls**: Configurable timeouts for all operations

### New Advanced Features
- **Self-Reflection**: Agent analyzes its own performance and adjusts strategy
- **Error Recovery**: Automatic retry with alternative approaches on failures
- **Task Completion Detection**: Intelligent assessment of task completion
- **Enhanced JSON Parsing**: Multi-strategy JSON extraction with error recovery
- **Performance Metrics**: Detailed execution statistics and analytics
- **Thought Visualization**: See the agent's reasoning process in real-time
- **State Management**: Reset capability for clean task transitions

### Expanded Tool Set (11 Tools vs 3)
| Tool | Description | Security Features |
|------|-------------|-------------------|
| `read_file` | Read file content | Path validation, size limits, line limits |
| `write_file` | Write/create files | Path validation, append mode support |
| `delete_file` | Safe file deletion | Path validation, no directory deletion |
| `list_directory` | List directory contents | Recursive option, glob filtering |
| `search_files` | Find files by pattern | Glob pattern matching |
| `grep_search` | Search text in files | Regex support, case sensitivity options |
| `copy_file` | Copy files | Dual path validation |
| `move_file` | Move/rename files | Dual path validation |
| `run_shell` | Execute shell commands | Command filtering, timeout control |
| `run_python` | Execute Python code | Sandboxed execution, import restrictions |
| `create_directory` | Create directories | Path validation |

## 📦 Installation

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install torch>=2.1.0 transformers>=4.40.0 accelerate>=0.30.0 \
    bitsandbytes>=0.43.0 huggingface-hub>=0.22.0 trl>=0.9.0 \
    datasets>=2.18.0 peft>=0.11.0 typer>=0.12.0 rich>=13.7.0
```

## ⚡ Requirements

- Python >= 3.10
- NVIDIA GPU with CUDA support
- 8GB+ VRAM recommended for 4-bit inference
- 16GB+ VRAM for full precision models

## 🎯 Usage

### Single Query Mode

```bash
# Basic usage
codeagent ask "Create a Python script that sorts a list of dictionaries"

# With options
codeagent ask "Refactor the code in src/" \
    --model "./models/Qwen3-Coder-3B" \
    --safe \
    --max-turns 15 \
    --temperature 0.2
```

### Interactive Mode

```bash
codeagent interactive --model "./models/Qwen3-Coder-3B" --safe
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model path or HF repo ID | `./models/Qwen3-Coder-3B` |
| `--safe, -s` | Enable safe mode (no shell) | `False` |
| `--max-turns, -t` | Maximum conversation turns | `10` |
| `--max-tokens` | Maximum context tokens | `4096` |
| `--temperature` | Sampling temperature | `0.1` |
| `--fp16` | Use FP16 instead of 4-bit | `False` |
| `--no-reflection` | Disable self-reflection | `False` |
| `--no-recovery` | Disable error recovery | `False` |
| `--quiet, -q` | Reduce output verbosity | `False` |

### Additional Commands

```bash
# List all available tools
codeagent tools

# Show version info
codeagent version
```

## 🔒 Security Architecture

### Allowed Directories
By default, the agent can only operate within the current working directory. This prevents:
- Access to system files (`/etc/`, `/root/`, etc.)
- Directory traversal attacks (`../`)
- Symlink-based escapes

### Blocked Commands
The following are automatically blocked:
- Destructive commands: `rm -rf`, `dd if=`, `mkfs`, `fdisk`
- Network tools: `wget`, `curl` (prevent data exfiltration)
- Privilege escalation: `sudo`, `su`, `chmod`, `chown`
- Subshell execution: `$()`, backticks
- System directory writes: `> /etc/`, `> /root/`

### Audit Trail
All tool executions are logged with:
- Timestamp
- Tool name and arguments (hashed)
- Success/failure status
- Result preview

## 🧠 Agent Architecture

### Think-Act Loop
1. **Think**: Model reasons about the task in `` tags
2. **Plan**: Determines required tools and approach
3. **Act**: Executes tool calls in JSON format
4. **Observe**: Analyzes tool outputs
5. **Reflect**: Evaluates success and adjusts strategy
6. **Repeat**: Continues until task complete or max turns

### Self-Improvement
When errors occur, the agent:
1. Logs the failure with context
2. Triggers self-reflection mode
3. Receives hints for improvement
4. Attempts alternative approaches
5. Tracks consecutive failures to prevent loops

## 📊 Performance Metrics

The agent tracks:
- Total turns used
- Tool call success rate
- Token consumption
- Execution time
- Error history
- Context window usage

Access metrics programmatically:
```python
agent = CodingAgent(model_id)
agent.run(query)
metrics = agent.get_metrics()
print(f"Success rate: {metrics['successful_tool_calls']}/{metrics['total_tool_calls']}")
```

## 🛠️ Fine-tuning Support

Complete SFT (Supervised Fine-Tuning) pipeline with QLoRA:

```bash
# Format dataset
python scripts/format_dataset.py --input raw_data.jsonl --output formatted.jsonl

# Train with QLoRA
python scripts/train.py train \
    --model-path "./models/Qwen3-Coder-3B" \
    --train-data "data/train/sft_train.jsonl" \
    --val-data "data/val/sft_val.jsonl" \
    --output-dir "outputs/checkpoints" \
    --qlora --epochs 2 --batch-size 2
```

## 📁 Project Structure

```
├── src/agent_cli/
│   ├── main.py          # CLI entry point with enhanced commands
│   ├── agent.py         # Core agent with self-reflection & recovery
│   ├── inference.py     # Model inference engine with limits
│   └── tools.py         # 11 secure tools with audit logging
├── scripts/
│   ├── download_model.py
│   ├── download_dataset.py
│   ├── format_dataset.py
│   ├── split_dataset.py
│   └── train.py
├── pyproject.toml
└── README.md
```

## 🔬 Comparison with Other Agents

| Feature | Coding Agent Elite | Claude Code | Standard Agents |
|---------|-------------------|-------------|-----------------|
| Security Validation | ✅ Comprehensive | ⚠️ Basic | ❌ None |
| Audit Logging | ✅ Full trail | ⚠️ Limited | ❌ None |
| Self-Reflection | ✅ Built-in | ⚠️ Prompt-based | ❌ None |
| Error Recovery | ✅ Automatic | ⚠️ Manual | ❌ None |
| Tool Count | ✅ 11 tools | ⚠️ ~5 tools | ⚠️ ~3 tools |
| Python Sandbox | ✅ Restricted | ❌ No | ❌ No |
| Performance Metrics | ✅ Detailed | ❌ No | ❌ No |
| Open Source | ✅ Yes | ❌ No | ✅ Yes |

## ⚠️ Safety Considerations

1. **Always use `--safe` mode** for untrusted tasks
2. **Review tool outputs** before trusting agent responses
3. **Run in isolated environments** (Docker, VM) for production
4. **Monitor audit logs** regularly
5. **Set appropriate timeouts** based on task complexity
6. **Limit context window** to prevent memory issues

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional secure tools
- Enhanced self-reflection algorithms
- Multi-agent collaboration
- Plugin architecture
- Web UI interface

## 📄 License

MIT License

## 🙏 Acknowledgments

Built with:
- Hugging Face Transformers
- PyTorch
- BitsAndBytes quantization
- Rich CLI framework
- Typer for command-line interfaces
