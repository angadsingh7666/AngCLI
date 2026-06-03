import subprocess
from pathlib import Path

TOOLS_SCHEMA = [
    {"name": "read_file", "description": "Read file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write/overwrite file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "run_shell", "description": "Execute shell command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}
]

def read_file(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else f"Error: '{path}' not found."

def write_file(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"✅ Written to {path}"

def run_shell(command: str) -> str:
    try:
        res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        out, err = res.stdout.strip(), res.stderr.strip()
        return out if res.returncode == 0 and out else f"Exit {res.returncode}\nSTDERR:\n{err}" if err else "Command succeeded"
    except subprocess.TimeoutExpired: return "⏱️ Timeout (30s)"
    except Exception as e: return f"❌ {e}"

TOOL_REGISTRY = {"read_file": read_file, "write_file": write_file, "run_shell": run_shell}