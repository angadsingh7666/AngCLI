import re
import json
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .inference import HFInferenceEngine
from .tools import TOOLS_SCHEMA, TOOL_REGISTRY

console = Console()


SYSTEM_BASE = """You are Qwen-Coder. Before answering or using a tool, you MUST think step-by-step.

RULES:
1. First, output your reasoning inside `` tags. Explain what the user wants and what tools you need.
2. After the  tags, output your final answer OR the JSON tool call.
3. DO NOT output markdown or code blocks unless calling a tool.

EXAMPLE:

The user is asking who I am. I should just answer briefly. I do not need to use any tools.
</think>

I am Qwen-Coder, an autonomous AI coding assistant.

EXAMPLE 2:

The user wants to read 'test.txt'. I need to use the 'read_file' tool.
</think>

{"name": "read_file", "arguments": {"path": "test.txt"}}
"""

class CodingAgent:
    def __init__(self, model_id: str, safe_mode: bool = False, max_turns: int = 10, use_4bit: bool = True, max_context_limit: int = 10000):
        self.engine = HFInferenceEngine(model_id, use_4bit=use_4bit)
        self.safe_mode = safe_mode
        self.max_turns = max_turns
        self.max_context_limit = max_context_limit
        self.active_tools = [t for t in TOOLS_SCHEMA if not (safe_mode and t["name"] == "run_shell")]
        
        self.model = self.engine.model
        self.tokenizer = self.engine.tokenizer

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Robust JSON extraction."""
        pattern = r'```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})'
        matches = re.findall(pattern, text, re.DOTALL)
        
        tool_calls = []
        for match in matches:
            raw = (match[0] or match[1]).strip()
            raw = re.sub(r',\s*([}\]])', r'\1', raw) 
            
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    tool_calls.append(parsed)
                elif isinstance(parsed, list):
                    tool_calls.extend(parsed)
            except json.JSONDecodeError:
                continue
                
        return tool_calls

    def _render_context_tracker(self, stats: Dict[str, int]):
        total_tokens = stats['total_tokens']
        percentage = min((total_tokens / self.max_context_limit) * 100, 100.0)
        
        if percentage < 60: color, status = "green", "Healthy"
        elif percentage < 85: color, status = "yellow", "Getting Full"
        else: color, status = "red", "DANGER: Near OOM"
            
        bar_length = 30
        filled = int(bar_length * percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        tracker_text = (
            f"[bold {color}]Status: {status}[/bold {color}]\n"
            f"Input : {stats['input_tokens']:,} tokens\n"
            f"Output: {stats['output_tokens']:,} tokens\n"
            f"Total : {total_tokens:,} / {self.max_context_limit:,} tokens\n"
            f"VRAM  : [{color}]{bar}[/{color}] {percentage:.1f}%"
        )
        
        if percentage > 85:
            tracker_text += f"\n[bold red]⚠️ WARNING: Nearing VRAM limit![/bold red]"
            
        console.print(Panel(tracker_text, title="📊 Context Window", border_style=color, expand=False))

    def run(self, query: str, max_new_tokens: int = 1024):
        tool_desc = json.dumps(self.active_tools, indent=2)
        messages = [
            {"role": "system", "content": f"{SYSTEM_BASE}\n\nAvailable Tools:\n{tool_desc}"},
            {"role": "user", "content": query}
        ]
        console.print(Panel(f"[bold cyan]{query}[/bold cyan]", title="📝 Task", border_style="cyan"))

        for turn in range(1, self.max_turns + 1):
            console.print(f"\n[dim]🔄 Turn {turn}/{self.max_turns}[/dim]")
            
            try:
                response_text, stats = self.engine.generate(messages, max_new_tokens=max_new_tokens)
            except Exception as e:
                console.print(Panel(f"[bold red]❌ Generation failed:\n{e}[/bold red]", title="Fatal Error", border_style="red"))
                return

            messages.append({"role": "assistant", "content": response_text})
            self._render_context_tracker(stats)

            tool_calls = self._parse_tool_calls(response_text)
            
            if not tool_calls:
                console.print(Panel(Markdown(response_text), title="✅ Final Answer", border_style="green"))
                return

            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                
                args_str = json.dumps(args)
                display_args = args_str[:100] + ('...' if len(args_str) > 100 else '')
                console.print(f"[bold magenta]🛠️ Executing:[/bold magenta] {name}({display_args})")
                
                if name in TOOL_REGISTRY:
                    try:
                        result = TOOL_REGISTRY[name](**args)
                        result_str = str(result)
                    except Exception as e:
                        result_str = f"❌ Error executing tool: {e}"
                else:
                    result_str = f"❌ Unknown tool: {name}"

                messages.append({"role": "tool", "content": f"Tool '{name}' output:\n{result_str}"})
                
                display_result = result_str[:250] + ('...' if len(result_str) > 250 else '')
                console.print(Panel(display_result, title=f"📤 Tool Output: {name}", border_style="dim", expand=False))

        console.print("\n[bold yellow]⚠️ Max turns reached. Agent stopped.[/bold yellow]")