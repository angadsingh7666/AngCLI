import re
import json
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Assuming these are imported from your local modules
from .inference import HFInferenceEngine
from .tools import TOOLS_SCHEMA, TOOL_REGISTRY

console = Console()

SYSTEM_BASE = """You are an autonomous coding agent. Solve tasks by reading, writing, and executing code.
RULES:
- Always verify files before modifying
- Use tools via valid JSON: {"name": "tool_name", "arguments": {...}}
- Return final answer when complete"""

class CodingAgent:
    def __init__(self, model_id: str, safe_mode: bool = False, max_turns: int = 10, use_4bit: bool = True, max_context_limit: int = 10000):
        self.engine = HFInferenceEngine(model_id, use_4bit=use_4bit)
        self.safe_mode = safe_mode
        self.max_turns = max_turns
        self.max_context_limit = max_context_limit
        self.active_tools = [t for t in TOOLS_SCHEMA if not (safe_mode and t["name"] == "run_shell")]
        
        # 🔓 EXPOSED FOR TRAINING/OPTIMIZATION
        self.model = self.engine.model
        self.tokenizer = self.engine.tokenizer

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Robust JSON extraction (handles markdown, single dicts vs lists, trailing commas)."""
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```|(\{.*\})', text, re.DOTALL)
        raw = (json_match.group(1) or json_match.group(2) or text).strip()
        raw = re.sub(r',\s*([}\]])', r'\1', raw)  # Fix trailing commas
        
        try:
            parsed = json.loads(raw)
            # Small models often output a single dict instead of a list of dicts
            if isinstance(parsed, dict):
                return [parsed]
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            return []

    def _render_context_tracker(self, stats: Dict[str, int]):
        """Renders a beautiful Rich context tracker to the console."""
        total_tokens = stats['total_tokens']
        percentage = min((total_tokens / self.max_context_limit) * 100, 100.0)
        
        if percentage < 60:
            color = "green"
            status = "Healthy"
        elif percentage < 85:
            color = "yellow"
            status = "Getting Full"
        else:
            color = "red"
            status = "DANGER: Near OOM"
            
        # Create a simple text-based progress bar
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
            tracker_text += f"\n[bold red]⚠️ WARNING: Nearing VRAM limit! Agent may crash on next turn.[/bold red]"
            
        console.print(Panel(tracker_text, title="📊 Context Window", border_style=color, expand=False))

    def run(self, query: str):
        tool_desc = json.dumps(self.active_tools, indent=2)
        messages = [
            {"role": "system", "content": f"{SYSTEM_BASE}\n\nAvailable Tools:\n{tool_desc}"},
            {"role": "user", "content": query}
        ]
        console.print(Panel(f"[bold cyan]{query}[/bold cyan]", title="📝 Task", border_style="cyan"))

        for turn in range(1, self.max_turns + 1):
            console.print(f"\n[dim]🔄 Turn {turn}/{self.max_turns}[/dim]")
            
            # 🔄 FIX: Unpack the tuple returned by the updated engine
            try:
                response_text, stats = self.engine.generate(messages)
            except Exception as e:
                console.print(Panel(f"[bold red]❌ Generation failed (likely OOM or CUDA error):\n{e}[/bold red]", title="Fatal Error", border_style="red"))
                return

            messages.append({"role": "assistant", "content": response_text})
            
            # 📊 Render Context Tracker
            self._render_context_tracker(stats)

            tool_calls = self._parse_tool_calls(response_text)
            
            # If no tools were called, the agent is done
            if not tool_calls:
                console.print(Panel(Markdown(response_text), title="✅ Final Answer", border_style="green"))
                return

            # Process Tool Calls
            for tc in tool_calls:
                name = tc.get("name", "")
                args = tc.get("arguments", {})
                
                # Truncate args for console display so it doesn't spam the terminal
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
                
                # Truncate tool output for console display
                display_result = result_str[:250] + ('...' if len(result_str) > 250 else '')
                console.print(Panel(display_result, title=f"📤 Tool Output: {name}", border_style="dim", expand=False))

        console.print("\n[bold yellow]⚠️ Max turns reached. Agent stopped.[/bold yellow]")