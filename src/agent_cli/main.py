import typer
from typing import Optional
from .agent import CodingAgent

app = typer.Typer(
    help="🤖 PyTorch Coding Agent Elite - Enhanced AI Coding Assistant",
    rich_markup_mode="rich"
)

@app.command()
def ask(
    query: str = typer.Argument(..., help="Task to solve"),
    model: str = typer.Option("./models/Qwen3-Coder-3B", "--model", "-m", help="Model path or HF repo ID"),
    safe: bool = typer.Option(False, "--safe", "-s", help="Enable safe mode (disables shell commands)"),
    turns: int = typer.Option(10, "--max-turns", "-t", help="Maximum conversation turns"),
    full_precision: bool = typer.Option(False, "--fp16", help="Use FP16 instead of 4-bit quantization"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum context tokens"),
    temperature: float = typer.Option(0.1, "--temperature", help="Sampling temperature"),
    no_reflection: bool = typer.Option(False, "--no-reflection", help="Disable self-reflection"),
    no_recovery: bool = typer.Option(False, "--no-recovery", help="Disable error recovery"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Reduce output verbosity"),
):
    """Execute a single task query."""
    agent = CodingAgent(
        model, 
        safe_mode=safe, 
        max_turns=turns, 
        use_4bit=not full_precision,
        max_context_limit=max_tokens,
        enable_self_reflection=not no_reflection,
        enable_error_recovery=not no_recovery,
        temperature=temperature,
        verbose=not quiet
    )
    agent.run(query)

@app.command()
def interactive(
    model: str = typer.Option("./models/Qwen3-Coder-3B", "--model", "-m"),
    safe: bool = typer.Option(False, "--safe", "-s"),
    full_precision: bool = typer.Option(False, "--fp16"),
    max_tokens: int = typer.Option(4096, "--max-tokens", "-t"),
    temperature: float = typer.Option(0.1, "--temperature"),
    turns: int = typer.Option(15, "--max-turns", help="Max turns per query"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
):
    """Start an interactive chat session with the agent."""
    typer.echo("🧠 [bold cyan]PyTorch Coding Agent Elite[/bold cyan]")
    typer.echo("Type 'exit', 'quit', or Ctrl+C to quit.\n")
    
    agent = CodingAgent(
        model, 
        safe_mode=safe, 
        use_4bit=not full_precision,
        max_context_limit=max_tokens,
        temperature=temperature,
        max_turns=turns,
        verbose=not quiet
    )
    
    try:
        while True:
            q = typer.prompt("👤 You")
            if q.lower() in ("exit", "quit"): 
                typer.echo("[bold green]Goodbye![/bold green]")
                break
            agent.run(q)
            typer.echo("\n" + "─" * 40 + "\n")
    except KeyboardInterrupt:
        typer.echo("\n[bold yellow]Session interrupted.[/bold yellow]")

@app.command()
def tools():
    """List all available tools and their descriptions."""
    from .tools import TOOLS_SCHEMA
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    table = Table(title="🛠️ Available Tools", show_header=True, header_style="bold magenta")
    table.add_column("Tool Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Parameters", style="yellow")
    
    for tool in TOOLS_SCHEMA:
        params = ", ".join(tool["parameters"]["properties"].keys())
        table.add_row(tool["name"], tool["description"], params)
    
    console.print(table)

@app.command()
def version():
    """Show version information."""
    typer.echo("🤖 PyTorch Coding Agent Elite v2.0.0")
    typer.echo("Enhanced with security, self-reflection, and advanced tooling")

if __name__ == "__main__":
    app()
