import typer
from .agent import CodingAgent

app = typer.Typer(help="🤖 PyTorch Coding Agent (GPU-ONLY | Qwen3-Coder-3B)")

@app.command()
def ask(
    query: str = typer.Argument(..., help="Task to solve"),
    model: str = typer.Option("./models/Qwen3-Coder-3B", "--model", help="HF repo ID or local path"),
    safe: bool = typer.Option(False, "--safe"),
    turns: int = typer.Option(10, "--max-turns"),
    full_precision: bool = typer.Option(False, "--fp16"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
):
    # FIX: Removed max_new_tokens=max_tokens from run()
    agent = CodingAgent(model, safe_mode=safe, max_turns=turns, use_4bit=not full_precision)
    agent.run(query)

@app.command()
def interactive(
    model: str = typer.Option("./models/Qwen3-Coder-3B"),
    safe: bool = typer.Option(False),
    full_precision: bool = typer.Option(False, "--fp16"),
    max_tokens: int = typer.Option(4096, "--max-tokens"),
):
    typer.echo("🧠 Starting... Type 'exit' to quit.\n")
    # FIX: Removed max_new_tokens=max_tokens from run()
    agent = CodingAgent(model, safe_mode=safe, use_4bit=not full_precision)
    while True:
        q = typer.prompt("👤 You")
        if q.lower() in ("exit", "quit"): break
        agent.run(q)
        typer.echo("\n" + "─" * 40 + "\n")
        
if __name__ == "__main__":
    app()