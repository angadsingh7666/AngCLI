import os
import json
import pandas as pd
import typer
from pathlib import Path

# Dynamically resolve the 'data' directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

app = typer.Typer(help="🛠️ Step 2: Format uncurated data into SFT chat format in data/curated/")

@app.command()
def format_data(
    input_path: str = typer.Argument(str(DATA_DIR / "uncurated" / "raw_data.jsonl"), help="Path to uncurated data (CSV, JSON, JSONL, or TXT)"),
    output_filename: str = typer.Option("formatted_data.jsonl", help="Output filename in curated folder"),
    instruction_col: str = typer.Option("instruction", help="Column name for user instruction/prompt"),
    response_col: str = typer.Option("response", help="Column name for assistant response"),
    input_col: str = typer.Option(None, help="Optional column name for additional input/context"),
    system_prompt: str = typer.Option("You are a helpful AI assistant.", help="Optional system prompt to prepend")
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    output_dir = DATA_DIR / "curated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename
    
    print(f"📖 Reading data from: {input_path}")
    ext = input_path.suffix.lower()
    
    # Load data based on extension
    if ext == ".csv":
        df = pd.read_csv(input_path)
    elif ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    elif ext == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    elif ext == ".txt":
        print("⚠️ TXT format detected. Assuming tab-separated values (instruction \\t response).")
        df = pd.read_csv(input_path, sep="\t", header=None, names=[instruction_col, response_col])
    else:
        raise ValueError(f"Unsupported file extension: {ext}. Please use CSV, JSON, JSONL, or TXT.")

    print(f"📊 Loaded {len(df)} rows. Columns: {list(df.columns)}")

    # Validate columns
    if instruction_col not in df.columns:
        raise ValueError(f"Instruction column '{instruction_col}' not found. Available: {list(df.columns)}")
    if response_col not in df.columns:
        raise ValueError(f"Response column '{response_col}' not found. Available: {list(df.columns)}")

    print(f"🔄 Formatting to SFT chat template...")
    formatted_data = []
    
    for idx, row in df.iterrows():
        instruction = str(row[instruction_col]).strip()
        response = str(row[response_col]).strip()
        
        # Skip empty or NaN rows
        if not instruction or not response or instruction.lower() in ["nan", "none", ""]:
            continue
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
            
        user_content = instruction
        if input_col and input_col in df.columns and pd.notna(row[input_col]):
            user_content += f"\n\nInput:\n{str(row[input_col]).strip()}"
            
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": response})
        
        formatted_data.append({"messages": messages})

    # Write to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Successfully formatted and saved {len(formatted_data)} samples to {output_path}")

if __name__ == "__main__":
    app()