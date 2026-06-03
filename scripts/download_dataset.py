import os
import typer
import requests
from pathlib import Path
from datasets import load_dataset

# Dynamically resolve the 'data' directory relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

app = typer.Typer(help="📥 Step 1: Download raw datasets to data/uncurated/")

@app.command()
def download(
    source: str = typer.Argument(..., help="URL or HF dataset ID (e.g., 'https://example.com/data.csv' or 'HuggingFaceH4/no_robots')"),
    filename: str = typer.Option("raw_data.jsonl", help="Output filename in uncurated folder"),
    hf_split: str = typer.Option("train", help="Dataset split to download (for HF datasets only)")
):
    output_dir = DATA_DIR / "uncurated"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    if source.startswith("http://") or source.startswith("https://"):
        print(f"🌐 Downloading from URL: {source}")
        response = requests.get(source, stream=True)
        response.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ Downloaded successfully to {output_path}")
    else:
        print(f"🤗 Downloading Hugging Face dataset: '{source}' (split: {hf_split})")
        dataset = load_dataset(source, split=hf_split)
        dataset.to_json(str(output_path), orient="records", lines=True)
        print(f"✅ Saved HF dataset to {output_path}")

if __name__ == "__main__":
    app()