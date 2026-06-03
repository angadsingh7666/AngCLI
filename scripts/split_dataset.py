import os
import typer
from pathlib import Path
from datasets import load_dataset

# Dynamically resolve the 'data' directory
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR / "data"

app = typer.Typer(help="✂️ Step 3: Split curated data into train/val sets")

@app.command()
def split_data(
    input_path: str = typer.Argument(str(DATA_DIR / "curated" / "formatted_data.jsonl"), help="Path to the curated JSONL file"),
    train_ratio: float = typer.Option(0.95, help="Ratio of data to use for training (e.g., 0.95 = 95% train, 5% val)"),
    seed: int = typer.Option(42, help="Random seed for reproducible splitting")
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"❌ Input file not found: {input_path}")

    print(f"📖 Loading curated dataset from: {input_path}")
    dataset = load_dataset("json", data_files=str(input_path), split="train")
    print(f"📊 Total samples: {len(dataset)}")

    # Split the dataset
    split_dataset = dataset.train_test_split(test_size=1.0 - train_ratio, seed=seed)
    train_ds = split_dataset["train"]
    val_ds = split_dataset["test"]

    print(f"✂️ Split complete: {len(train_ds)} train samples, {len(val_ds)} validation samples.")

    # Define output paths
    train_dir = DATA_DIR / "train"
    val_dir = DATA_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_output = train_dir / "sft_train.jsonl"
    val_output = val_dir / "sft_val.jsonl"

    # Save to disk
    print(f"💾 Saving train split to: {train_output}")
    train_ds.to_json(str(train_output), orient="records", lines=True)
    
    print(f"💾 Saving validation split to: {val_output}")
    val_ds.to_json(str(val_output), orient="records", lines=True)

    print("🎉 Dataset pipeline complete! You can now run your training script.")

if __name__ == "__main__":
    app()