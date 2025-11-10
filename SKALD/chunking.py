import pandas as pd
import psutil
import os

def split_csv_by_ram(data_dir="data"):
    """
    Automatically detects the single CSV file inside `data_dir`
    and splits it into chunks based on 1/4th of total system RAM.
    """
    os.makedirs("chunks", exist_ok=True)

    # --- Find the single CSV file ---
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found inside '{data_dir}'")
    if len(csv_files) > 1:
        raise ValueError(f"More than one CSV found in '{data_dir}': {csv_files}")

    input_csv = os.path.join(data_dir, csv_files[0])
    print(f"Detected CSV: {input_csv}")

    # --- Detect RAM and compute chunk size ---
    total_ram_bytes = psutil.virtual_memory().total
    chunk_ram_bytes = total_ram_bytes // 4  # one-fourth of RAM
    chunk_ram_gb = chunk_ram_bytes / (1024 ** 3)

    print(f"Total RAM: {total_ram_bytes / (1024 ** 3):.2f} GB")
    print(f"Using ~{chunk_ram_gb:.2f} GB per chunk")

    # --- Estimate rows per chunk ---
    sample = pd.read_csv(input_csv, nrows=10000)
    avg_row_size = sample.memory_usage(index=True, deep=True).sum() / len(sample)
    rows_per_chunk = int(chunk_ram_bytes / avg_row_size)
    print(f"≈ {rows_per_chunk:,} rows per chunk (~{chunk_ram_gb:.2f} GB)")

    # --- Stream the CSV into chunk files ---
    reader = pd.read_csv(input_csv, chunksize=rows_per_chunk)
    for i, chunk in enumerate(reader, start=1):
        out_path = os.path.join("chunks", f"chunk_{i}.csv")
        chunk.to_csv(out_path, index=False)
        print(f"wrote {out_path} ({len(chunk):,} rows)")

    print("Done splitting CSV into chunks.")
