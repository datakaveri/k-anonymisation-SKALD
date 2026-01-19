import pandas as pd

def test_split_csv_by_ram(tmp_path, monkeypatch):
    # create a single CSV in data/
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df = pd.DataFrame({"x": list(range(10000))})
    csv = data_dir / "big.csv"
    df.to_csv(csv, index=False)

    # monkeypatch psutil to force small chunks
    import psutil
    class VM:
        total = 64 * 1024 * 1024  # 64MB
    monkeypatch.setattr(psutil, "virtual_memory", lambda: VM)

    from SKALD.chunking import split_csv_by_ram
    # switch to temp CWD so it writes "chunks" here
    monkeypatch.chdir(tmp_path)
    split_csv_by_ram(str(data_dir))

    chunks_dir = tmp_path / "chunks"
    files = sorted(p.name for p in chunks_dir.glob("chunk_*.csv"))
    assert files, "no chunk files created"
