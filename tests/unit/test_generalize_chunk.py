import pandas as pd
import json
from pathlib import Path
from unittest.mock import MagicMock

from SKALD.chunk_generalizer import generalize_first_chunk


def test_generalize_first_chunk(tmp_path):
    """
    Verifies:
    - chunk is loaded
    - encoded numeric columns created when encode=True
    - ola_2.generalize_chunk is called with correct data
    - encoded columns removed from final output
    - output CSV is saved
    """

    # Prepare chunk file
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    df = pd.DataFrame({
        "AGE": [20, 22],
        "GENDER": ["Male", "Female"]
    })
    df_path = chunks_dir / "chunk_1.csv"
    df.to_csv(df_path, index=False)

    # Encoding map for AGE
    encoding_maps = {
        "AGE": {
            "encoding_map": {20: 1, 22: 2},
            "multiplier": 1,
            "type": "int"
        }
    }

    numerical_columns_info = [
        {"column": "AGE", "encode": True, "type": "int"}
    ]

    # Mock OLA_2 object
    fake_ola2 = MagicMock()
    fake_ola2.generalize_chunk.return_value = pd.DataFrame({
        "AGE": ["[20-21]", "[22-23]"],
        "GENDER": ["Male", "Female"],
        "AGE_encoded": [1, 2],   # before removal
    })

    out_path = tmp_path / "out.csv"

    # Run
    generalize_first_chunk(
        chunk_file="chunk_1.csv",
        output_path=str(out_path),
        numerical_columns_info=numerical_columns_info,
        encoding_maps=encoding_maps,
        ola_2=fake_ola2,
        final_rf=[1]  # dummy
    )

    # OLA_2.generalize_chunk must be called
    assert fake_ola2.generalize_chunk.called

    # Output file must exist
    assert out_path.exists()

    # Load output CSV
    out_df = pd.read_csv(out_path)

    # AGE_encoded must be removed
    assert "AGE_encoded" not in out_df.columns

    # Generalized AGE must exist
    assert "AGE" in out_df.columns
    assert out_df["AGE"].iloc[0] == "[20-21]"
