import pandas as pd
import numpy as np
from pathlib import Path

from SKALD.chunk_processing import process_chunks_for_histograms
from SKALD.quasi_identifier import QuasiIdentifier
from SKALD.generalization_rf import OLA_2


def _setup_ola2_for_test():
    """
    Create a simple OLA_2 object with known QIs and prebuilt domains.
    Used to avoid domain/broadcasting issues.
    """
    qis = [
        QuasiIdentifier("AGE", is_categorical=False, min_value=20, max_value=30),
        QuasiIdentifier("GENDER", is_categorical=True),
    ]

    ola2 = OLA_2(
        quasi_identifiers=qis,
        total_records=10,
        suppression_limit=100,
        multiplication_factors={"AGE": 2},
        sensitive_parameter="STATENAME",
    )

    # Build minimal tree
    ola2.build_tree([1, 1])

    # Manually set domains to avoid mismatch
    ola2.domains = [
        list(range(20, 31)),         # AGE domain (20–30)
        ["Male", "Female"],          # GENDER domain
    ]

    return ola2


def test_process_chunks_basic(tmp_path):
    """
    Basic: ensure two chunks produce two histograms
    and each histogram has non-zero total count matching rows.
    """
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    # Make two mini chunks
    df1 = pd.DataFrame({
        "AGE": [20, 21],
        "GENDER": ["Male", "Female"],
        "STATENAME": ["KA", "KA"]
    })
    df2 = pd.DataFrame({
        "AGE": [22, 23],
        "GENDER": ["Male", "Female"],
        "STATENAME": ["MH", "MH"]
    })

    df1.to_csv(chunks_dir / "a.csv", index=False)
    df2.to_csv(chunks_dir / "b.csv", index=False)

    ola2 = _setup_ola2_for_test()

    numerical_cols = [{"column": "AGE", "encode": False, "type": "int"}]

    histograms = process_chunks_for_histograms(
        ["a.csv", "b.csv"],
        str(chunks_dir),
        numerical_cols,
        encoding_maps={},
        ola_2=ola2,
        initial_ri=[1, 1],
    )

    assert len(histograms) == 2
    assert isinstance(histograms[0], np.ndarray)
    assert histograms[0].sum() == 2
    assert histograms[1].sum() == 2


def _ola2_with_encoded_age():
    # AGE is encoded → note is_encoded=True and the column_name stays "AGE" (the code auto-uses AGE_encoded)
    q_num = QuasiIdentifier("AGE_encoded", is_categorical=False, is_encoded=True, min_value=1, max_value=3)
    q_cat = QuasiIdentifier("GENDER", is_categorical=True)
    ola2 = OLA_2([q_num, q_cat], total_records=3, suppression_limit=100,
                 multiplication_factors={"AGE": 2}, sensitive_parameter="STATENAME")
    # initial RI
    ola2.build_tree([1, 1])
    # domains: encoded age values 1..3 and genders
    ola2.domains = [
        [1, 2, 3],
        ["Female", "Male"],
    ]
    return ola2

def test_process_chunks_for_histograms(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    pd.DataFrame({"AGE": [20, 21], "GENDER": ["Male", "Female"], "STATENAME": ["KA", "KA"]}).to_csv(chunks / "a.csv", index=False)
    pd.DataFrame({"AGE": [21, 23], "GENDER": ["Male", "Female"], "STATENAME": ["MH", "MH"]}).to_csv(chunks / "b.csv", index=False)

    # not encoded case → QI is NOT encoded
    qis = [
        QuasiIdentifier("AGE", is_categorical=False, min_value=19, max_value=85),
        QuasiIdentifier("GENDER", is_categorical=True),
    ]
    ola2 = OLA_2(qis, total_records=4, suppression_limit=0.5,
                 multiplication_factors={"AGE": 2}, sensitive_parameter="STATENAME")
    initial_ri = [1, 1]
    ola2.build_tree(initial_ri)

    numerical_cols = [{"column": "AGE", "encode": False, "type": "int"}]
    encoding_maps = {}

    hists = process_chunks_for_histograms(
        ["a.csv", "b.csv"], str(chunks), numerical_cols, encoding_maps, ola2, initial_ri
    )
    assert len(hists) == 2
    assert sum(h.sum() for h in hists) == 4  # all rows counted

def test_process_chunks_with_encoding(tmp_path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    df = pd.DataFrame({
        "AGE": [20, 22, 24],
        "GENDER": ["Male", "Female", "Male"],
        "STATENAME": ["KA", "KA", "KA"],
    })
    df.to_csv(chunks_dir / "c.csv", index=False)

    # encoding map for AGE
    encoding_maps = {
        "AGE": {"encoding_map": {20: 1, 22: 2, 24: 3}, "multiplier": 1, "type": "int"}
    }
    numerical_cols = [{"column": "AGE", "encode": True, "type": "int"}]

    ola2 = _ola2_with_encoded_age()

    hists = process_chunks_for_histograms(
        ["c.csv"],
        str(chunks_dir),
        numerical_cols,
        encoding_maps=encoding_maps,
        ola_2=ola2,
        initial_ri=[1, 1],
    )

    assert len(hists) == 1
    assert hists[0].sum() == 3



def test_chunk_processing_histogram_shape(tmp_path):
    """
    Ensure histogram shape matches OLA_2.domains.
    """

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()

    df = pd.DataFrame({
        "AGE": [20, 21],
        "GENDER": ["Male", "Female"],
        "STATENAME": ["KA", "KA"],
    })
    df.to_csv(chunks_dir / "z.csv", index=False)

    ola2 = _setup_ola2_for_test()

    numerical_cols = [{"column": "AGE", "encode": False, "type": "int"}]

    histograms = process_chunks_for_histograms(
        ["z.csv"],
        str(chunks_dir),
        numerical_cols,
        encoding_maps={},
        ola_2=ola2,
        initial_ri=[1, 1],
    )

    h = histograms[0]

    # Shape should be len(AGE domain) × len(GENDER domain)
    assert h.shape == (len(ola2.domains[0]), len(ola2.domains[1]))
