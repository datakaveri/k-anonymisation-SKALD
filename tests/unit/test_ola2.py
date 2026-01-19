# tests/unit/test_ola2.py
import os, json
import numpy as np
import pandas as pd
import pytest
from SKALD.generalization_rf import OLA_2
from SKALD.quasi_identifier import QuasiIdentifier


def _basic_qis():
    num = QuasiIdentifier("AGE", is_categorical=False, min_value=20, max_value=30)
    cat = QuasiIdentifier("GENDER", is_categorical=True)
    return [num, cat]


# -------------------------------------------------------------
# EXISTING TESTS (unchanged)
# -------------------------------------------------------------
def test_build_domains():
    qis = _basic_qis()
    df = pd.DataFrame({"AGE": [20, 21, 22], "GENDER": ["Male", "Female", "Male"]})
    ola = OLA_2(qis, total_records=3, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    ola.build_domains(df)
    assert ola.domains[0] == [20, 21, 22]
    assert sorted(ola.domains[1]) == ["Female", "Male"]


def test_process_chunk_numeric_and_categorical():
    qis = _basic_qis()
    df = pd.DataFrame({"AGE": [20, 21, 22], "GENDER": ["Male", "Male", "Female"], "STATE": ["KA", "KA", "MH"]})
    ola = OLA_2(qis, total_records=3, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    h = ola.process_chunk(df, bin_widths=[1, 1])
    assert h.shape == (3, 2)
    assert h.sum() == 3


def test_merge_axis_simple():
    arr = np.array([[1, 2, 3, 4]])
    ola = OLA_2([], 0, 0, {}, "STATE")
    merged = ola.merge_axis(arr, axis=1, group_size=2)
    assert merged.tolist() == [[3, 7]]


def test_merge_sets_axis_simple():
    arr = np.empty((4,), dtype=object)
    arr[0], arr[1], arr[2], arr[3] = {1}, {2}, {3}, {4}
    ola = OLA_2([], 0, 0, {}, "STATE")
    merged = ola.merge_sets_axis(arr, axis=0, group_size=2)
    assert merged.tolist() == [{1, 2}, {3, 4}]


def test_merge_histograms():
    h1 = np.array([[1, 2], [0, 1]])
    h2 = np.array([[0, 1], [3, 0]])
    ola = OLA_2([], 0, 0, {}, "STATE")
    merged = ola.merge_histograms([h1, h2])
    assert (merged == np.array([[1, 3], [3, 1]])).all()


def test_check_k_anonymity_basic():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=4, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    histogram = np.array([[2, 0],[0, 2]])
    ola.sensitive_sets = np.empty_like(histogram, dtype=object)
    for idx in np.ndindex(*histogram.shape):
        ola.sensitive_sets[idx] = set()
    ola.sensitive_sets[0, 0] = {"KA"}
    ola.sensitive_sets[1, 1] = {"MH"}

    assert ola.check_k_anonymity(histogram, k=2, l=1) is True
    assert ola.check_k_anonymity(histogram, k=3, l=1) is False
    assert ola.check_k_anonymity(histogram, k=1, l=2) is False


def test_merge_equivalence_classes():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=3, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    histogram = np.array([[1, 1],[0, 1]])
    sensitive = np.empty_like(histogram, dtype=object)
    for idx, val in zip(np.ndindex(*histogram.shape), [{"A"}, {"B"}, {"C"}, {"D"}]):
        sensitive[idx] = val

    merged_h, merged_s = ola.merge_equivalence_classes(histogram, sensitive, new_bin_widths=[2, 1])
    assert merged_h.shape[0] == 1


def test_equivalence_stats():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=4, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    histogram = np.array([[2, 1],[1, 0]])
    ola.sensitive_sets = np.empty_like(histogram, dtype=object)
    for idx in np.ndindex(*histogram.shape):
        ola.sensitive_sets[idx] = {"X"}
    stats = ola.get_equivalence_class_stats(histogram, bin_widths=[1, 1], k=2)
    assert stats.get(2) == 1


def test_generalize_chunk_numeric(monkeypatch, tmp_path):
    enc_dir = tmp_path / "encodings"
    monkeypatch.setenv("SKALD_ENCODING_DIR", str(enc_dir))
    os.makedirs(enc_dir, exist_ok=True)
    with open(enc_dir / "age_encoding.json", "w") as f:
        json.dump({"encoding_map": {"20": 1, "25": 2}, "multiplier": 1}, f)

    df = pd.DataFrame({"AGE": [20, 25], "GENDER": ["Male", "Female"]})
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=2, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    out = ola.generalize_chunk(df, bin_widths=[5, 1])
    assert out["AGE"].dtype.name == "category"


def test_generalize_chunk_categorical():
    df = pd.DataFrame({"AGE": [20], "GENDER": ["Male"]})
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=1, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    out = ola.generalize_chunk(df, bin_widths=[1, 2])
    assert out["GENDER"].iloc[0] == "Other"


def test_combine_generalized_chunks(tmp_path):
    ola = OLA_2([], total_records=0, suppression_limit=0, multiplication_factors={}, sensitive_parameter="X")
    c1 = pd.DataFrame({"A": [1]})
    c2 = pd.DataFrame({"A": [2]})
    out_path = tmp_path / "combined.csv"
    combined = ola.combine_generalized_chunks_to_csv([c1, c2], output_path=str(out_path))
    assert combined.shape == (2, 1)
    assert out_path.exists()


# -------------------------------------------------------------
# NEW HIGH-COVERAGE TESTS BELOW
# -------------------------------------------------------------

def test_build_tree_multiple_levels_encoded_column():
    qis = [
        QuasiIdentifier("AGE_encoded", is_categorical=False, is_encoded=True, min_value=1, max_value=8),
        QuasiIdentifier("GENDER", is_categorical=True),
    ]
    ola = OLA_2(qis, total_records=10, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="X")
    tree = ola.build_tree([1, 1])
    assert len(tree) >= 2  # must expand
    assert tuple([1, 1]) in ola.node_status


def test_get_index_tuple_error_path():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=3, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    df = pd.DataFrame({"AGE": [20], "GENDER": ["Unknown"]})
    ola.build_domains(pd.DataFrame({"AGE": [20], "GENDER": ["Male"]}))
    with pytest.raises(ValueError):
        ola.get_index_tuple(df.iloc[0])


def test_process_chunk_skips_bad_rows():
    qis = _basic_qis()

    # build domains using clean data
    clean = pd.DataFrame({
        "AGE": [20, 25],
        "GENDER": ["Male", "Female"]
    })

    df = pd.DataFrame({
        "AGE": [20, "bad", 25],
        "GENDER": ["Male", "Female", "Female"],
        "STATE": ["KA", "KA", "KA"],
    })

    ola = OLA_2(qis, total_records=3, suppression_limit=10,
                multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")

    ola.build_domains(clean)   # domains already valid

    h = ola.process_chunk(df, bin_widths=[1, 1])
    assert h.sum() == 2          # only valid rows count



def test_check_k_anonymity_sensitive_none():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=3, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    with pytest.raises(ValueError):
        ola.check_k_anonymity(np.array([[1]]), 1, 1)

def test_check_k_anonymity_shape_mismatch():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=3, suppression_limit=10,
                multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")

    histogram = np.array([[2, 0]])
    ola.sensitive_sets = np.empty((1, 1), dtype=object)
    ola.sensitive_sets[0, 0] = {"A"}

    # allow l=0 so rebuilt empty sets won't fail
    assert ola.check_k_anonymity(histogram, k=1, l=0) is True



def test_find_best_rf_selects_best_node():
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=10, suppression_limit=100,
                multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")

    ola.tree = [[[1, 1]], [[2, 1]], [[4, 1]]]
    ola.node_status = {(1, 1): "pass", (2, 1): "pass", (4, 1): "pass"}

    histogram = np.array([[3, 0], [0, 3]])
    ola.sensitive_sets = np.empty_like(histogram, dtype=object)
    for idx in np.ndindex(*histogram.shape):
        ola.sensitive_sets[idx] = {"X"}

    ola.find_best_rf(histogram,
                     pass_nodes=[[1, 1], [2, 1], [4, 1]],
                     k=2, l=1,
                     sensitive_sets=ola.sensitive_sets)

    # algorithm uses 'last best' tie-breaking, so [4,1] is correct
    assert ola.smallest_passing_rf == [4, 1]


def test_generalize_chunk_numeric_no_encoding(tmp_path):
    df = pd.DataFrame({"AGE": [21, 22], "GENDER": ["Male", "Female"]})
    qis = _basic_qis()
    ola = OLA_2(qis, total_records=2, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    out = ola.generalize_chunk(df, bin_widths=[2, 1])
    assert out["AGE"].dtype.name == "category"


def test_generalize_chunk_profession_fallback():
    df = pd.DataFrame({"AGE": [22], "GENDER": ["Male"], "Profession": ["Designer"]})
    qis = [
        QuasiIdentifier("AGE", is_categorical=False, min_value=20, max_value=30),
        QuasiIdentifier("Profession", is_categorical=True)
    ]
    ola = OLA_2(qis, total_records=1, suppression_limit=10, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    out = ola.generalize_chunk(df, bin_widths=[1, 4])
    assert out["Profession"].iloc[0] == "*"


def test_get_suppressed_percent():
    qis = _basic_qis()
    histogram = np.array([[1, 0], [2, 3]])
    ola = OLA_2(qis, total_records=6, suppression_limit=100, multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")
    ola.sensitive_sets = np.empty_like(histogram, dtype=object)
    for idx in np.ndindex(*histogram.shape):
        ola.sensitive_sets[idx] = {"X"}
    percent = ola.get_suppressed_percent([1, 1], histogram, k=2)
    assert percent > 0

def test_get_final_binwidths_full(monkeypatch):
    """
    Full behavioral test of get_final_binwidths:
    - build a 3-level tree: [1,1], [2,1], [4,1]
    - let [1,1] and [2,1] PASS
    - force [4,1] to FAIL
    - ensure correct RF is chosen and correct pass/fail marking happens
    """

    from SKALD.generalization_rf import OLA_2
    from SKALD.quasi_identifier import QuasiIdentifier

    # Simple QIs
    qi_num = QuasiIdentifier("AGE", is_categorical=False, min_value=0, max_value=10)
    qi_cat = QuasiIdentifier("GENDER", is_categorical=True)
    qis = [qi_num, qi_cat]

    ola = OLA_2(
        quasi_identifiers=qis,
        total_records=1,              # critical to force fail branch
        suppression_limit=0,          # no suppression allowed → fail branch triggered
        multiplication_factors={"AGE": 2},
        sensitive_parameter="STATE"
    )

    # Build a fixed tree manually
    ola.tree = [
        [[1, 1]],     # level 0
        [[2, 1]],     # level 1
        [[4, 1]]      # level 2
    ]

    # All nodes unmarked at start
    ola.node_status = {
        (1, 1): None,
        (2, 1): None,
        (4, 1): None
    }

    # Tiny histogram
    histogram = np.array([[3, 3]])
    ola.sensitive_sets = np.empty_like(histogram, dtype=object)
    ola.sensitive_sets[0, 0] = {"X"}

    # Track calls for pass and fail
    passed_nodes = []
    failed_nodes = []

    # Fake progress bar so `.close()` works
    class DummyPbar:
        def update(self, n): pass
        def close(self): pass

    monkeypatch.setattr("SKALD.generalization_rf.tqdm", lambda *a, **k: DummyPbar())

    # Track the current node being evaluated
    state = {"current_node": None}

    def fake_merge(hist, sens, node):
        state["current_node"] = node
        return hist, sens

    def fake_check(hist, k, l):
        node = state["current_node"]
        # FAIL only [4,1]
        return node != [4, 1]

    # Patch merge + check
    monkeypatch.setattr(ola, "merge_equivalence_classes", fake_merge)
    monkeypatch.setattr(ola, "check_k_anonymity", fake_check)

    # Patch subtree marking
    def fake_mark_pass(node, pbar=None):
        passed_nodes.append(tuple(node))
        ola.node_status[tuple(node)] = "pass"

    def fake_mark_fail(node, pbar=None):
        failed_nodes.append(tuple(node))
        ola.node_status[tuple(node)] = "fail"

    monkeypatch.setattr(ola, "_mark_subtree_pass", fake_mark_pass)
    monkeypatch.setattr(ola, "_mark_parents_fail", fake_mark_fail)

    # Patch find_best_rf (we don’t test DM* here)
    def fake_find_rf(hist, pass_nodes, k, l, sens):
        ola.smallest_passing_rf = [1, 1]

    monkeypatch.setattr(ola, "find_best_rf", fake_find_rf)

    # === RUN ===
    rf = ola.get_final_binwidths(histogram, k=2, l=1)

    # === ASSERTIONS ===
    assert rf == [1, 1]  # from fake_find_rf

    # [1,1] and [2,1] should be PASS
    assert (1, 1) in passed_nodes
    assert (2, 1) in passed_nodes

    # [4,1] MUST be FAIL
    assert (4, 1) in failed_nodes

def test_mark_subtree_pass():
    from SKALD.generalization_rf import OLA_2
    from SKALD.quasi_identifier import QuasiIdentifier

    qi = [
        QuasiIdentifier("AGE", is_categorical=False, min_value=0, max_value=10),
        QuasiIdentifier("GENDER", is_categorical=True)
    ]

    ola = OLA_2(qi, total_records=10, suppression_limit=50,
                multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")

    # Build small tree manually
    ola.tree = [
        [[1, 1]],
        [[2, 1], [1, 2]],
        [[3, 1], [2, 2]]
    ]

    # All unmarked initially
    ola.node_status = {
        (1, 1): None,
        (2, 1): None,
        (1, 2): None,
        (3, 1): None,
        (2, 2): None,
    }

    # Call function
    ola._mark_subtree_pass([2, 1], pbar=None)

    # Assertions
    # [2,1] is pass
    assert ola.node_status[(2, 1)] == "pass"

    # children ([3,1] and [2,2]) must also pass
    assert ola.node_status[(3, 1)] == "pass"
    assert ola.node_status[(2, 2)] == "pass"

    # unrelated sibling should NOT be marked
    assert ola.node_status[(1, 2)] is None

    # root should stay unmarked unless ≥ in all dimensions
    assert ola.node_status[(1, 1)] is None

def test_mark_parents_fail():
    from SKALD.generalization_rf import OLA_2
    from SKALD.quasi_identifier import QuasiIdentifier

    qi = [
        QuasiIdentifier("AGE", is_categorical=False, min_value=0, max_value=10),
        QuasiIdentifier("GENDER", is_categorical=True)
    ]

    ola = OLA_2(qi, total_records=10, suppression_limit=50,
                multiplication_factors={"AGE": 2}, sensitive_parameter="STATE")

    ola.tree = [
        [[1, 1]],
        [[2, 1], [1, 2]],
        [[3, 1], [2, 2]]
    ]

    # All unmarked
    ola.node_status = {
        (1, 1): None,
        (2, 1): None,
        (1, 2): None,
        (3, 1): None,
        (2, 2): None,
    }

    ola._mark_parents_fail([3, 1], pbar=None)

    # Node itself must fail
    assert ola.node_status[(3, 1)] == "fail"

    # Parents (lower levels, all dims <= 3,1)
    assert ola.node_status[(2, 1)] == "fail"
    assert ola.node_status[(1, 1)] == "fail"

    # Node [2,2] is NOT a parent (2 ≤ 3 but 2 ≤ 1 fails)
    assert ola.node_status[(2, 2)] is None

    # Node [1,2] is NOT a parent
    assert ola.node_status[(1, 2)] is None
