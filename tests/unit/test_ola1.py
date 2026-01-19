import pytest
from SKALD.generalization_ri import OLA_1
from SKALD.quasi_identifier import QuasiIdentifier


def _qis_for_test():
    # Simple: 1 numerical, 1 categorical
    # Numerical col range = 10 (0–9)
    num_qi = QuasiIdentifier("AGE", is_categorical=False, min_value=0, max_value=9)
    cat_qi = QuasiIdentifier("Gender", is_categorical=True)

    multiplication_factors = {"AGE": 2}
    return [num_qi, cat_qi], multiplication_factors


def test_build_tree_structure():
    qis, factors = _qis_for_test()
    ola = OLA_1(qis, n=100, max_equivalence_classes=9999, multiplication_factors=factors)

    tree = ola.build_tree()

    # Tree must start with [1,1]
    assert tree[0] == [[1, 1]]

    # There must be >1 levels
    assert len(tree) > 1

    # Each node must be unique
    all_nodes = {tuple(node) for level in tree for node in level}
    assert len(all_nodes) == sum(len(level) for level in tree)


def test_calculate_equivalence_classes():
    qis, factors = _qis_for_test()
    ola = OLA_1(qis, n=100, max_equivalence_classes=9999, multiplication_factors=factors)

    # AGE range=10, bin_width=1 → 10 classes
    # Gender: level1 → 2 classes
    classes = ola.calculate_equivalence_classes([1, 1])
    assert classes == pytest.approx(20)

    # Higher generalization reduces classes
    classes2 = ola.calculate_equivalence_classes([2, 1])
    assert classes2 < classes


def test_find_smallest_passing_ri():
    qis, factors = _qis_for_test()
    ola = OLA_1(qis, n=1000, max_equivalence_classes=25, multiplication_factors=factors)
    ola.build_tree()

    passing = ola.find_smallest_passing_ri(n=1000)

    # Must return a node list
    assert isinstance(passing, list)
    assert len(passing) == 2

    # check equivalence classes indeed <= threshold
    assert ola.calculate_equivalence_classes(passing) <= 25


def test_precision_scoring():
    qis, factors = _qis_for_test()
    ola = OLA_1(qis, n=100, max_equivalence_classes=9999, multiplication_factors=factors)

    p_low = ola.get_precision([1, 1])
    p_high = ola.get_precision([4, 2])

    assert p_low < p_high  # higher generalization => worse precision


def test_get_optimal_ri_and_array():
    qis, factors = _qis_for_test()
    ola = OLA_1(qis, n=1000, max_equivalence_classes=50, multiplication_factors=factors)
    ola.build_tree()
    ola.find_smallest_passing_ri(1000)

    arr = ola.get_optimal_ri_array()
    assert isinstance(arr, list)
    assert len(arr) > 0

    best = ola.get_optimal_ri()
    assert isinstance(best, list)
    assert len(best) == len(qis)
    assert best in arr


def test_mark_parents_fail_is_called():
    qis, factors = _qis_for_test()
    # Set max_equivalence_classes VERY low so almost everything fails
    ola = OLA_1(qis, n=100, max_equivalence_classes=1, multiplication_factors=factors)

    # Build a small tree:
    # Level 0 : [1,1]
    # Level 1 : something like [2,1], [1,2]
    ola.build_tree()

    # Force the child node to FAIL
    failing_node = ola.tree[1][0]          # Pick the first node from level 1
    classes = ola.calculate_equivalence_classes(failing_node)
    assert classes > 1                     # Confirm: this WILL fail the threshold

    # Call mark_parents_fail explicitly
    ola._mark_parents_fail(failing_node)

    # Node must be marked as fail
    assert ola.node_status[tuple(failing_node)] == "fail"

    # Parent node [1,1] must also be marked fail
    parent = [1, 1]
    assert ola.node_status[tuple(parent)] == "fail"
