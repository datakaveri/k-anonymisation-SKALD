import pytest
from chunkanon.generalization_ri import OLA_1
from chunkanon.quasi_identifier import QuasiIdentifier


@pytest.fixture
def sample_quasi_identifiers():
    return [
        QuasiIdentifier("Age", is_categorical=False, min_value=0, max_value=100),
        QuasiIdentifier("Blood Group", is_categorical=True),
        QuasiIdentifier("Profession", is_categorical=True)
    ]


@pytest.fixture
def ola(sample_quasi_identifiers):
    return OLA_1(
        quasi_identifiers=sample_quasi_identifiers,
        n=1000,
        max_equivalence_classes=100,
        multiplication_factors={
            "Age": 2,
            "Blood Group": 1,
            "Profession": 1
        }
    )


def test_init(ola):
    assert ola.n == 1000
    assert ola.max_equivalence_classes == 100
    assert len(ola.quasi_identifiers) == 3
    assert ola.tree == []
    assert ola.smallest_passing_ri is None


def test_calculate_equivalence_classes_numerical_only():
    qis = [QuasiIdentifier("Age", is_categorical=False, min_value=0, max_value=100)]
    ola = OLA_1(qis, n=100, max_equivalence_classes=10, multiplication_factors={"Age": 2})
    result = ola.calculate_equivalence_classes([10])
    assert result == 10


def test_calculate_equivalence_classes_categorical(sample_quasi_identifiers):
    ola = OLA_1(
        quasi_identifiers=sample_quasi_identifiers,
        n=1000,
        max_equivalence_classes=100,
        multiplication_factors={"Age": 2, "Blood Group": 1, "Profession": 1}
    )
    # bin_widths: Age = 20, BG = level 2 (4 classes), Profession = level 3 (2 classes)
    result = ola.calculate_equivalence_classes([20, 2, 3])
    expected = (100 / 20) * 4 * 2  # = 5 * 4 * 2 = 40
    assert result == expected


def test_build_tree(ola):
    tree = ola.build_tree()
    assert isinstance(tree, list)
    assert all(isinstance(level, list) for level in tree)
    assert tree[0][0] == [1, 1, 1]


def test_get_precision_numerical_only():
    qis = [QuasiIdentifier("Age", is_categorical=False, min_value=0, max_value=64)]
    ola = OLA_1(qis, n=100, max_equivalence_classes=10, multiplication_factors={"Age": 2})
    precision = ola.get_precision([4])
    # log base 2 of 64 = 6 → max_levels = 8, log base 2 of 4 = 2 → level = 4, precision = 4/8 = 0.5
    assert round(precision, 2) == 0.5


def test_find_smallest_passing_ri(sample_quasi_identifiers):
    ola = OLA_1(
        quasi_identifiers=sample_quasi_identifiers,
        n=1000,
        max_equivalence_classes=100000,  # make it large so something will pass
        multiplication_factors={"Age": 2, "Blood Group": 1, "Profession": 1}
    )
    ola.build_tree()
    ri = ola.find_smallest_passing_ri(1000)
    assert ri is not None
    assert isinstance(ri, list)
    assert all(isinstance(val, (int, float)) for val in ri)


def test_get_optimal_ri(sample_quasi_identifiers):
    ola = OLA_1(
        quasi_identifiers=sample_quasi_identifiers,
        n=1000,
        max_equivalence_classes=100000,
        multiplication_factors={"Age": 2, "Blood Group": 1, "Profession": 1}
    )
    ola.build_tree()
    ola.find_smallest_passing_ri(1000)
    ri = ola.get_optimal_ri()
    assert isinstance(ri, list)
    assert len(ri) == len(sample_quasi_identifiers)


def test_get_optimal_ri_array(sample_quasi_identifiers):
    ola = OLA_1(
        quasi_identifiers=sample_quasi_identifiers,
        n=1000,
        max_equivalence_classes=100000,
        multiplication_factors={"Age": 2, "Blood Group": 1, "Profession": 1}
    )
    ola.build_tree()
    ola.find_smallest_passing_ri(1000)
    candidates = ola.get_optimal_ri_array()
    assert isinstance(candidates, list)
    assert all(isinstance(ri, list) for ri in candidates)
