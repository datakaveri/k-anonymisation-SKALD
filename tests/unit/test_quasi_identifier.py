from SKALD.quasi_identifier import QuasiIdentifier

def test_quasi_identifier_numeric_range():
    qi = QuasiIdentifier("AGE", is_categorical=False, min_value=10, max_value=20)
    assert qi.get_range() == 11.0

def test_quasi_identifier_categorical_known_levels():
    qi = QuasiIdentifier("Gender", is_categorical=True)
    assert qi.get_range() == 2.0

def test_quasi_identifier_numeric_defaults():
    qi = QuasiIdentifier("HEIGHT", is_categorical=False)
    assert qi.get_range() == 0.0
