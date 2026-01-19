import pytest

from SKALD.build_QI import build_quasi_identifiers
from SKALD.quasi_identifier import QuasiIdentifier


def test_build_qi_numeric_non_encoded():
    numerical_columns_info = [
        {"column": "AGE", "encode": False, "type": "int"}
    ]
    categorical_columns = []
    encoding_maps = {}
    hardcoded_min_max = {"AGE": (10, 50)}

    qis, cols = build_quasi_identifiers(
        numerical_columns_info,
        categorical_columns,
        encoding_maps,
        hardcoded_min_max,
    )

    assert len(qis) == 1
    assert cols == ["AGE"]

    qi = qis[0]
    assert isinstance(qi, QuasiIdentifier)
    assert qi.column_name == "AGE"
    assert qi.is_categorical is False
    assert qi.is_encoded is False
    assert qi.min_value == 10
    assert qi.max_value == 50


def test_build_qi_numeric_encoded():
    numerical_columns_info = [
        {"column": "PINCODE", "encode": True, "type": "int"}
    ]
    categorical_columns = []
    # encoding map with 5 unique encoded values
    encoding_maps = {
        "PINCODE": {"encoding_map": {100: 1, 200: 2, 300: 3, 400: 4, 500: 5}}
    }
    hardcoded_min_max = {}

    qis, cols = build_quasi_identifiers(
        numerical_columns_info,
        categorical_columns,
        encoding_maps,
        hardcoded_min_max,
    )

    assert len(qis) == 1
    assert cols == ["PINCODE_encoded"]

    qi = qis[0]
    assert qi.column_name == "PINCODE_encoded"
    assert qi.is_encoded is True
    assert qi.min_value == 1
    assert qi.max_value == 5


def test_build_qi_categorical_only():
    numerical_columns_info = []
    categorical_columns = ["GENDER", "Blood Group"]

    encoding_maps = {}
    hardcoded_min_max = {}

    qis, cols = build_quasi_identifiers(
        numerical_columns_info,
        categorical_columns,
        encoding_maps,
        hardcoded_min_max,
    )

    assert len(qis) == 2
    assert cols == ["GENDER", "Blood Group"]

    assert qis[0].is_categorical is True
    assert qis[1].is_categorical is True


def test_build_qi_numeric_and_categorical():
    numerical_columns_info = [
        {"column": "AGE", "encode": False, "type": "int"}
    ]
    categorical_columns = ["GENDER"]

    encoding_maps = {}
    hardcoded_min_max = {"AGE": (0, 100)}

    qis, cols = build_quasi_identifiers(
        numerical_columns_info,
        categorical_columns,
        encoding_maps,
        hardcoded_min_max,
    )

    assert cols == ["AGE", "GENDER"]
    assert len(qis) == 2

    # AGE QI
    qi_age = qis[0]
    assert qi_age.column_name == "AGE"
    assert qi_age.min_value == 0
    assert qi_age.max_value == 100

    # GENDER QI
    qi_gender = qis[1]
    assert qi_gender.is_categorical is True


def test_build_qi_raises_when_no_qi():
    with pytest.raises(ValueError):
        build_quasi_identifiers([], [], {}, {})
