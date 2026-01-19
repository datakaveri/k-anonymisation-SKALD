import pytest
from SKALD.categorical import CategoricalGeneralizer


def test_blood_group_levels():
    gen = CategoricalGeneralizer()

    # Level 1: no generalization
    assert gen.generalize_blood_group("A+", 1) == "A+"
    assert gen.generalize_blood_group("O-", 1) == "O-"

    # Level 2: grouped by A/B/AB/O
    assert gen.generalize_blood_group("A+", 2) == "A"
    assert gen.generalize_blood_group("B-", 2) == "B"
    assert gen.generalize_blood_group("AB+", 2) == "AB"
    assert gen.generalize_blood_group("O+", 2) == "O"

    # Level 3: fully generalized
    assert gen.generalize_blood_group("A+", 3) == "*"
    assert gen.generalize_blood_group("O-", 3) == "*"


def test_blood_group_unknown_value():
    gen = CategoricalGeneralizer()

    # Unknown value → returns "Other"
    assert gen.generalize_blood_group("XYZ", 2) == "Other"


def test_gender_levels():
    gen = CategoricalGeneralizer()

    # Level 1: no generalization
    assert gen.generalize_gender("Male", 1) == "Male"
    assert gen.generalize_gender("Female", 1) == "Female"

    # Level 2: fully generalized
    assert gen.generalize_gender("Male", 2) == "*"
    assert gen.generalize_gender("Female", 2) == "*"


def test_gender_unknown_value():
    gen = CategoricalGeneralizer()
    # No mapping → return "Other"
    assert gen.generalize_gender("Unknown", 2) == "Other"


def test_profession_levels():
    gen = CategoricalGeneralizer()

    # Level 1: no generalization (lambda level)
    assert gen.generalize_profession("Software Engineering", 1) == "Software Engineering"

    # Level 2: mapped to subcategories
    assert gen.generalize_profession("Software Engineering", 2) == "Engineering"
    assert gen.generalize_profession("Medical Specialists", 2) == "Healthcare"

    # Level 3: broader categories
    assert gen.generalize_profession("Software Engineering", 3) == "Non-Service"
    assert gen.generalize_profession("Medical Specialists", 3) == "Service Sector"

    # Level 4: lambda returning "*"
    assert gen.generalize_profession("Software Engineering", 4) == "*"


def test_profession_unknown_value():
    gen = CategoricalGeneralizer()

    # Unknown value → at dictionary levels returns "Other"
    assert gen.generalize_profession("RandomJob", 2) == "Other"

    # Level 4 → always "*"
    assert gen.generalize_profession("RandomJob", 4) == "*"


def test_unknown_column_falls_back_to_original():
    gen = CategoricalGeneralizer()
    assert gen.generalize("NotAColumn", "ABC", 1) == "ABC"
