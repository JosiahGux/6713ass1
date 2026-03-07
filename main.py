import sys
import importlib
from pathlib import Path

import pandas as pd


CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


# ---------------------------------------
# Unit Test Loading
# ---------------------------------------


def load_unit_tests(path="unit_tests.csv"):
    """
    Loads public unit tests.

    Required columns:
    - sentence
    - source_level
    - target_level

    Optional:
    - expected_output
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(file_path)

    required_columns = {"sentence", "source_level", "target_level"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "unit_tests.csv must contain columns: "
            "sentence, source_level, target_level"
        )

    return df


# ---------------------------------------
# Validation
# ---------------------------------------


def validate_levels(source_level, target_level):
    if source_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid source CEFR level: {source_level}")
    if target_level not in CEFR_LEVELS:
        raise ValueError(f"Invalid target CEFR level: {target_level}")


# ---------------------------------------
# Student Import
# ---------------------------------------


def load_student_module(zid_name):
    """
    Loads student module (e.g. z1234567.py).

    The module must define:
    transform_sentence(sentence, source_level, target_level)
    """
    try:
        module = importlib.import_module(zid_name)
    except Exception as e:
        raise ImportError(
            f"Could not import {zid_name}.py. "
            f"Ensure it is in the same directory as main.py. "
            f"Original error: {e}"
        )

    if not hasattr(module, "transform_sentence"):
        raise AttributeError(
            f"{zid_name}.py must define "
            f"transform_sentence(sentence, source_level, target_level)"
        )

    return module


# ---------------------------------------
# Runner
# ---------------------------------------


def run_tests(student_module, tests_df):
    has_expected = "expected_output" in tests_df.columns

    for i, row in enumerate(tests_df.itertuples(index=False), 1):
        sentence = row.sentence
        source = row.source_level
        target = row.target_level

        print(f"\nTest {i}")
        print("Input sentence:", sentence)
        print("Source level  :", source)
        print("Target level  :", target)

        try:
            validate_levels(source, target)
            output = student_module.transform_sentence(sentence, source, target)
        except Exception as e:
            print("ERROR:", e)
            continue

        print("Output        :", output)

        if has_expected:
            print("Expected      :", row.expected_output)


# ---------------------------------------
# Main
# ---------------------------------------


def main():
    """
    Usage:
        python main.py <zid>

    Example:
        python main.py z1234567
    """
    if len(sys.argv) != 2:
        print("Usage: python main.py <zid>")
        print("Example: python main.py z1234567")
        sys.exit(1)

    zid_name = sys.argv[1]

    tests = load_unit_tests("unit_tests.csv")
    student_module = load_student_module(zid_name)

    run_tests(student_module, tests)


if __name__ == "__main__":
    main()
