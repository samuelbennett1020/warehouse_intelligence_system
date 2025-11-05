import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from preprocessing.ingest import (
    standardize_column_names,
    group_similar_text,
    confirm_overwrite,
    get_csv_file_path,
    convert_to_categorical,
    CONFIG
)

# Disable logging for unit tests
CONFIG["disable_logging"] = True


def test_standardize_column_names():
    """Test that columns are lowercased and spaces replaced with underscores."""
    df = pd.DataFrame({"Column A": [1], "COLUMN B": [2], "columN C ": [3]})
    result = standardize_column_names(df)
    assert list(result.columns) == ["column_a", "column_b", "column_c"]


def test_group_similar_text():
    """Test fuzzy grouping of similar text entries."""
    series = pd.Series(["Apple", "apple ", "APPLE", "Banana", "banana"])
    result = group_similar_text(series, threshold=80)
    assert set(result.unique()) == {"apple", "banana"}


@patch("builtins.input", return_value="y")
def test_confirm_overwrite_yes(mock_input):
    """Test confirm_overwrite returns True for 'y' input."""
    file_path = MagicMock(spec=Path)
    file_path.exists.return_value = True
    assert confirm_overwrite(file_path) is True


@patch("builtins.input", return_value="n")
def test_confirm_overwrite_no(mock_input):
    """Test confirm_overwrite returns False for 'n' input."""
    file_path = MagicMock(spec=Path)
    file_path.exists.return_value = True
    assert confirm_overwrite(file_path) is False


def test_get_csv_file_path():
    """Test get_csv_file_path returns Path and string correctly."""
    path_obj = get_csv_file_path(10, scan_month=6, scan_year=2024, csv_name="results")
    path_str = get_csv_file_path(10, scan_month=6, scan_year=2024, csv_name="results", as_str=True)
    assert isinstance(path_obj, Path)
    assert isinstance(path_str, str)
    assert "results.csv" in str(path_obj)
    assert "results.csv" in path_str


def test_convert_to_categorical_basic():
    """Test categorical conversion and fuzzy grouping."""
    df = pd.DataFrame({
        "sub_status": ["ok", "OK", "fail", "Fail"],
        "front": ["A", "B", "C", "D"],
        "occupancy": [1, 2, 3, 4]
    })

    orig_cat = CONFIG["columns_to_categorical"]
    orig_fuzzy = CONFIG["columns_to_fuzzy_match"]
    orig_thresh = CONFIG["fuzzy_match_threshold"]

    CONFIG["columns_to_categorical"] = ["front", "occupancy"]
    CONFIG["columns_to_fuzzy_match"] = ["sub_status"]
    CONFIG["fuzzy_match_threshold"] = 80

    result = convert_to_categorical(df)
    assert result["sub_status"].dtype.name == "category"
    assert result["front"].dtype.name == "category"
    assert result["occupancy"].dtype.name == "category"
    assert set(result["sub_status"].cat.categories) == {"ok", "fail"}

    CONFIG["columns_to_categorical"] = orig_cat
    CONFIG["columns_to_fuzzy_match"] = orig_fuzzy
    CONFIG["fuzzy_match_threshold"] = orig_thresh
