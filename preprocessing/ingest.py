import pandas as pd
from pathlib import Path
from datetime import date, datetime
import json
import yaml
import logging
from rapidfuzz import process, fuzz

# =========================================================
# DEFAULT CONFIGURATION
# =========================================================
DEFAULT_CONFIG = {
    # ----------------------------
    # DATA PATH SETTINGS
    # ----------------------------
    "data_path": "../data/raw_data",           # Directory containing raw CSV/JSON scan data
    "output_dir": "../data/processed_data",         # Directory where processed CSVs will be written

    # ----------------------------
    # OUTPUT CSV SETTINGS
    # ----------------------------
    "warehouse_csv_name": "warehouse_layout.csv",   # CSV filename for warehouse layout
    "subset_csv_name": "subset_merged_timesteps.csv",  # CSV for intermediate merged subset
    "final_parquet_name": "timeseries_data.parquet",   # parquet for final processed timeseries data

    "write_subset_csv": True,       # Whether to write the subset CSV
    "ask_before_overwrite": True,   # Prompt user before overwriting existing CSVs

    # ----------------------------
    # CATEGORICAL / TEXT PROCESSING
    # ----------------------------
    "columns_to_categorical": [
        "front",      # Rack face front column
        "Occupancy",  # Occupancy value column
        "Expected",   # Expected items column
        "Scanned",    # Scanned items column
        "Status",     # Status column
        "Client"      # Client name column
    ],

    "columns_to_fuzzy_match": [
        "Sub Status"  # Columns to apply fuzzy text grouping for similar entries
    ],

    # ----------------------------
    # DYNAMIC COLUMNS FOR MERGE
    # ----------------------------
    "dynamic_columns": [
        "Aisle",                  # Aisle identifier
        "Occupancy",              # Occupancy column
        "Expected",               # Expected items
        "Scanned",                # Scanned items
        "Status",                 # Status column
        "Sub Status",             # Sub-status column
        "Result Age",             # Age of result
        "Client",                 # Client name
        "Barcode",                # Barcode scanned
        "Barcode actually scanned at",  # Timestamp/location scanned
        "Barcode should be at"          # Expected location
    ],

    # Warehouse headers extracted from JSON
    "warehouse_headers": [
        "Location",
        "id",
        "name",
        "type",
        "front",
        "column",
        "shelf",
        "min_x",
        "min_y",
        "min_z",
        "max_x",
        "max_y",
        "max_z"
    ],

    # ----------------------------
    # SCAN DATA SETTINGS
    # ----------------------------
    "dates_to_merge": [10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21],  # Days of the month to merge
    "scan_month": 6,        # Month of scan data
    "scan_year": 2024,      # Year of scan data
    "fuzzy_match_threshold": 85,  # Similarity threshold for fuzzy text grouping (0-100)

    # ----------------------------
    # LOGGING
    # ----------------------------
    "disable_logging": True   # Set True to disable logging (e.g., during unit tests)
}


# =========================================================
# CONFIG LOADING
# =========================================================
def load_config(config_path: Path = Path("../config/data_ingest_config.yaml")) -> dict:
    """Load configuration from YAML and merge with defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path.exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)

    config["output_dir"] = Path(config["output_dir"]).resolve()
    config["data_path"] = Path(config["data_path"]).resolve()

    return config


CONFIG = load_config()


# =========================================================
# LOGGING SETUP
# =========================================================
def setup_logging():
    """Setup logging to file and console unless disabled in config."""
    if CONFIG.get("disable_logging", False):
        return

    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = CONFIG["output_dir"] / f"ingest_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Log file: {log_file}")


# =========================================================
# UTILITY FUNCTIONS
# =========================================================
def confirm_overwrite(file_path: Path) -> bool:
    """Ask user whether to overwrite existing file."""
    if not CONFIG["ask_before_overwrite"]:
        return True
    if not file_path.exists():
        return True
    choice = input(f"File '{file_path}' exists. Overwrite? [y/N]: ").strip().lower()
    return choice == "y"


def ensure_output_dir():
    """Ensure the output directory exists."""
    CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)


def get_csv_file_path(scan_day: int, scan_month: int = 6,
                      scan_year: int = 2024, csv_name: str = "results",
                      as_str: bool = False) -> Path | str:
    """Construct the path to a scan CSV for a given day."""
    scan_date = date(scan_year, scan_month, scan_day)
    csv_name += ".csv"
    path = CONFIG["data_path"] / str(scan_date) / csv_name
    return str(path) if as_str else path


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns: lowercase + underscores instead of spaces."""
    df = df.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df


# =========================================================
# TEXT CLEANING / FUZZY GROUPING
# =========================================================
def group_similar_text(series: pd.Series, threshold: int = 85,
                       scorer=fuzz.token_sort_ratio) -> pd.Series:
    """Group similar text entries using fuzzy matching."""
    series = series.astype(str).str.strip().str.lower()
    unique_vals = series.unique().tolist()
    mapping = {}
    for val in unique_vals:
        if val in mapping:
            continue
        matches = process.extract(val, unique_vals, scorer=scorer)
        for match_val, score, _ in matches:
            if score >= threshold:
                mapping[match_val] = val
    return series.map(mapping)


# =========================================================
# CORE PROCESSING FUNCTIONS
# =========================================================
def read_warehouse_json(warehouse_layout_fn: str = "warehouse-layout") -> pd.DataFrame:
    """Read warehouse layout JSON and optionally write CSV."""
    file_path = CONFIG["data_path"] / (warehouse_layout_fn + ".json")
    logging.info(f"Reading warehouse layout JSON: {file_path}")

    with open(file_path) as f:
        data = json.load(f)["rack_face_areas"]

    rows = []
    for entry in data:
        parent_id = entry["id"]
        parent_name = entry["name"]
        for loc in entry.get("locations", []):
            rows.append({
                "Location": loc["name"], "id": parent_id, "name": parent_name,
                "type": loc["type"], "front": loc["front"], "column": loc["column"],
                "shelf": loc["shelf"], "min_x": loc["bounds"]["min"]["x"],
                "min_y": loc["bounds"]["min"]["y"], "min_z": loc["bounds"]["min"]["z"],
                "max_x": loc["bounds"]["max"]["x"], "max_y": loc["bounds"]["max"]["y"],
                "max_z": loc["bounds"]["max"]["z"]
            })

    df = pd.DataFrame(rows)
    df = standardize_column_names(df)

    output_path = CONFIG["output_dir"] / CONFIG["warehouse_csv_name"]
    if confirm_overwrite(output_path):
        df.to_csv(output_path, index=False)
        logging.info(f"Warehouse layout CSV written: {output_path}")
    return df


def merge_csvs(base_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily scan CSVs with warehouse layout."""
    dynamic_cols = CONFIG["dynamic_columns"]
    all_timesteps = []
    csv_files = [
        get_csv_file_path(day, scan_month=CONFIG["scan_month"],
                          scan_year=CONFIG["scan_year"])
        for day in CONFIG["dates_to_merge"]
    ]
    base_df = standardize_column_names(base_df)

    for i, file in enumerate(csv_files, start=1):
        logging.info(f"Reading scan CSV for timestep {i}: {file}")
        df = pd.read_csv(file, low_memory=False)
        df = standardize_column_names(df)

        if "location" not in df.columns:
            raise KeyError("Expected 'location' column in scan CSV.")

        df = df[df["location"].isin(base_df["location"])]
        cols_to_keep = ["location"] + [
            col.lower().replace(" ", "_")
            for col in dynamic_cols
            if col.lower().replace(" ", "_") in df.columns
        ]
        df = df[cols_to_keep].copy()
        df["timestep"] = i
        all_timesteps.append(df)

    merged_long_df = pd.concat(all_timesteps, ignore_index=True)
    merged_long_df = merged_long_df.merge(base_df, on="location", how="left")
    return standardize_column_names(merged_long_df)


def downselect_relevant_columns(merged_df: pd.DataFrame) -> pd.DataFrame:
    """Optionally write subset CSV."""
    output_path = CONFIG["output_dir"] / CONFIG["subset_csv_name"]
    if CONFIG["write_subset_csv"]:
        if confirm_overwrite(output_path):
            merged_df.to_csv(output_path, index=False)
            logging.info(f"Subset CSV written: {output_path}")
    return merged_df


def convert_to_categorical(subset_df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to categorical and apply fuzzy grouping."""
    subset_df = subset_df.copy()
    columns_to_clean = [
        col for col in subset_df.columns
        if any(col.startswith(prefix.lower().replace(" ", "_"))
               for prefix in CONFIG["columns_to_fuzzy_match"])
    ]

    for col in columns_to_clean:
        subset_df[col] = group_similar_text(
            subset_df[col], threshold=CONFIG["fuzzy_match_threshold"]
        )
        subset_df[col] = subset_df[col].astype("category")

    for col in CONFIG["columns_to_categorical"]:
        col_clean = col.lower().replace(" ", "_")
        if col_clean in subset_df.columns:
            subset_df[col_clean] = subset_df[col_clean].astype("category")
    return subset_df


def ingest_data() -> pd.DataFrame:
    """Run the full ingestion, cleaning, and processing pipeline."""
    warehouse_df = read_warehouse_json()
    merged_df = merge_csvs(warehouse_df)
    subset_df = downselect_relevant_columns(merged_df)
    subset_df = convert_to_categorical(subset_df)

    final_output_path = CONFIG["output_dir"] / CONFIG["final_csv_name"]
    if confirm_overwrite(final_output_path):
        #subset_df.to_csv(final_output_path, index=False)
        subset_df.to_parquet(final_output_path, engine='pyarrow', index=False)
        logging.info(f"Final Parquet written: {final_output_path}")
    return subset_df


if __name__ == "__main__":
    setup_logging()
    output_df = ingest_data()
    print(output_df.dtypes)
