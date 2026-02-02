# save_diagnostic_results.py 
from pathlib import Path
from datetime import date
import pandas as pd

def _ensure_results_dir(save_root: str | Path, report_date: str | None = None, report_name: str | None = None) -> Path:
    """
    Create (or return) the directory where per-test CSVs will be saved.

    save_root: base directory where report lives (e.g. report save_dir)
    report_date: optional 'YYYY-MM-DD' string to use in folder name (if your report already has a date)
                 if None uses today's date.
    Returns Path object for created directory.
    """
    save_root = Path(save_root)
    if report_date is None:
        report_date = date.today().isoformat()  # e.g. '2026-01-28'
    if report_name is None:
        dir_name = f"DiagnosticResults_{report_date}"
    else:
        dir_name = f"{report_name}_{report_date}"
    results_dir = save_root / dir_name
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir

def save_test_results(results, test_name: str, save_root: str | Path,
                      feature_names: list | None = None, report_date: str | None = None, report_name: str | None = None) -> str:
    """
    Save results from a single diagnostic test into a CSV inside a timestamped directory.

    - results: Accepts either:
        * dict mapping comparison_key -> scalar (overall single-number results) or
        * dict mapping comparison_key -> 1D array-like of per-feature results
        * or a DataFrame already (written directly)
        * or a mapping of comparison_key -> dict (then flattened into a DataFrame row)
    - test_name: name used for the CSV filename (e.g. 'Levene_Test' or 'KS_Test')
    - save_root: folder under which DiagnosticResults_YYYY-MM-DD/ will be created
    - feature_names: optional list of column names for featurewise results (lowercase 'feature_names')
                     if None and an array is present, a fallback 'feature_0'.. will be used.
    - report_date: optional 'YYYY-MM-DD' to control directory naming (your report timestamp)

    Minimal and defensive: doesn't change your results structure, only converts it into DataFrame(s)
    and writes one CSV named f"{test_name}.csv".
    """
    if report_name is None:
        results_dir = _ensure_results_dir(save_root, report_date)
    else:
        results_dir = _ensure_results_dir(save_root, report_date, report_name)
       
    csv_path = results_dir / f"{test_name}.csv"

    # If already a DataFrame, just save
    if isinstance(results, pd.DataFrame):
        results.to_csv(csv_path, index=True)
        return str(csv_path)

    # If results is a mapping/dict
    if isinstance(results, dict):
        # detect whether values are scalar or array-like
        rows = {}
        array_detected = False
        for comp_key, value in results.items():
            # stringify comparison key for the row label
            row_label = str(comp_key)

            # if value is dict -> flatten to columns per key
            if isinstance(value, dict):
                rows[row_label] = value
                continue

            # if it's scalar or list-like
            try:
                # use pandas to help detect array-like
                ser = pd.Series(value)
                if ser.ndim == 0 or (ser.size == 1 and ser.index.tolist() == [0]):
                    # scalar-like
                    rows[row_label] = {"result": float(ser.iloc[0])}
                else:
                    # array-like: will produce columns per feature
                    array_detected = True
                    rows[row_label] = ser.values  # store array for special handling
            except Exception:
                # fallback: treat as scalar string
                rows[row_label] = {"result": str(value)}

        if array_detected:
            # build DataFrame where columns are feature names (if provided),
            # otherwise fallback to feature_0, feature_1, ...
            # We expect all array-like rows to have same length; if not, pandas will align by index and
            # produce NaNs where lengths differ (transparent behavior).
            arrays = {rlabel: rows[rlabel] for rlabel in rows if not isinstance(rows[rlabel], dict)}
            scalars = {rlabel: rows[rlabel] for rlabel in rows if isinstance(rows[rlabel], dict)}

            # convert arrays to DataFrame
            arr_df = pd.DataFrame.from_dict({k: list(v) for k, v in arrays.items()}, orient="index")
            # set column names
            if feature_names:
                if len(feature_names) != arr_df.shape[1]:
                    # if mismatch, fall back to generic names
                    col_names = [f"feature_{i}" for i in range(arr_df.shape[1])]
                else:
                    col_names = list(feature_names)
            else:
                col_names = [f"feature_{i}" for i in range(arr_df.shape[1])]
            arr_df.columns = col_names
            # If there are also scalar rows, add them as extra columns (e.g. 'result' column) to arr_df
            if scalars:
                scalar_df = pd.DataFrame.from_dict(scalars, orient="index")
                # join scalars to arr_df (outer join so all rows are preserved)
                final_df = arr_df.join(scalar_df, how="outer")
            else:
                final_df = arr_df

            final_df.index.name = "comparison"
            final_df.to_csv(csv_path, index=True)
            return str(csv_path)

        else:
            # purely scalar/dict rows -> convert to DataFrame where each row is a comparison
            df = pd.DataFrame.from_dict(rows, orient="index")
            df.index.name = "comparison"
            df.to_csv(csv_path, index=True)
            return str(csv_path)

    # Otherwise, try to coerce to DataFrame and save
    try:
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=True)
        return str(csv_path)
    except Exception as exc:
        raise ValueError(f"Unable to save results for {test_name}: {exc}")