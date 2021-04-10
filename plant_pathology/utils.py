# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/00_utils.ipynb (unless otherwise specified).

__all__ = ['load_data', 'kaggle_submit_command', 'average_preds', 'get_averaged_preds']

# Cell
from typing import List, Tuple

import pandas as pd
from fastcore.all import Path

# Cell
def load_data(data_path: Path, with_folds: bool = False, pseudo_labels_path: str = None) -> Tuple[Path, pd.DataFrame]:
    """Load data (with/without cross-validation folds) into DataFrame."""
    train_df = pd.read_csv(data_path/('train_folds.csv' if with_folds else 'train.csv'))
    if pseudo_labels_path is not None:
        # Add pseudo labels to DataFrame
        train_df = pd.concat([train_df, pd.read_csv(pseudo_labels_path)], ignore_index=True)
    return data_path, train_df

# Cell
def kaggle_submit_command() -> str:
    """Print terminal command to submit submission file."""
    print("kaggle competitions submit -c plant-pathology-2020-fgvc7 -f {submission_path} -m 'message'")

# Cell
def average_preds(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Average predictions on test examples across prediction DataFrames in `dfs`."""
    all_preds_df = pd.concat(dfs)
    avg_preds_df = all_preds_df.groupby(all_preds_df.image_id).mean()
    return avg_preds_df

# Cell
def get_averaged_preds(path: Path, verbose: bool = False) -> Path:
    """Returns DataFrame of averaged of averaged predictions of prediction CSVs in `path` dir."""
    # Load test set prediction CSVs for each of 5 CV folds
    prediction_files = list(path.glob("predictions_fold_[0-4].csv"))
    if verbose:
        print(prediction_files)
    return average_preds([pd.read_csv(fn) for fn in prediction_files])