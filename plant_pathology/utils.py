# AUTOGENERATED! DO NOT EDIT! File to edit: 00_utils.ipynb (unless otherwise specified).

__all__ = ['load_data', 'create_folds']

# Cell
from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold

# Cell
def load_data(with_folds=True) -> (Path, pd.DataFrame):
    path = Path("/home/jupyter/kaggle/plant-pathology/data/plant-pathology-2020/")
    path_train = path/('train_folds.csv' if with_folds else 'train.csv')
    df_train = pd.read_csv(path_train)
    return path, df_train

# Cell
def create_folds(path: Path, df: pd.DataFrame, prn_stats: bool = False) -> pd.DataFrame:
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    lbls = df.apply(lambda r: df.columns[r==1].item(), axis=1)

    # Create 5 folds
    kf = StratifiedKFold(n_splits=5)
    for fold, (train_idxs, val_idxs) in enumerate(kf.split(df, lbls.values)):
        print(f"Fold {fold}: {len(train_idxs)/len(df)}, {len(val_idxs)/len(df)}")
        df.loc[val_idxs, "fold"] = fold

    if prn_stats: print(df.groupby("fold").describe())

    # Save to file
    df.to_csv(path/"train_folds.csv", index=False)
    return df