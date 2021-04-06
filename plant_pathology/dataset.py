# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/01_dataset.ipynb (unless otherwise specified).

__all__ = ['get_datablock', 'get_dls', 'get_dls_all_in_1']

# Cell
from fastai.vision.all import *
from .utils import *
from typing import *

# Cell
def get_datablock(path: Path, df: pd.DataFrame, presize: int,
                  resize: int, val_fold: int=4) -> DataBlock:
    def get_y(row): return df.columns[row==1][0]
    return DataBlock(blocks=(ImageBlock, CategoryBlock),
                get_x=ColReader("image_id", pref=path/'images', suff=".jpg"),
                get_y=get_y,
                splitter=MaskSplitter(df["fold"]==val_fold),
                item_tfms=Resize(presize),
                batch_tfms=aug_transforms(mult=1.5, max_rotate=22.5, min_zoom=0.9,
                                         size=resize, min_scale=0.5, flip_vert=True,
                                         max_zoom=1.2))

# Cell
def get_dls(path: Path, df: pd.DataFrame, presize: Union[tuple, int]=(682, 1024),
            resize: int=256, bs: int=256, val_fold: int=4) -> DataLoaders:
    return get_datablock(path, df, presize, resize, val_fold).dataloaders(df, bs=bs)

# Cell
@delegates(get_dls)
def get_dls_all_in_1(data_path: Path, pseudo_labels_path: str=None, **kwargs) -> DataLoaders:
    path, df = load_data(data_path, with_folds=True, pseudo_labels_path=pseudo_labels_path)
    return get_dls(path, df, **kwargs)