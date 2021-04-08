# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/02_evaluate.ipynb (unless otherwise specified).

__all__ = ['infer_on_test_set', 'format_submission', 'evaluate']

# Cell
from .config import TEST_DATA_PATH
from fastai.vision.all import *
from typing import *

# Cell
def infer_on_test_set(
    learn: Learner, path: Path, tta: bool=False, **kwargs
) -> Tensor:
    """Infers on test CSV at `path` using `learn`, optionally performing TTA."""
    df_test = pd.read_csv(path)
    test_dl = learn.dls.test_dl(df_test)
    preds, _ = (learn.tta if tta else learn.get_preds)(dl=test_dl, **kwargs)
    return preds

# Cell
def format_submission(preds: Tensor, save_path: Union[Path, str]) -> Path:
    # Build submission CSV
    image_filenames = [f"Test_{i}" for i in range(len(preds))]
    column_names = ["healthy", "multiple_diseaes", "rust", "scab"]
    submission = pd.DataFrame(preds, index=image_filenames, columns=column_names)

    # Make parent dirs
    save_path = Path(save_path)
    Path(save_path.parent).mkdir(parents=True, exist_ok=True)

    # Save submission
    submission.to_csv(save_path)
    return save_path

# Cell
def evaluate(learn: Learner, path: Path, name: str = "submission.csv", tta: bool=False) -> Path:
    """Evaluates `learn` on test CSV at `path` and saves as `name`, optionally applying TTA."""
    preds = infer_on_test_set(learn, pathtta=tta)
    return format_submission(preds, name)