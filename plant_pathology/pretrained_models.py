# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/07_pretrained_models.ipynb (unless otherwise specified).

__all__ = ['MODELS', 'get_model']

# Cell
from fastai.data.external import untar_data
from fastai.learner import load_learner
from fastcore.test import ExceptionExpected

# Cell
MODELS = {
    "resnet18_2021-04-08": "https://github.com/bwolfson97/plant_pathology/releases/download/v0.1.1-alpha/resnet18_2021-04-08.tar.gz"
}

# Cell
def get_model(model_name: str):
    """Downloads and builds pretrained model."""
    try:
        url = MODELS[model_name]
    except KeyError:
        raise KeyError(f"Invalid model name. Received: {model_name}")

    pickle_file = untar_data(url)
    return load_learner(pickle_file)