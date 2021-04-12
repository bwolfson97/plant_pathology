# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/07_pretrained_models.ipynb (unless otherwise specified).

__all__ = ['MODELS', 'get_model']

# Cell
import numpy as np
from fastai.data.external import untar_data
from fastai.learner import Learner, load_learner
from fastcore.basics import patch

# Cell
MODELS = {
    "resnet18_2021-04-08": "https://github.com/bwolfson97/plant_pathology/releases/download/v0.1.1-alpha/resnet18_2021-04-08.tar.gz"
}

# Cell
def get_model(model_name: str):
    """Downloads and builds pretrained model.

    `model_name` must be in `MODELS` dict.
    """
    try:
        url = MODELS[model_name]
    except KeyError:
        raise KeyError(f"Invalid model name. Received: {model_name}")

    pickle_file = untar_data(url)
    return load_learner(pickle_file)

# Cell
@patch
def predict_leaf(self: Learner, image, decimals: int = 2):
    """Predict on image and return predicted class and decoded probabilities.

    Rounds probabilities to `decimals` decimal places.
    """
    predicted_class, _, probabilities = self.predict(image)

    # Round probabilities
    probabilities = np.around(probabilities.tolist(), decimals=decimals)

    # Decode probabilities using class names
    decoded_probabilities = dict(zip(self.dls.vocab, probabilities))

    # Format results
    return {
        "predicted_class": predicted_class,
        "probabilities": decoded_probabilities
    }