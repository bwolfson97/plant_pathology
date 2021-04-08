# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/07_pretrained_models.ipynb (unless otherwise specified).

__all__ = ['MODELS', 'get_model']

# Cell
MODELS = {
    "resnet18_2021-04-07": ""
}

# Cell
def get_model(model_name: str):
    """Downloads and builds pretrained model."""
    # TODO: Download file from URL and untar it
    learner_state = MODELS[model_name]

    # TODO: Use load_learner to load pickle file and return learner