# Plant Pathology Classifier
> A neural network classifies images of leaves as healthy or diseased!

<img alt="A picture of a leaf" src="nbks/images/leaf.jpg" width="700" style="max-width: 700px">

Recently, I've been learning about [Fast.ai](https://docs.fast.ai/) and [PyTorch](https://pytorch.org/) in my free time and wanted to apply my knowledge. I'm trying to learn how to win Kaggle competitions, so I decided to build a model for the completed Kaggle [Plant Pathology Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview).

I built this model using [Nbdev](https://nbdev.fast.ai/), which provides an [literate programming](https://en.wikipedia.org/wiki/Literate_programming) environment as originally envisioned by Donald Knuth. This means the notebooks in the `nbks` folder are the library's "source code". They get converted into regular python files, a full [documentation site](https://bwolfson97.github.io/plant_pathology/), and contain unit and functional tests, all in one place.

## Install

`pip install plant-pathology`

## How to use

### Inference example

```python
from plant_pathology.pretrained_models import get_model

model = get_model("resnet18_2021-04-08")
prediction = model.predict_leaf("images/leaf.jpg")
prediction
```








    {'predicted_class': 'rust',
     'probabilities': {'healthy': 0.05,
      'multiple_diseases': 0.04,
      'rust': 0.88,
      'scab': 0.03}}



### Training

### Run training script

```
❯ python -m plant_pathology.train -h
usage: train.py [-h] [--epochs EPOCHS] [--lr LR] [--frz FRZ] [--pre PRE [PRE ...]] [--re RE] [--bs BS] [--smooth] [--arch ARCH] [--dump] [--log] [--save] [--mixup MIXUP] [--tta] [--fp16] [--do_eval] [--val_fold VAL_FOLD] [--pseudo PSEUDO] [--export]
                path

positional arguments:
  path                 Path to data dir

optional arguments:
  -h, --help           show this help message and exit
  --epochs EPOCHS      Number of unfrozen epochs (default: 1)
  --lr LR              Initial learning rate (default: 0.0003)
  --frz FRZ            Number of frozen epochs (default: 1)
  --pre PRE [PRE ...]  Image presize (default: (682, 1024))
  --re RE              Image resize (default: 256)
  --bs BS              Batch size (default: 256)
  --smooth             Label smoothing? (default: False)
  --arch ARCH          Architecture (default: resnet18)
  --dump               Don't train, just print model (default: False)
  --log                Log w/ W&B (default: False)
  --save               Save model based on RocAuc (default: False)
  --mixup MIXUP        Mixup (0.4 is good) (default: 0.0)
  --tta                Test-time augmentation (default: False)
  --fp16               Mixed-precision training (default: False)
  --do_eval            Evaluate model and save predictions CSV (default: False)
  --val_fold VAL_FOLD  Don't go cross-validation, just do 1 fold (or pass 9 to train on all data)
  --pseudo PSEUDO      Path to pseudo labels to train on
  --export             Export learner(s) to export_val_on_{fold}.pkl (default: False)
```

## Testing

To run all the tests in the notebooks in parallel, just run `nbdev_test_nbs` from the terminal! :)

## Web App

I deployed the classifier as a simple [web app](https://plant-pathology-classifier.herokuapp.com/) using [Streamlit](https://streamlit.io/) and [Heroku](https://www.heroku.com/). Note, it may take a few minutes to start up.

The code for the app is [here](https://github.com/bwolfson97/plant_pathology_app).
