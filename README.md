
# Plant Pathology Classifier
> A neural network that takes in an image of a leaf and classifies the leaf as healthy or diseased!


![A leaf](nbks/../example_image.jpg)

Recently, I've been learning about [Fast.ai](https://docs.fast.ai/) and [PyTorch](https://pytorch.org/) in my free time and wanted to apply my knowledge. I'm trying to learn how to win Kaggle competitions, so I decided to build a model for the completed Kaggle [Plant Pathology Competition](https://www.kaggle.com/c/plant-pathology-2020-fgvc7/overview).

I built this model using [Nbdev](https://nbdev.fast.ai/), which provides an [literate programming](https://en.wikipedia.org/wiki/Literate_programming) environment as originally envisioned by Donald Knuth. This means the notebooks in the `nbks` folder are the library's "source code". They get converted into regular python files, a full documentation site, and contain unit and functional test, all in one place.

## Install

`pip install plant_pathology`

## How to use

### Inference example

```python
from plant_pathology.pretrained_models import get_model

model = get_model("**FIXME**")
prediction = model.predict("/path/to/image.jpg")
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-8-14608c339b37> in <module>
    ----> 1 from plant_pathology.pretrained_models import get_model
          2 
          3 model = get_model("**FIXME**")
          4 prediction = model.predict(image_path)


    ModuleNotFoundError: No module named 'plant_pathology.pretrained_models'


**FIXME**: Colab notebook with an example: ![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

### Training

### Run training script

```
python -m plant_pathology.train -h
usage: train.py [-h] [--frz FRZ] [--pre PRE] [--re RE] [--bs BS] [--smooth] [--arch ARCH] [--dump] [--log] [--save] [--mixup MIXUP] [--tta] [--fp16] [--eval_dir EVAL_DIR] [--val_fold VAL_FOLD] [--pseudo PSEUDO] path epochs lr

positional arguments:
  path                 Path to data dir
  epochs               Number of unfrozen epochs
  lr                   Initial learning rate

optional arguments:
  -h, --help           show this help message and exit
  --frz FRZ            Number of frozen epochs (default: 1)
  --pre PRE            Image presize (default: (682, 1024))
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
  --eval_dir EVAL_DIR  Evaluate model and save results in dir
  --val_fold VAL_FOLD  Don't go cross-validation, just do 1 fold (or pass 9 to train on all data)
  --pseudo PSEUDO      Path to pseudo labels to train on
```

## Testing

To run the tests found in all the notebook in parallel, just run `nbdev_test_nbs` from the terminal! :)

## Web app

**TODO**
