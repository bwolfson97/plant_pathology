# AUTOGENERATED! DO NOT EDIT! File to edit: nbks/03_train.ipynb (unless otherwise specified).

__all__ = ['train', 'train_cv']

# Cell
from .dataset import *
from .evaluate import *

from fastai.vision.all import *
from fastcore.script import *
from fastai.callback.wandb import *
import wandb

# Cell
def train(
    epochs: int, lr: float, frz: int=1, pre: int=800, re: int=256,
    bs: int=256, fold: int=4, smooth: bool=True,
    arch: str='resnet18', dump: bool=False, log: bool=False, mixup: float=0.,
    fp16: bool=False, dls: DataLoaders=None,
 ):
    # Prep Data, Opt, Loss, Arch
    if dls is None: dls = get_dls_all_in_1(presize=pre, resize=re, bs=bs, val_fold=fold)
    if log: wandb.init(project="plant-pathology")
    if smooth: loss_func = LabelSmoothingCrossEntropyFlat()
    else:      loss_func = CrossEntropyLossFlat()
    m = globals()[arch]

    # Add callbacks
    cbs = [WandbCallback(), SaveModelCallback()] if log else []
    if mixup: cbs.append(MixUp(mixup))

    # Build learner
    learn = cnn_learner(dls, m, loss_func=loss_func,
                    metrics=[accuracy, RocAuc()], cbs=cbs)
    if dump: print(learn.model); exit()
    if fp16: learn.to_fp16()

    # Train
    learn.freeze()
    learn.fit_one_cycle(frz, lr)
    learn.unfreeze()
    learn.fit_one_cycle(epochs, slice(lr/100, lr/2))  # Explore other divs
    return learn

# Cell
@call_parse
def train_cv(
    epochs:   Param("Number of unfrozen epochs", int),
    lr:       Param("Initial learning rate", float),
    frz:      Param("Number of frozen epochs", int)=1,
    pre:      Param("Presize", int)=800,
    re:       Param("Resize", int)=256,
    bs:       Param("Batch size", int)=256,
    smooth:   Param("Label smoothing?", bool_arg)=False,
    arch:     Param("Architecture", str)='resnet18',
    dump:     Param("Print model", bool_arg)=False,
    log:      Param("Log w/ W&B", bool_arg)=False,
    mixup:    Param("Mixup", float)=0.0,
    tta:      Param("Test-time augmentation", bool_arg)=False,
    fp16:     Param("Use mixed-precision", bool_arg)=False,
    eval_dir: Param("Evaluate model, save results in dir", Path)=None,
):
    print(locals())
    scores = []
    for fold in range(5):
        print(f"\nTraining on fold {fold}")
        learn = train(epochs, lr, frz=frz, pre=pre, re=re, bs=bs, smooth=smooth,
                      arch=arch, dump=dump, log=log, fold=fold, mixup=mixup,
                      fp16=fp16,)
        scores.append(learn.final_record)

        # Create submission file for this model
        if eval_dir:
            eval_dir = Path(eval_dir)
            evaluate(learn, eval_dir/f"predictions_fold_{fold}.csv")

        # Delete learner to avoid OOM
        del learn
    scores = np.array(scores)
    print(f"Scores: {scores}\n")
    print(f"Mean: {scores.mean(0)}")