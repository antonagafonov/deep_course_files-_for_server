import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.optim.lr_scheduler import ExponentialLR
import torch.optim as optim
from torchvision.datasets import CIFAR10
from hw2.mlp import MLP
from cs236781.train_results import FitResult
from typing import Any, Tuple, Callable, Optional, cast
from .cnn import CNN, ResNet, YourCNN

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
    "ycn": YourCNN,
}

from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
    device: Optional[torch.device] = None
):
    # TODO:
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    #  - Use the validation set for threshold selection.
    #  - Set optimal threshold and evaluate one epoch on the test set.
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    #  Note: use print_every=0, verbose=False, plot=False where relevant to prevent
    #  output from this function.
    # ====== YOUR CODE: ======
    
    #  - Create a BinaryClassifier model.
    #  - Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.
    
    # set and define device (cuda0 or cpu)
    
    model = BinaryClassifier(
                            model=MLP(
                            in_dim=2,
                            dims=[*[width]*depth, 2],
                            nonlins=[*['tanh']*depth, 'logsoftmax'] #logsoftmax for NLLLoss
                            ),
                            threshold=0.5,
                            ).to(device)
    # print(f'{width=},{depth=},{list(model.parameters())=}')
    loss_fn = torch.nn.NLLLoss
    # loss_fn = torch.nn.CrossEntropyLoss
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=5e-4, weight_decay=0.01, momentum=0.99)
    # scheduler = ExponentialLR(optimizer, gamma=0.9)
    trainer = ClassifierTrainer(model, loss_fn, optimizer,device=device)
    
    # fit before thresh
    actual_num_epochs, train_loss, train_acc, valid_loss, valid_acc = trainer.fit(dl_train, dl_valid,
                              num_epochs=n_epochs, print_every=0,early_stopping = 10,verbose=False)
    
    #  - Use the validation set for threshold selection.
    optimal_thresh = select_roc_thresh(model, *dl_valid.dataset.tensors,plot=True,device=device)
    model.threshold = optimal_thresh
    # print(f'{optimal_thresh=}')
    thresh = optimal_thresh
    #  - Set optimal threshold and evaluate one epoch on the test set.
    # dl_train_full = ConcatDataset([dl_train, dl_valid]) 
    actual_num_epochs, full_train_loss, full_train_acc, test_loss, test_acc = trainer.fit(dl_train,
                      dl_test, num_epochs=1, print_every=0,verbose=False)
    # print('Printing ACCURACIES')
    # print(f"{(full_train_acc)=},{(test_acc)=}")
    # print(f"{len(full_train_acc)=},{len(test_acc)=}")
    # print(f"{type(full_train_acc)=},{type(test_acc)=}")
    # ========================
    
    #  - Return the model, the optimal threshold value, the accuracy on the validation
    #    set (from the last epoch) and the accuracy on the test set (from a single
    #    epoch).
    return model, thresh, valid_acc[-1], test_acc[0]


def cnn_experiment(
    run_name,
    out_dir="./results",
    seed=None,
    device=None,
    # Training params
    bs_train=128,
    bs_test=None,
    batches=100,
    epochs=5,
    early_stopping=3,
    checkpoints=None,
    lr=1e-3,
    reg=1e-3,
    # Model params
    filters_per_layer=[64],
    layers_per_block=2,
    pool_every=2,
    hidden_dims=[1024],
    model_type="cnn",
    # You can add extra configuration for your experiments here
    **kw,
):
    """
    Executes a single run of a Part3 experiment with a single configuration.

    These parameters are populated by the CLI parser below.
    See the help string of each parameter for it's meaning.
    """
    if not seed:
        seed = random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    if not bs_test:
        bs_test = max([bs_train // 4, 1])
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    ds_test = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Experement is running with device = {device}')
        
    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]

    # TODO: Train
    #  - Create model, loss, optimizer and trainer based on the parameters.
    #    Use the model you've implemented previously, cross entropy loss and
    #    any optimizer that you wish.
    #  - Run training and save the FitResults in the fit_res variable.
    #  - The fit results and all the experiment parameters will then be saved
    #   for you automatically.
    fit_res = None
    # ====== YOUR CODE: ======
    # in size
    x0,_ = ds_train[0]
    in_size = x0.shape
    num_classes = 10
    print(f'{model_type=}')
    # define test params

    if model_type == "resnet":   
        test_params = dict(  
                        in_size = in_size, out_classes=num_classes, channels=filters_per_layer*layers_per_block,
                        pool_every=pool_every, hidden_dims=hidden_dims,
                        activation_type='lrelu', activation_params=dict(negative_slope=0.01),
                        pooling_type='avg', pooling_params=dict(kernel_size=2),
                        batchnorm=True, dropout=0.1,
                        bottleneck=False
                        )
    elif model_type == 'cnn':
        test_params = dict(
                    in_size = in_size, out_classes = num_classes, 
                    channels=filters_per_layer*layers_per_block,
                    pool_every=pool_every, hidden_dims=hidden_dims,
                    conv_params=dict(kernel_size=3, stride=1, padding=1),
                    pooling_params=dict(kernel_size=2)
                    )
    elif model_type == "ycn":
        pass
    
    # Loaders
    dl_train = DataLoader(ds_train, batch_size=bs_train,shuffle=True, num_workers=1)
    dl_test  = DataLoader(ds_test, batch_size=bs_test,shuffle=False, num_workers=1)
    
    #  - Create model, loss, optimizer and trainer based on the parameters.        
    model = ArgMaxClassifier(model=model_cls(**test_params))
    model = model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss  # One of the torch.nn losses
    # Arguments for SGD optimizer
    # optimizer = torch.optim.SGD(params=model.parameters(), lr=lr, weight_decay=0.01, momentum=0.99)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, betas=(0.9, 0.999), 
                                 eps=1e-08, weight_decay=0.01, amsgrad=False)
    trainer = ClassifierTrainer(model, loss_fn, optimizer, device)
    # fit before thresh
    fit_res = trainer.fit(dl_train, dl_test,num_epochs=epochs,max_batches=batches, print_every=0,
                          early_stopping = early_stopping,verbose=False)                                   
    # ========================

    save_experiment(run_name, out_dir, cfg, fit_res)


def save_experiment(run_name, out_dir, cfg, fit_res):
    output = dict(config=cfg, results=fit_res._asdict())

    cfg_LK = (
        f'L{cfg["layers_per_block"]}_K'
        f'{"-".join(map(str, cfg["filters_per_layer"]))}'
    )
    output_filename = f"{os.path.join(out_dir, run_name)}_{cfg_LK}.json"
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(output, f, indent=2)

    print(f"*** Output file {output_filename} written")


def load_experiment(filename):
    with open(filename, "r") as f:
        output = json.load(f)

    config = output["config"]
    fit_res = FitResult(**output["results"])

    return config, fit_res


def parse_cli():
    p = argparse.ArgumentParser(description="CS236781 HW2 Experiments")
    sp = p.add_subparsers(help="Sub-commands")

    # Experiment config
    sp_exp = sp.add_parser(
        "run-exp", help="Run experiment with a single " "configuration"
    )
    sp_exp.set_defaults(subcmd_fn=cnn_experiment)
    sp_exp.add_argument(
        "--run-name", "-n", type=str, help="Name of run and output file", required=True
    )
    sp_exp.add_argument(
        "--out-dir",
        "-o",
        type=str,
        help="Output folder",
        default="./results",
        required=False,
    )
    sp_exp.add_argument(
        "--seed", "-s", type=int, help="Random seed", default=None, required=False
    )
    sp_exp.add_argument(
        "--device",
        "-d",
        type=str,
        help="Device (default is autodetect)",
        default=None,
        required=False,
    )

    # # Training
    sp_exp.add_argument(
        "--bs-train",
        type=int,
        help="Train batch size",
        default=128,
        metavar="BATCH_SIZE",
    )
    sp_exp.add_argument(
        "--bs-test", type=int, help="Test batch size", metavar="BATCH_SIZE"
    )
    sp_exp.add_argument(
        "--batches", type=int, help="Number of batches per epoch", default=100
    )
    sp_exp.add_argument(
        "--epochs", type=int, help="Maximal number of epochs", default=100
    )
    sp_exp.add_argument(
        "--early-stopping",
        type=int,
        help="Stop after this many epochs without " "improvement",
        default=3,
    )
    sp_exp.add_argument(
        "--checkpoints",
        type=int,
        help="Save model checkpoints to this file when test " "accuracy improves",
        default=None,
    )
    sp_exp.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
    sp_exp.add_argument("--reg", type=float, help="L2 regularization", default=1e-3)

    # # Model
    sp_exp.add_argument(
        "--filters-per-layer",
        "-K",
        type=int,
        nargs="+",
        help="Number of filters per conv layer in a block",
        metavar="K",
        required=True,
    )
    sp_exp.add_argument(
        "--layers-per-block",
        "-L",
        type=int,
        metavar="L",
        help="Number of layers in each block",
        required=True,
    )
    sp_exp.add_argument(
        "--pool-every",
        "-P",
        type=int,
        metavar="P",
        help="Pool after this number of conv layers",
        required=True,
    )
    sp_exp.add_argument(
        "--hidden-dims",
        "-H",
        type=int,
        nargs="+",
        help="Output size of hidden linear layers",
        metavar="H",
        required=True,
    )
    sp_exp.add_argument(
        "--model-type",
        "-M",
        choices=MODEL_TYPES.keys(),
        default="cnn",
        help="Which model instance to create",
    )

    parsed = p.parse_args()

    if "subcmd_fn" not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f"*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}")
    subcmd_fn(**vars(parsed_args))
