import os
import sys
import json
import torch
import random
import argparse
import itertools
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
import torch.optim as optim
from .train_results import FitResult
from datetime import datetime


from .cnn import CNN, ResNet, YourCNN
from .mlp import MLP
from .training import ClassifierTrainer
from .classifier import ArgMaxClassifier, BinaryClassifier, select_roc_thresh

DATA_DIR = os.path.expanduser("~/.pytorch-datasets")

MODEL_TYPES = {
    ###
    "cnn": CNN,
    "resnet": ResNet,
    "ycn": YourCNN,
}


def mlp_experiment(
    depth: int,
    width: int,
    dl_train: DataLoader,
    dl_valid: DataLoader,
    dl_test: DataLoader,
    n_epochs: int,
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

    #  Create a BinaryClassifier model.
    model = BinaryClassifier(
        model=MLP(in_dim=2, dims=[*[width] * depth, 2], nonlins=[*['tanh'] * depth, 'none']),
        threshold=0.5
    )

    # Train using our ClassifierTrainer for n_epochs, while validating on the
    #    validation set.

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.03)
    trainer = ClassifierTrainer(model, loss_fn, optimizer)

    fit_result = trainer.fit(dl_train, dl_valid, num_epochs=n_epochs, print_every=0)
    valid_acc = fit_result.test_acc[-1]
    # Set optimal threshold
    thresh = select_roc_thresh(model, *dl_valid.dataset.tensors, plot=False)
    model.threshold = thresh

    # evaluate one epoch on the test set.
    test_result = trainer.test_epoch(dl_test, verbose=False)
    test_acc = test_result.accuracy

    # ========================
    return model, thresh, valid_acc, test_acc


def cnn_experiment(
        run_name,
        model=None,
        out_dir="./results",
        seed=None,
        device=None,
        # Dataset
        ds_train=None,
        ds_val=None,
        ds_test=None,
        # Training params
        bs_train=128,
        bs_val=None,
        batches=None,
        epochs=100,
        early_stopping=3,
        checkpoints=None,
        lr=1e-3,
        reg=1e-3,
        features_loss_fns=None,
        features_loss_weights=None,
        label_loss_fns=None,
        label_loss_weights=None,
        optimizer=None,
        # Model params
        filters_per_layer=[64],
        layers_per_block=2,
        pool_every=2,
        hidden_dims=[1024],
        model_type="cnn",
        train_nn_space=None,
        val_nn_space=None,
        test_nn_space=None,
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
    if not bs_val:
        bs_val = max([bs_train // 4, 1])
    if checkpoints:
        now = datetime.now()

        checkpoints = checkpoints+'_'+now.strftime("%d_%m_%Y_%H_%M_%S")
    cfg = locals()

    tf = torchvision.transforms.ToTensor()
    if ds_train is None:
        ds_train = CIFAR10(root=DATA_DIR, download=True, train=True, transform=tf)
    if ds_val is None:
        ds_val = CIFAR10(root=DATA_DIR, download=True, train=False, transform=tf)

    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")
    model_cls = MODEL_TYPES[model_type]
    fit_res = None

    dl_train = torch.utils.data.DataLoader(ds_train, bs_train, shuffle=True, drop_last=True)
    dl_val = torch.utils.data.DataLoader(ds_val, bs_val, shuffle=True, drop_last=True)
    dl_test = torch.utils.data.DataLoader(ds_test, bs_val, shuffle=True, drop_last=True)
    # get some random training images
    data_iter = iter(dl_train)
    features1, _, _ = data_iter.next()

    in_size = []
    in_size += features1[0].shape
    out_size = in_size[0]
    while len(in_size) < 3:
        in_size+=[1]

    channels = [layer for layer in filters_per_layer for i in range(layers_per_block)]

    if model is None:
        model = model_cls(in_size=in_size, out_classes=len(ds_train.classes),
                          channels=channels, pool_every=pool_every, hidden_dims=hidden_dims,
                          **kw)

    # Writer will output to ./runs/ directory by default
    model = model.to(device)
    writer = SummaryWriter()
    writer.add_graph(model, features1.to(device))

    print("Using Device: ", device)

    if optimizer is "identity":
        optimizer = None
    elif optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # TODO tweak scheduler parameters
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, threshold=0.0001,
                                                     threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08,
                                                     verbose=True)

    trainer = ClassifierTrainer(model, features_loss_fns, features_loss_weights, label_loss_fns, label_loss_weights,
                                optimizer, scheduler, device, train_nn_space=train_nn_space,
                                val_nn_space=val_nn_space,
                                test_nn_space=test_nn_space)
    fit_res = trainer.fit(dl_train=dl_train, dl_val=dl_val,dl_test=dl_test, num_epochs=epochs, checkpoints=checkpoints,
                          early_stopping=early_stopping, print_every=1, **{'max_batches': batches})


    # save_experiment(run_name, out_dir, cfg, fit_res)
    return fit_res


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
