import csv
import os.path
import DeepLearning.MSResnet
import DeepLearning.experiments
import pklDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import argparse

import torchvision
import torchvision.transforms as transforms

from Insight.insightface.recognition.arcface_torch.backbones import get_model
from Insight.insightface.recognition.arcface_torch.dataset import MXFaceDataset, SyntheticDataset, DataLoaderX
from Insight.insightface.recognition.arcface_torch.partial_fc import PartialFC
from Insight.insightface.recognition.arcface_torch.utils.utils_config import get_config
from Insight.insightface.recognition.arcface_torch.utils.utils_logging import AverageMeter, init_logging
import logging

from prettytable import PrettyTable


# Method for calculating how many parameters were used in the network
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main(args):
    # Define batch size for stochastic gradient descent

    fields = {}
    csv_name = r'accuracy_chart.csv'
    batch_size = 256
    filter_size = 3
    epochs = 60
    dropout_rate = 0.05
    num_of_filters = 32
    num_of_filters_2 = 64
    num_of_filters_3 = 32
    num_of_filters_4 = 32
    learning_rate = 0.001
    weight_decay = 0
    pooling_size = 2
    momentum = 0.9
    num_of_fc = 1
    test_name = '3X3 filter'

    fields['Batch Size'] = batch_size
    fields['Dropout Rate'] = dropout_rate
    fields['Filter Size'] = filter_size
    fields['Filter Count'] = num_of_filters
    fields['Filter Count 2'] = num_of_filters_2
    fields['Filter Count 3'] = num_of_filters_3
    fields['Filter Count 4'] = num_of_filters_4
    fields['Epochs'] = epochs
    fields['Learning Rate'] = learning_rate
    fields['Weight Decay'] = 0
    fields['Pooling Size'] = pooling_size
    fields['Number of FC'] = num_of_fc

    # Define directory to save dataset images
    data_directory = './data'
    pkl_path1 = 'Insight/insightface/recognition/arcface_torch/r18_features'
    pkl_path2 = 'Insight/insightface/recognition/arcface_torch/r50_features'

    # Datasets

    pkl_dataset = pklDataset.PklEmbeddingsDataset(pkl_path1,pkl_path2)

    [train_set, val_set, test_set] = pklDataset.SplitDataset(pkl_dataset, n_labels=40000, ratios = [0.006 , 0.006])
    #train_nn_space = pklDataset.CreateMeanLabels(train_set, 'train_not_interp')
    val_nn_space = pklDataset.CreateNNLabels(val_set, 'val_new_split')
    test_nn_space = pklDataset.CreateNNLabels(test_set, 'test_new_split')
    #pklDataset.InterpolateDatasetRandom(train_set,pkl_dataset)
    #train_nn_space = pklDataset.CreateMeanLabels(train_set, 'train')

    features_loss_fns = [torch.nn.CosineSimilarity(), torch.nn.MSELoss()]
    features_loss_weights = [1,300]
    label_loss_fns = []
    label_loss_weights = []

    msresnet = DeepLearning.MSResnet.MSResNet(input_channel=1, layers=[1, 1, 1, 1],num_classes=len(pkl_dataset.ds1.classes),embedding_size=512)
    #
    # # Backbone configs
    # cfg = get_config(args.config)
    # try:
    #     world_size = int(os.environ['WORLD_SIZE'])
    #     rank = int(os.environ['RANK'])
    #     dist.init_process_group('nccl')
    # except KeyError:
    #     world_size = 1
    #     rank = 0
    #     dist.init_process_group(backend='nccl', init_method="tcp://127.0.0.1:12584", rank=rank, world_size=world_size)
    #
    # local_rank = args.local_rank
    # torch.cuda.set_device(local_rank)
    # os.makedirs(cfg.output, exist_ok=True)
    # init_logging(rank, cfg.output)
    #
    # backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).to(local_rank)
    #
    # if cfg.resume:
    #     try:
    #         backbone_pth = os.path.join(cfg.output, "backbone.pth")
    #         backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(local_rank)))
    #         if rank == 0:
    #             logging.info("backbone resume successfully!")
    #     except (FileNotFoundError, KeyError, IndexError, RuntimeError):
    #         if rank == 0:
    #             logging.info("resume fail, backbone init successfully!")
    #
    # backbone = torch.nn.parallel.DistributedDataParallel(
    #     module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    # margin_softmax = losses.get_loss(cfg.loss)
    # module_partial_fc = PartialFC(
    #     rank=rank, local_rank=local_rank, world_size=world_size, resume=cfg.resume,
    #     batch_size=cfg.batch_size, margin_softmax=margin_softmax, num_classes=cfg.num_classes,
    #     sample_rate=cfg.sample_rate, embedding_size=cfg.embedding_size, prefix=cfg.output)



    fit_res_msresnet = DeepLearning.experiments.cnn_experiment(model=msresnet, run_name="rs50_features", ds_train=train_set,
                                                               ds_test=val_set, ds_test_for_realzis=test_set,
                                                               bs_train=batch_size, bs_test=batch_size, optimizer=None,
                                                               epochs=200, early_stopping=10,
                                                               filters_per_layer=[64, 128, 512],
                                                               layers_per_block=0, pool_every=4, hidden_dims=[],
                                                               lr=0.001, features_loss_fns=features_loss_fns,
                                                               features_loss_weights=features_loss_weights,
                                                               label_loss_fns=label_loss_fns,
                                                               label_loss_weights=label_loss_weights,
                                                               model_type="cnn", batches=None,
                                                               train_nn_space=val_nn_space,
                                                               val_nn_space=val_nn_space,
                                                               test_nn_space=test_nn_space
                                                               )

    print("Training MLP")
    fit_res_mlp = DeepLearning.experiments.cnn_experiment(model=None, run_name="rs50_features", ds_train=train_set,
                                                          ds_test=val_set, ds_test_for_realzis=test_set,
                                                          bs_train=batch_size, bs_test=batch_size, optimizer=None,
                                                          epochs=200, early_stopping=10,
                                                          filters_per_layer=[64, 128, 512],
                                                          layers_per_block=0, pool_every=4, hidden_dims=[],
                                                          lr=0.001, features_loss_fns=features_loss_fns,
                                                          features_loss_weights=features_loss_weights,
                                                          label_loss_fns=label_loss_fns,
                                                          label_loss_weights=label_loss_weights,
                                                          model_type="cnn", batches=None,
                                                          train_nn_space=val_nn_space,
                                                          val_nn_space=val_nn_space,
                                                          test_nn_space=test_nn_space
                                                          )

    print("Finished")


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('config', type=str, help='py config file')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    main(parser.parse_args())
