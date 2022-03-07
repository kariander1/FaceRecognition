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

import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable
def show_loss_graph(train_loss_vector, test_loss_vector, test_accuracy_vector, iterations_vector, figure=None):
    if figure is None:
        figure = plt.figure()
        plot = figure.add_subplot(1, 1, 1)
        sec_plot = plot.twinx()

    else:
        plot = figure.axes[0]
        sec_plot = figure.axes[1]
        sec_plot.clear()
        plot.clear()

    plot.plot(iterations_vector, train_loss_vector, label="Training Loss")
    plot.plot(iterations_vector, test_loss_vector, label="Test Loss")
    sec_plot.plot(iterations_vector, test_accuracy_vector, label="Test Accuracy", color="purple")
    # naming the x axis

    plot.set_xlabel('Iteration [#]')
    # naming the y axis
    plot.set_ylabel('Loss')
    sec_plot.set_ylabel('Accuracy [%]')
    # giving a title to my graph
    plot.set_title('Train Loss Graph')

    plt.close(figure)
    # function to show the plot
    figure.show()

    # show a legend on the plot
    figure.legend()

    return figure


# function to show an image
def im_show(img):
    img = img / 2 + 0.5  # denormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


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
train_set, val_set, test_set = pklDataset.SplitDataset(pkl_dataset,n_labels=40000 ,val_ratio=0.1, test_ratio=0.1)
#pklDataset.InterpolateDatasetRandom(train_set,pkl_dataset)

features_loss_fns = [torch.nn.CosineSimilarity(), torch.nn.MSELoss()]
features_loss_weights = [1,10]
label_loss_fns = []
label_loss_weights = []

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)


msresnet = DeepLearning.MSResnet.MSResNet(input_channel=1, layers=[1, 1, 1, 1],num_classes=len(pkl_dataset.ds1.classes),embedding_size=512)

fit_res = DeepLearning.experiments.cnn_experiment(model=msresnet, run_name="rs50_features", ds_train=train_set,
                                                  ds_test=val_set,
                                                  bs_train=batch_size, bs_test=batch_size, optimizer=None,
                                                  epochs=200, early_stopping=5,
                                                  filters_per_layer=[64, 128, 512],
                                                  layers_per_block=0, pool_every=4, hidden_dims=[],
                                                  lr=0.001, features_loss_fns=features_loss_fns,
                                                  features_loss_weights=features_loss_weights,
                                                  label_loss_fns=label_loss_fns,
                                                  label_loss_weights=label_loss_weights,
                                                  model_type="cnn",batches=None)

print("Finished")