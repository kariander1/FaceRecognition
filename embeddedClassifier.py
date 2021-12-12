import csv
import os.path

import DeepLearning.experiments
import pklDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from prettytable import PrettyTable

# Define NET class with RELU activation
class Net(nn.Module):
    def __init__(self, dropout_rate, filter_size, pooling_size, num_of_filters, num_of_filters_2, num_of_filters_3,
                 num_of_filters_4):
        super().__init__()
        self.conv_layer = nn.Sequential(

            nn.Conv2d(in_channels=3, out_channels=num_of_filters, kernel_size=filter_size, stride=1,
                      padding='same'),
            # TODO: Try batch after activations
            nn.BatchNorm2d(num_of_filters),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(pooling_size, pooling_size),
            # nn.Dropout2d(dropout_rate),
            nn.Conv2d(num_of_filters, num_of_filters_2, filter_size, padding='same'),
            nn.BatchNorm2d(num_of_filters_2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(pooling_size, pooling_size),
            nn.Conv2d(num_of_filters_2, num_of_filters_3, filter_size, padding='same'),
            nn.BatchNorm2d(num_of_filters_3),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(pooling_size, pooling_size),
            nn.Conv2d(num_of_filters_3, num_of_filters_4, filter_size, padding='same'),
            nn.BatchNorm2d(num_of_filters_4),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(pooling_size, pooling_size)
        )
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_of_filters_4 * 2 * 2, 10)
            # self.fc2 = nn.Linear(120, 10)
            # self.fc3 = nn.Linear(84, 10)
        )

    def forward(self, x):
        """Perform forward."""

        # conv layers
        x = self.conv_layer(x)

        # flatten
        x = x.view(x.size(0), -1)

        # fc layer
        x = self.fc_layer(x)

        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        return x


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


def calc_test_loss(net, test_loader, criterion):
    running_test_loss = 0
    num_of_test_images = 0
    for test_data in test_loader:
        num_of_test_images += 1
        test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
        # calculate outputs by running images through the network
        test_outputs = net(test_images)
        test_loss = criterion(test_outputs, test_labels)
        running_test_loss += test_loss.item()

    return running_test_loss / num_of_test_images


# Trains the given train loader and saves the trained net in the path given
def train_net(net, train_loader, test_loader, optimizer, scheduler, criterion, save_path, passes=2,
              status_every_batch=2000):
    train_losses = []
    iterations = []
    test_accuracies = []
    test_losses = []
    learning_rates = []
    iteration = 1
    loss_figure = None
    max_acc = 0
    #sched_cnt = 0
    for epoch in range(passes):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % status_every_batch == status_every_batch - 1:  # print every status_every_batch mini-batches

                # Calculate AVG train loss across the batch
                train_loss = running_loss / status_every_batch
                running_loss = 0.0
                # Run test data and calculate loss
                test_loss = calc_test_loss(net=net, test_loader=test_loader, criterion=criterion)
                test_acc = calc_accuracy(test_loader, net)
                learning_rate = optimizer.param_groups[0]['lr']

                if test_acc > max_acc:
                    # Save current NET
                    save_path_new = save_path.replace('.pth', str(test_acc) + '.pth')
                    torch.save(net.state_dict(), save_path_new)
                    max_acc = test_acc

                iterations.append(iteration)
                train_losses.append(train_loss)
                test_losses.append(test_loss)
                learning_rates.append(learning_rate)
                test_accuracies.append(test_acc)
                print('[%d, %5d] train loss: %.3f test loss: %.3f test accuracy: %.3f %% LR: %f' %
                      (epoch + 1, i + 1, train_loss, test_loss, test_acc,learning_rate))
                loss_figure = show_loss_graph(train_loss_vector=train_losses,
                                              test_loss_vector=test_losses,
                                              test_accuracy_vector=test_accuracies,
                                              iterations_vector=iterations,
                                              figure=loss_figure)
                scheduler.step(test_loss)

            iteration = iteration + 1
        #scheduler.step()
        #sched_cnt += 1
        #print("sched step " + str(sched_cnt) + " - lr is " + str(optimizer.param_groups[0]['lr']))


    print('Finished Training')
    torch.save(net.state_dict(), save_path)

    data_path = save_path.replace('.pth', '.csv')
    print('Exporting train data to ' + data_path)

    # Export data to csv
    fields = ['Iterations', 'Train Loss', 'Test Loss', 'Learning Rate', 'Test Accuracy']
    with open(data_path, 'w',newline='') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        write.writerow(fields)
        for i in range(len(iterations)):
            row_str = [iterations[i], train_losses[i], test_losses[i], learning_rates[i], test_accuracies[i]]
            write.writerow(row_str)

    return [train_losses[-1], max_acc]


def calc_accuracy(test_loader, net, calc_per_class=False, print_acc=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    if print_acc is True:
        print('Accuracy of the network on the 10000 test images: %.2f %%' % (
            accuracy))

    test_accuracy = accuracy
    if calc_per_class:
        # prepare to count predictions for each class
        correct_pred = {classname: 0 for classname in classes}
        total_pred = {classname: 0 for classname in classes}

        # again no gradients needed
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = net(images)
                _, predictions = torch.max(outputs, 1)
                # collect the correct predictions for each class
                for label, prediction in zip(labels, predictions):
                    if label == prediction:
                        correct_pred[classes[label]] += 1
                    total_pred[classes[label]] += 1

        # print accuracy for each class
        for classname, correct_count in correct_pred.items():
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                                 accuracy))
    return test_accuracy


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
batch_size = 64
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
pkl_path = 'Insight/insightface/recognition/arcface_torch/r50_features.pkl'
# Download CIFAR10 data set into ./data directory
# Datasets
pkl_dataset = pklDataset.PklDataset(pkl_path, infer_classes_and_n_records=True)
# train_set = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=train_transform)

# TODO enable also shuffle
# pkl_loader = torch.utils.data.DataLoader(pkl_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

loss_fn = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
fit_res = DeepLearning.experiments.cnn_experiment(run_name="rs50_features", ds_train=pkl_dataset, ds_test=pkl_dataset,
                                                  bs_train=128, bs_test=128,
                                                  batches=1000, epochs=100, early_stopping=20,
                                                  filters_per_layer=[64, 128, 256],
                                                  layers_per_block=0, pool_every=4, hidden_dims=[32, 64, 32, 32],
                                                  lr=0.001, loss_fn=loss_fn,
                                                  model_type="cnn")

print("Finished")