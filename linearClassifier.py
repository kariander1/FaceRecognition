import csv
import os.path

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
    def __init__(self, filter_size, pooling_size, num_of_filters, num_of_filters_2):
        super().__init__()
        # TODO: Define layers for better accuracy (depth)
        # TODO: Define filter size
        # TODO: Check how to define/random weights if needed
        self.conv1 = nn.Conv2d(3, num_of_filters, filter_size, padding='same')
        self.pool = nn.MaxPool2d(pooling_size, pooling_size)
        self.conv2 = nn.Conv2d(num_of_filters, num_of_filters_2, filter_size, padding='same')
        self.conv3 = nn.Conv2d(num_of_filters_2, 25, filter_size)
        #self.conv4 = nn.Conv2d(8, 8, filter_size, padding='same')
        #self.identity = nn.Identity()
        #self.fc1 = nn.Linear(100, 84)
        self.fc2 = nn.Linear(100, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        #x = self.identity(x) + F.relu(self.conv4(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        #x = self.fc1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show_loss_graph(train_loss_vector, test_loss_vector,test_accuracy_vector, iterations_vector, figure=None):
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

    # show a legend on the plot
    figure.legend()

    # function to show the plot
    figure.show()


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
        test_images, test_labels = test_data
        # calculate outputs by running images through the network
        test_outputs = net(test_images)
        test_loss = criterion(test_outputs, test_labels)
        running_test_loss += test_loss.item()

    return running_test_loss / num_of_test_images


# Trains the given train loader and saves the trained net in the path given
def train_net(net, train_loader, test_loader, optimizer,criterion, save_path, passes=2, status_every_batch=4000):
    train_losses = []
    iterations = []
    test_accuracies = []
    test_losses = []
    iteration = 1
    loss_figure = None
    for epoch in range(passes):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)


            # print statistics
            running_loss += loss.item()

            if i % status_every_batch == status_every_batch - 1:  # print every 2000 mini-batches
                train_loss = running_loss / status_every_batch
                train_losses.append(train_loss)
                running_loss = 0.0
                # Run test data and calculate loss

                test_loss = calc_test_loss(net=net, test_loader=test_loader, criterion=criterion)
                scheduler.step(test_loss)
                test_losses.append(test_loss)

                test_acc = calc_accuracy(test_loader, net)
                test_accuracies.append(test_acc)

                iterations.append(iteration)
                print('[%d, %5d] train loss: %.3f test loss: %.3f test accuracy: %d %%' %
                      (epoch + 1, i + 1, train_loss, test_loss, test_acc))
                # loss_figure = show_loss_graph(train_loss_vector=train_losses,
                #                               test_loss_vector=test_losses,
                #                               test_accuracy_vector=test_accuracies,
               #                               iterations_vector=iterations,
                #                               figure=loss_figure)

            loss.backward()
            optimizer.step()
            iteration = iteration + 1

    print('Finished Training')
    torch.save(net.state_dict(), save_path)

    loss_figure = show_loss_graph(train_loss_vector=train_losses,
                                  test_loss_vector=test_losses,
                                  test_accuracy_vector=test_accuracies,
                                  iterations_vector=iterations,
                                  figure=loss_figure)

    # Return loss
    return train_losses[-1]


def calc_accuracy(test_loader, net, calc_per_class=False):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100.0 * correct / total
    #print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))

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

    #print('Accuracy of the network on the 10000 test images: %d %%' % accuracy)
    return accuracy


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
# for i in range(1, 2):

#************************
#************************
#************************

fields = {}
csv_name = r'accuracy_chart.csv'
batch_size = 8
filter_size = 5
epochs = 20
num_of_filters = 35  # 29
num_of_filters_2 = 50  # 16
learning_rate = 0.001
weight_decay = 0.001
pooling_size = 2
momentum = 0.9
num_of_fc = 3
test_name = 'test21'

#******************************
#************************
#************************

fields['Batch Size'] = batch_size
fields['Filter Size'] = filter_size
fields['Filter Count'] = num_of_filters
fields['Filter Count 2'] = num_of_filters_2
fields['Epochs'] = epochs
fields['Learning Rate'] = learning_rate
fields['Weight Decay'] = 0
fields['Pooling Size'] = pooling_size
fields['Number of FC'] = num_of_fc

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
train_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     transforms.RandomHorizontalFlip(0.5),
     transforms.ColorJitter(brightness=.5, hue=.3),
     transforms.RandomApply(transforms=[transforms.RandomResizedCrop(size=(32, 32))], p=0.5)
     ])

test_transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

# Define directory to save dataset images
data_directory = './data'
# Download CIFAR10 data set into ./data directory
# Dataset
train_set = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Changed num_workers to 0 since running on windows.
# Test set
test_set = torchvision.datasets.CIFAR10(root=data_directory, train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
data_iter = iter(train_loader)
images, labels = data_iter.next()

# show images
#im_show(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# Try selecting GPU with CUDA processors:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Create new Net
net = Net(filter_size=filter_size, pooling_size=pooling_size, num_of_filters=num_of_filters,
          num_of_filters_2=num_of_filters_2)
# writer = SummaryWriter('logs/')
# writer.add_graph(net, images)
# writer.close()
net.to(device)
# TODO : Test different losses
criterion = nn.CrossEntropyLoss()
criterion_str = type(criterion).__name__
fields["Loss Function"] = criterion_str
# TODO : Change optimizer (maybe to ADAM)
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=1)

optimizer_str = type(optimizer).__name__
fields["Optimizer"] = optimizer_str
net_path = 'cifar_net_' + str(batch_size) + '_' + str(filter_size) + '_' + str(epochs) + '_' + str(
    learning_rate) + '_' + str(pooling_size) + '_' + str(
    num_of_fc) + '_' + optimizer_str + '_' + str(num_of_filters) + '_' + str(
    num_of_filters_2) + '_' + test_name + '_' + str(
    weight_decay) + '_' + criterion_str + '.pth'

fields['Train Loss'] = 'Not Trained'
if not os.path.isfile(net_path):
    # If the net doesn't exist
    print("Network wasn't found, training a new network:")
    train_loss = train_net(net=net, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                           criterion=criterion,
                           save_path=net_path, passes=epochs)
    fields['Train Loss'] = train_loss

# Take the first batch
data_iter = iter(test_loader)
images, labels = data_iter.next()

# print images and labels given to the batch
#im_show(torchvision.utils.make_grid(images))

# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Load the net that was saved
# net = Net()
net.load_state_dict(torch.load(net_path))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

# print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
#                              for j in range(4)))

# Calculate accuracy on test images
test_accuracy = calc_accuracy(test_loader, net)
fields["Test Accuracy"] = test_accuracy
# Count how many parameters we used foreach layer
param_count = count_parameters(net)

fields["Parameters"] = param_count
fields['Comment'] = test_name
with open(csv_name, 'a', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(fields.values())

