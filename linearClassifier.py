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
#from torch.utils.tensorboard import SummaryWriter



# Define NET class with RELU activation
class Net(nn.Module):
    def __init__(self, filter_size, pooling_size):
        super().__init__()
        # TODO: Define layers for better accuracy (depth)
        # TODO: Define filter size
        # TODO: Check how to define/random weights if needed
        self.conv1 = nn.Conv2d(3, 6, filter_size)
        self.conv2 = nn.Conv2d(6, 10, filter_size)
        self.pool1 = nn.MaxPool2d(pooling_size, pooling_size)
        self.pool2 = nn.MaxPool2d(pooling_size, pooling_size)
        self.conv3 = nn.Conv2d(10, 14, filter_size)
        self.fc1 = nn.Linear(14 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show_loss_graph(train_loss_vector, test_accuracy_vector, iterations_vector):
    # plotting the line 1 points
    plt.plot(iterations_vector, train_loss_vector, label="Training Loss")

    # naming the x axis

    plt.xlabel('Iteration [#]')
    # naming the y axis
    plt.ylabel('Loss')
    # giving a title to my graph
    plt.title('Train Loss Graph')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()

    # plotting the line 2 points
    plt.plot(iterations_vector, test_accuracy_vector, label="Test Accuracy")

    # naming the x axis
    plt.xlabel('Iteration [#]')
    # naming the y axis
    plt.ylabel('Accuracy [%]')
    # giving a title to my graph
    plt.title('Test Accuracy Graph')

    # show a legend on the plot
    plt.legend()

    # function to show the plot
    plt.show()
# function to show an image
def im_show(img):
    img = img / 2 + 0.5  # unnormalize
    np_img = img.numpy()
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.show()


# Trains the given train loader and saves the trained net in the path given
def train_net(net, train_loader, test_loader, optimizer, save_path, passes=2, status_every_batch=2000):
    train_losses = []
    iterations = []
    test_accuracies = []
    iteration = 1
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
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            if i % status_every_batch == status_every_batch - 1:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / status_every_batch))
                train_losses.append(running_loss / status_every_batch)

                running_loss = 0.0
                # Run test data and calculate loss
                test_acc = calc_accuracy(test_loader, net)
                test_accuracies.append(test_acc)
                iterations.append(iteration)

            iteration = iteration + 1

    print('Finished Training')
    torch.save(net.state_dict(), save_path)
    # Plot the testing loss graph
    # plt.figure(figsize=(10, 5))
    # plt.title("Training and Validation Loss")
    # plt.plot(train_losses, label="train")
    # plt.xlabel("iterations")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()
    show_loss_graph(train_losses, test_accuracies, iterations)
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
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        accuracy))

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
fields = {}
csv_name = r'accuracy_chart.csv'
batch_size = 4
filter_size = 5
epochs = 2
learning_rate = 0.0007
weight_decay = 0
pooling_size = 2
momentum = 0.9
num_of_fc = 2
test_name = 'Different pools'

fields['Batch Size'] = batch_size
fields['Filter Size'] = filter_size
fields['Epochs'] = epochs
fields['Learning Rate'] = learning_rate
fields['Weight Decay'] = 0
fields['Pooling Size'] = pooling_size
fields['Number of FC'] = num_of_fc
fields['Comment'] = test_name

# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define directory to save dataset images
data_directory = './data'
# Download CIFAR10 data set into ./data directory
# Dataset
train_set = torchvision.datasets.CIFAR10(root=data_directory, train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Changed num_workers to 0 since running on windows.
# Test set
test_set = torchvision.datasets.CIFAR10(root=data_directory, train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
data_iter = iter(train_loader)
images, labels = data_iter.next()

# show images
im_show(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# Try selecting GPU with CUDA processors:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

# Create new Net
net = Net(filter_size, pooling_size)
#writer = SummaryWriter('logs/')
#writer.add_graph(net, images)
#writer.close()
net.to(device)
# TODO : Test different losses
criterion = nn.CrossEntropyLoss()
criterion_str = type(criterion).__name__
fields["Loss Function"] = criterion_str
# TODO : Change optimizer (maybe to ADAM)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
optimizer_str = type(optimizer).__name__
fields["Optimizer"] = optimizer_str
net_path = 'cifar_net_' + str(batch_size) + '_' + str(filter_size) + '_' + str(epochs) + '_' + str(learning_rate) + '_' \
           + str(pooling_size) + '_' + str(num_of_fc) + '_' + optimizer_str + '_' + test_name + '_' + str(
    weight_decay) + '_' + criterion_str + '.pth'

fields['Train Accuracy'] = 'Not Trained'
if not os.path.isfile(net_path):
    # If the net doesn't exist
    print("Network wasn't found, training a new network:")
    train_accuracy = 1 - train_net(net=net, train_loader=train_loader, test_loader=test_loader, optimizer=optimizer,
                                   save_path=net_path, passes=epochs)
    fields['Train Accuracy'] = train_accuracy

# Take the first batch
data_iter = iter(test_loader)
images, labels = data_iter.next()

# print images and labels given to the batch
im_show(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Load the net that was saved
# net = Net()
net.load_state_dict(torch.load(net_path))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# Calculate accuracy on test images
test_accuracy = calc_accuracy(test_loader, net)
fields["Test Accuracy"] = test_accuracy
# Count how many parameters we used foreach layer
param_count = count_parameters(net)

fields["Parameters"] = param_count
with open(csv_name, 'a', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(fields.values())
