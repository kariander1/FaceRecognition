r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers


def part1_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers = 3
    hidden_dims = 32
    activation = 'relu'
    out_activation = 'relu'
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part1_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, momentum, weight_decay = 0.03, 0.0, 0.0
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part1_q1 = r"""
**Your answer:**

1. The optimization error is incurred by not finding the optimal solution of the in-sample loss, meaning that we find a
minimizer which is an approximation to the optimal one. The error is usually decreased when we allocate more time and 
computation resources to find the solution, however the given datasets are synthetically generated and are limited to
$N=10,000$ and we are able to cycle through all samples, therefore we would say that the optimization error is
negligible and also by looking at the loss graphs we can infer that we have reached a plateau and thus converged.
Hence using more epochs might not improve the error, and even cause over-fitting.


2. The generalization  error is incurred by defining our optimization problem on an empirical in-sample loss and not
 based on the actual distribution. We think the error in the first binary classifier may not be negligible and can be
 improved by generating a larger number of samples.
 
 
3. The approximation error is incurred by limiting our estimators to a specific group of models, in this case the MLP 
model, which might differ from the actual optimal estimator which might be outside this group.
We believe this model is quite effective for this specific distribution, even though some samples overlap in both
decision boundaries, the binary classification boundary that is fits this problem quite effectively.

"""

part1_q2 = r"""
**Your answer:**

Looking at the generation process of the validation-set, we can see that the validation set was rotated and had bigger
variance in its distribution (as seen by the noise argument). Therefore if our model was trained on a less noisy dataset
it would make sense that the false negative/positive ratios be higher since there will be more overlaps between 
positive and negative samples, as can be seen in the decision boundary plot where the decision boundary mimics the angle 
of the training set and applied to the different angle (and noise) of the validation set. 

"""

part1_q3 = r"""
**Your answer:**

1. According to the data given, a cases of false-positive is most desirable to minimize, since they are both costly and
life-endangering. 
Furthermore knowing that both low-risking symptoms occur if not diagnosed early and a low-cost and low-risk treatment 
exists, we would prefer to always classify patients as NEGATIVE to reduce cost and risk to the patients.  
Therefore we will give a much bigger weight to reducing FPR, to the extent of classifying all patients as NEGATIVE.


2. 
Case 1: Assuming cost-efficiency and life risk minimizing are equally important, there is no value in sending even
the FP patients to further testings since their life will be risked either way, and the further tests are expensive,
so we choose to not send them at all and save expenses. This way we reach an all NEGATIVE classifier.

Case 2: Assuming minimizing risk is of greater importance than expenses, and the further diagnosing test is safer than
the disease itself, we would like both FP and FN as much as possible. However the risk of FN is greater than FP to the
patient and therefore we will give slightly more weigh to minimize FNR when deciding the threshold.


"""


part1_q4 = r"""
**Your answer:**

1. Given a constant depth, the width will determine the amount of features per layer which in turn provide more parameters
to each layer. These parameters provide greater robustness to the model and better fits the training samples and distibution.

2. Given a constant width, increasing the depth of the model will increase the amount of non-linearity applied, and might
make the classification boundaries more sophisticated, as seen in the comparison - when the depth is low, the boundary is
tends to be close to a line or a single $tanh$ function (which is the activation function we chose), and when increasing
the depth we can see a more complex boundary.

3. A) We can easily identify that the $depth=1, width=32$ lacks the non-linearity that is necessary to draw a more
complex contour while the $depth=4, width=8$ has enough non-linearity to provide a complex contour, and also enough
perceptrons to each layer.   


B) Similar to A), although $depth=1, width=128$ has improved, it is still under performs opposed to $depth=4, width=32$.
We can see that increasing the width does improve the performance, however what the model actually lacks is a deeper net
for more non-linearity that makes the contour more complex.

4. As we can see immediately from "Threshold Selection" section the model improved when evaluated again on the validation
set.
Regarding the test set, since the test dataset distribution was generated with a closer angle and closer variance to the
validation set, we can conclude that changing the threshold to fit the validation set provided an improved classification
of the test-set as opposed of remaining at a threshold of 0.5 amd evaluated only at the train set.


"""
# ==============
# Part 2 answers


def part2_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 0.01
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part2_q1 = r"""
**Your answer:**

1.


The number of parameters for a certain conv layer can be calculated as follows:

$ params = ((width \times height \times in channels)+1)\times out channels$

In the case of residual block, we shall multiply the product by 2 to account for the two layers.
Therefore we can calculate the number of parameters:


$$
Residual Blocks:
((3 \times 3 \times 256)+1)\times 256 \times 2 = 1180160
$$

For the bottleneck, we can divide the net to 3 sub modules, one is the first input convolution layer with 1X1 conv,
the second is the actual 3X3 convolutional layers and the third is the output layer with 1X1 conv:

$$
Residual Bottle Necks:
(256 \times 1 + 1)\times 64+ (64 \times 3^{2} + 1)\times 64 + (64 \times 1 + 1)\times 256 = 70016
$$

2.

Taking the previous expressions for the number of parameters multiplied by the input width will yield a qualitative number
of floating point operations:

$Number of multiplications = C_in \times K^2$

$Number of Additions = C_in \times K^2-1$

$Number of bias  = 1$

Therefore for each layer we get the following approximation for operations:
$ 2C_{in} \times K^2 +1 $

Accounting for dimensions of the image we get:
$ operations = 2\times(I_{width}\times I_{height})\times(C_{in} \times K^2 +1)\times out channels$

Therefore the number of operations in each case:
$$
Residual Blocks Operations: \approx
2\times (I_{width}\times I_{height})\times((3 \times 3 \times 256)+1)\times 256 \times 2 \approx (I_{width}\times I_{height})\times 2M
$$


$$
Residual Bottle Neck Operations: \approx
2\times(I_{width}\times I_{height})\times[(256 \times 1 + 1)\times 64+ (64 \times 3^{2} + 1)\times 64 + (64 \times 1 + 1)\times 
256] \approx (I_{width}\times I_{height})\times 150K
$$ 

As we can see the bottle neck has less parameters by some significant order of magnitude for this small example.
When scaled to larger net, the differences get larger even more.


3.

On the classic block we use 3X3 convolution across layers and thus leverage the spatial locality between neighbors in
the feature maps. Moreover, we have two of these convolution layers, and therefore we also leverage a stronger relation across
feature maps from different layers in the classic block.
In the bottleneck we perform only one 3X3 conv layer which leverage only once the spatial locality, and afterwards we only
perform 1X1 convs and therefore not leveraging that much the cross relation of feature maps from different layers.
"""

# ==============

# ==============
# Part 3 answers


part3_q1 = r"""
**Your answer:**

1.
For K=32 we can see that the results do not differ that much on the test, and the accuracies converge into about 65%
However for K=64, we can see that L=4 produces the best results which is probably the best depth in this case.
Other L values perform worse.
We can infer that there is a certain depth which performs which performs best with a given dimensions, and when taking
L values too low or too high we lose performance.


2. We can see that for L=16 the network performed worse that other L values, and moreover it wasn't trainable as we can see
that it didn't perform better than 10%.
That is because we are facing the problem of vanishing gradients. There are some ways to cope with it:

a) Use skip connections like in the residual networks

b) Use batch normalizations

"""

part3_q2 = r"""
**Your answer:**

In this experiment we reached better results than before, we can infer that this is due to the fact that the
K parameter affects more the performance than the L parameter. We can see that increasing the channels does make the results
better.
In particular, using a deeper network (L=8) gave the best results with a large amount of conv (K=128).
The results are all in all better than before, and none were suffering from vanishing gradients. they were all trainable. 


"""

part3_q3 = r"""
**Your answer:**

First of all we can see that for L=1 we have overfitted as seen in the test_loss graph.
We did not get better results than 1.2, and the best performance was achieved with L=2 and L=3 with favor to L=2
Which converged faster as well.

"""

part3_q4 = r"""
**Your answer:**


The results all in all do not perform better than previous experiments, however we can notice that resnets converge
better with deeper networks (Large L) and this allows usage of deeper networks since we do not face vanishing gradients
like we did in CNN.

Comparing to 1.1 as we said - in 1.4 all models were trainable and reached a better performance than 1.1.
Comparing to 1.3 the same could be said - we were able to train a deeper network, and also reach a bit more performance
(but not more than 1.2).

"""

part3_q5 = r"""
**Your answer:**

1. We Chose to implement the inception module as inspired by GoogleLeNet architecture.
We chose to implement a sub class of a inception block (like in the resnet block case)
which consists of 4 branches:

Narrow path - a path consisting of a 1X1 Conv channel decreaser followed by a 3X3 Conv layers and activation.
The channels of the convs are defined by K.

Wide path -  Same as the narrow, with the kernel size of the convolutions being a 5X5.

Shortcut path - similar to the shortcut path in the resnet block which will connect the input to the output
and will have a 1X1 conv channel normalizer if needed.

Pooling path - a path consisting of Max Pooling with stride of 1 to keep dimensions same (padding also =1) followed
by a 1X1 conv to match the channels with the other channels.

On the forward pass we sum all paths and apply a RELU similarly to resnet.


2. The results of some of our models perform reach an accuracy of 80% which 10% better than previous
models in the experiments.
The network can converge with deep networks as seen, and performed best with deep networks (L=12).

"""
# ==============
