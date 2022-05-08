import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}


# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
        self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)


        super().__init__()

        layers = []
        if len(nonlins) == 0:
            self.fc_layers = nn.Sequential(*layers)
            return
        self.in_dim = in_dim
        self.out_dim = dims[-1]
        mod_dims = [in_dim] + dims
        for i_layer in range(0, len(dims)):

            layers.append(nn.Linear(in_features=mod_dims[i_layer], out_features=mod_dims[i_layer + 1]))

            nonlin = nonlins[i_layer]
            if isinstance(nonlin, str):
                # Non linearity is a defined dictionary activation
                nonlin = ACTIVATIONS[nonlin](**ACTIVATION_DEFAULT_KWARGS[nonlin])
            layers.append(nonlin)
        self.fc_layers = nn.Sequential(*layers)

        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """


        # Flatten to 2D tensors
        x = torch.reshape(x, (x.shape[0], -1))
        y_pred = self.fc_layers(x)

        return  y_pred

