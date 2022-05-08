import torch
import torch.nn as nn
import itertools as it
from torch import Tensor
from typing import Sequence

from .mlp import MLP, ACTIVATIONS, ACTIVATION_DEFAULT_KWARGS

POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class CNN(nn.Module):
    """
    A simple convolutional neural network model based on PyTorch nn.Modules.

    Has a convolutional part at the beginning and an MLP at the end.
    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: Sequence[int],
            pool_every: int,
            hidden_dims: Sequence[int],
            conv_params: dict = {'kernel_size': 3, 'stride': 1, 'padding': 1},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {'kernel_size': 2},
            is_identity: bool = False,
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
       # assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        output_size = in_size[0]
        if is_identity:
            output_size = None
        self.features_fc = self._make_mlp(output_size)
        self.label_classifier = self._make_mlp(self.out_classes,in_dim=in_size[0])

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []

        act = ACTIVATIONS[self.activation_type](**self.activation_params)

        pool = POOLINGS[self.pooling_type](**self.pooling_params)
        mod_channels = [in_channels] + self.channels
        for i_channel in range(1, len(mod_channels)):
            layers += [nn.Conv2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                 **self.conv_params), act]
            if i_channel % self.pool_every == 0:
                layers += [pool]

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        try:

            in_channels, in_h, in_w, = tuple(self.in_size)
            # Conv params
            conv_kernel_size = self.conv_params['kernel_size']
            conv_kernel_stride = 1 if 'stride' not in self.conv_params.keys() else self.conv_params['stride']
            conv_kernel_padding = 0 if 'padding' not in self.conv_params.keys() else self.conv_params['padding']

            # Pool params
            pool_kernel_size = self.pooling_params['kernel_size']
            pool_kernel_stride = pool_kernel_size if 'stride' not in self.pooling_params.keys() else self.pooling_params['stride']
            pool_kernel_padding = 0 if 'padding' not in self.pooling_params.keys() else self.pooling_params['padding']

            mod_channels = [in_channels] + self.channels

            for i_channel in range(1, len(mod_channels)):
                in_h = ((in_h - conv_kernel_size + 2 * conv_kernel_padding) // conv_kernel_stride) + 1
                in_w = ((in_w - conv_kernel_size + 2 * conv_kernel_padding) // conv_kernel_stride) + 1

                if i_channel % self.pool_every == 0:
                    in_h = ((in_h - pool_kernel_size + 2 * pool_kernel_padding) // pool_kernel_stride) + 1
                    in_w = ((in_w - pool_kernel_size + 2 * pool_kernel_padding) // pool_kernel_stride) + 1

            output_channels = in_channels if len(self.channels) is 0 else self.channels[-1]
            return int(in_h * in_w * output_channels)
            # ========================
        finally:
            torch.set_rng_state(rng_state)

    def _make_mlp(self, out_dim, in_dim=None):
        mlp: MLP = None
        activations = []
        hidden_dims = self.hidden_dims.copy()
        if out_dim is not None:
            hidden_dims += [out_dim]
            activations = [ACTIVATIONS[self.activation_type](**self.activation_params)] * len(self.hidden_dims) + [
                'none']
        if in_dim is None:
            in_dim = self._n_features()
        mlp = MLP(in_dim=in_dim, dims=hidden_dims, nonlins=activations)

        return mlp

    def forward(self, x: Tensor):

        out: Tensor = None
        x = torch.unsqueeze(torch.unsqueeze(x, 2), 3)
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        x = self.features_fc(x)
        out = self.label_classifier(x)

        return x,out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None


        main_layers = []
        shortcut_layers = []
        act = ACTIVATIONS[activation_type](**activation_params)
        mod_channels = [in_channels] + channels
        for i_channel in range(1, len(mod_channels)):
            main_layers += [nn.Conv2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                      kernel_size=kernel_sizes[i_channel - 1], bias=True, padding='same')]
            if i_channel is not len(mod_channels) - 1:
                if dropout > 0:
                    main_layers += [nn.Dropout2d(p=dropout)]
                if batchnorm:
                    main_layers += [nn.BatchNorm2d(mod_channels[i_channel])]
                main_layers += [act]

        if in_channels is not channels[-1]:
            shortcut_layers += [nn.Conv2d(in_channels=in_channels, out_channels=channels[-1],
                                          kernel_size=1, bias=False, padding='same')]

        self.main_path = nn.Sequential(*main_layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)


    def forward(self, x: Tensor):

        out: Tensor = None

        out = self.main_path(x) + self.shortcut_path(x)

        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions, excluding the
            block input and output convolutions.
            For example, if in_out_channels=10 and inner_channels=[5],
            the block will have three convolutions, with channels 10->5->10.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        assert len(inner_channels) > 0
        assert len(inner_channels) == len(inner_kernel_sizes)

        super().__init__(in_channels=in_out_channels, channels=[inner_channels[0]] + inner_channels + [in_out_channels],
                         kernel_sizes=[1] + inner_kernel_sizes + [1], **kwargs)



class ResNet(CNN):
    def __init__(
            self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        bottleneck: bool = False,
        **kwargs,
    ):
        """
        See arguments of CNN & ResidualBlock.
        :param bottleneck: Whether to use a ResidualBottleneckBlock to group together
            pool_every convolutions, instead of a ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.bottleneck = bottleneck
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        default_kernel_size = 3

        p = self.pool_every
        n_channels = len(self.channels)
        pool = POOLINGS[self.pooling_type](**self.pooling_params)

        for i_channel in range(0, len(self.channels), p):

            upper_bound = n_channels if i_channel + p >= n_channels else i_channel + p

            channels = self.channels[i_channel:upper_bound]
            if self.bottleneck and in_channels == channels[-1]:
                in_out_channels = in_channels
                mod_channels = channels[1:-1]
                layers += [ResidualBottleneckBlock(
                    in_out_channels=in_out_channels, inner_channels=mod_channels,
                    inner_kernel_sizes=[default_kernel_size] * len(mod_channels),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params)]
            else:
                layers += [ResidualBlock(
                    in_channels=in_channels, channels=channels, kernel_sizes=[default_kernel_size] * len(channels),
                    batchnorm=self.batchnorm, dropout=self.dropout, activation_type=self.activation_type,
                    activation_params=self.activation_params)]

            in_channels = channels[-1]

            if i_channel+p <= n_channels:
                layers += [pool]

        seq = nn.Sequential(*layers)
        return seq


class YourCNN(CNN):
    def __init__(self, *args, **kwargs):
        """
        See CNN.__init__
        """
        super().__init__(*args, **kwargs)

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []

        default_kernel_size = [3, 5]

        p = self.pool_every
        n_channels = len(self.channels)
        pool = POOLINGS[self.pooling_type](**self.pooling_params)

        for i_channel in range(0, len(self.channels), p):

            upper_bound = n_channels if i_channel + p >= n_channels else i_channel + p
            channels = self.channels[i_channel:upper_bound]
            layers += [InceptionBlock(
                in_channels=in_channels, channels=channels, kernel_sizes=default_kernel_size,
                batchnorm=True, dropout=0.0, activation_type=self.activation_type,
                activation_params=self.activation_params)]

            in_channels = channels[-1]

            if i_channel + p <= n_channels:
                layers += [pool]


        seq = nn.Sequential(*layers)
        return seq



class InceptionBlock(nn.Module):
    """
    A general purpose inception block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: Sequence[int],
            kernel_sizes: Sequence[int],
            batchnorm: bool = False,
            dropout: float = 0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.narrow_path, self.shortcut_path = None, None
        self.pooling_path, self.wide_path = None, None

        layer_params_decreaser = nn.Conv2d(in_channels=in_channels, out_channels=channels[0],
                                           kernel_size=1, bias=True, padding='same')
        narrow_layers = [layer_params_decreaser]
        wide_layers = [layer_params_decreaser]
        pooling_layers = [
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1,ceil_mode=True),
            nn.Conv2d(in_channels=in_channels, out_channels=channels[-1],
                      kernel_size=1, bias=True, padding='same')
        ]
        shortcut_layers = []
        act = ACTIVATIONS[activation_type](**activation_params)
        mod_channels = [channels[0]] +channels
        for i_channel in range(1, len(mod_channels)):
            narrow_layers += [nn.Conv2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                        kernel_size=kernel_sizes[0], bias=True, padding='same')]
            wide_layers += [nn.Conv2d(in_channels=mod_channels[i_channel - 1], out_channels=mod_channels[i_channel],
                                      kernel_size=kernel_sizes[1], bias=True, padding='same')]
            if i_channel is not len(mod_channels) - 1:
                if dropout > 0:
                    narrow_layers += [nn.Dropout2d(p=dropout)]
                    wide_layers += [nn.Dropout2d(p=dropout)]
                if batchnorm:
                    narrow_layers += [nn.BatchNorm2d(mod_channels[i_channel])]
                    wide_layers += [nn.BatchNorm2d(mod_channels[i_channel])]
                narrow_layers += [act]
                wide_layers += [act]

        if in_channels is not channels[-1]:
            shortcut_layers += [nn.Conv2d(in_channels=in_channels, out_channels=channels[-1],
                                          kernel_size=1, bias=False, padding='same')]

        self.pooling_path = nn.Sequential(*pooling_layers)
        self.narrow_path = nn.Sequential(*narrow_layers)
        self.wide_path = nn.Sequential(*wide_layers)
        self.shortcut_path = nn.Sequential(*shortcut_layers)
        # ========================

    def forward(self, x: Tensor):
        out: Tensor = None

        out = self.narrow_path(x) + self.wide_path(x) + self.pooling_path(x) + self.shortcut_path(x)
        # ========================
        out = torch.relu(out)
        return out

