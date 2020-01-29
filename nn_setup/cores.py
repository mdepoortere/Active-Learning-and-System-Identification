import torch
import torch.nn as nn
from mlutils.layers.cores import Core2d
from collections import OrderedDict
from mlutils import regularizers


class Stacked2dCoreDropOut(Core2d, nn.Module):
    def __init__(
        self,
        input_channels,
        hidden_channels,
        input_kern,
        hidden_kern,
        dropout_p=0.1,
        layers=3,
        gamma_hidden=0,
        gamma_input=0.0,
        skip=0,
        final_nonlinearity=True,
        bias=False,
        momentum=0.1,
        pad_input=True,
        batch_norm=True,
        hidden_dilation=1,
        laplace_padding=0,
        input_regularizer="LaplaceL2",
    ):
        """
        Args:
            input_channels:     Integer, number of input channels as in
            hidden_channels:    Number of hidden channels (i.e feature maps) in each hidden layer
            input_kern:     kernel size of the first layer (i.e. the input layer)
            hidden_kern:    kernel size of each hidden layer's kernel
            layers:         number of layers
            gamma_hidden:   regularizer factor for group sparsity
            gamma_input:    regularizer factor for the input weights (default: LaplaceL2, see mlutils.regularizers)
            skip:           Adds a skip connection
            final_nonlinearity: Boolean, if true, appends an ELU layer after the last BatchNorm (if BN=True)
            bias:           Adds a bias layer. Note: bias and batch_norm can not both be true
            momentum:       BN momentum
            pad_input:      Boolean, if True, applies zero padding to all convolutions
            batch_norm:     Boolean, if True appends a BN layer after each convolutional layer
            hidden_dilation:    If set to > 1, will apply dilated convs for all hidden layers
            laplace_padding: Padding size for the laplace convolution. If padding = None, it defaults to half of
                the kernel size (recommended). Setting Padding to 0 is not recommended and leads to artefacts,
                zero is the default however to recreate backwards compatibility.
            normalize_laplace_regularizer: Boolean, if set to True, will use the LaplaceL2norm function from
                mlutils.regularizers, which returns the regularizer as |laplace(filters)| / |filters|

            input_regularizer: String that must match one of the regularizers in ..regularizers
        """

        super().__init__()

        assert not bias or not batch_norm, "bias and batch_norm should not both be true"

        regularizer_config = (
            dict(padding=laplace_padding, kernel=input_kern)
            if input_regularizer == "GaussianLaplaceL2"
            else dict(padding=laplace_padding)
        )
        self._input_weights_regularizer = regularizers.__dict__[input_regularizer](**regularizer_config)

        self.layers = layers
        self.gamma_input = gamma_input
        self.gamma_hidden = gamma_hidden
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.skip = skip
        self.features = nn.Sequential()

        # --- first layer
        layer = OrderedDict()
        layer["conv"] = nn.Conv2d(
            input_channels, hidden_channels, input_kern, padding=input_kern // 2 if pad_input else 0, bias=bias
        )
        if batch_norm:
            layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
        if layers > 1 or final_nonlinearity:
            layer["nonlin"] = nn.ELU(inplace=True)
        layer['drop'] = nn.Dropout(p=dropout_p, inplace=False)
        self.features.add_module("layer0", nn.Sequential(layer))

        # --- other layers
        h_pad = ((hidden_kern - 1) * hidden_dilation + 1) // 2
        for l in range(1, self.layers):
            layer = OrderedDict()
            layer["conv"] = nn.Conv2d(
                hidden_channels if not skip > 1 else min(skip, l) * hidden_channels,
                hidden_channels,
                hidden_kern,
                padding=h_pad,
                bias=bias,
                dilation=hidden_dilation,
            )
            if batch_norm:
                layer["norm"] = nn.BatchNorm2d(hidden_channels, momentum=momentum)
            if final_nonlinearity or l < self.layers - 1:
                layer["nonlin"] = nn.ELU(inplace=True)
            layer['drop'] = nn.Dropout(p=dropout_p, inplace=False)
            self.features.add_module("layer{}".format(l), nn.Sequential(layer))

        self.apply(self.init_conv)

    def forward(self, input_):
        ret = []
        for l, feat in enumerate(self.features):
            do_skip = l >= 1 and self.skip > 1
            input_ = feat(input_ if not do_skip else torch.cat(ret[-min(self.skip, l):], dim=1))
            ret.append(input_)
        return torch.cat(ret, dim=1)

    def laplace(self):
        return self._input_weights_regularizer(self.features[0].conv.weight)

    def group_sparsity(self):
        ret = 0
        for l in range(1, self.layers):
            ret = ret + self.features[l].conv.weight.pow(2).sum(3, keepdim=True).sum(2, keepdim=True).sqrt().mean()
        return ret / ((self.layers - 1) if self.layers > 1 else 1)

    def regularizer(self):
        return self.group_sparsity() * self.gamma_hidden + self.gamma_input * self.laplace()

    @property
    def outchannels(self):
        return len(self.features) * self.hidden_channels

