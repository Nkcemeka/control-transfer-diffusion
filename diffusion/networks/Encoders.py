import torch
import torch.nn as nn
import cached_conv as cc
import gin
from torch.nn import init


def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        # The hook we want to remove is an instance of WeightNorm class, so
        # normally we would do `if isinstance(...)` but this class is not accessible
        # because of shadowing, so we check the module name directly.
        # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            print("Removing weight_norm from %s", self.__class__.__name__)
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')


@gin.configurable
class V2ConvBlock1D(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, act=nn.SiLU, res=True):
        super().__init__()
        self.res = res

        self.conv1 = nn.Conv1d(in_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")
        self.conv2 = nn.Conv1d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding="same")

        self.gn1 = nn.GroupNorm(min(in_c, 16), in_c)
        self.gn2 = nn.GroupNorm(min(out_c, 16), out_c)
        self.act = act()
        self.dp = nn.Dropout(p=0.15)

    def forward(self, x):
        if self.res:
            res = x.clone()

        x = self.gn1(x)
        x = self.act(x)

        x = self.conv1(x)
        x = self.gn2(x)
        x = self.act(x)
        x = self.dp(x)
        x = self.conv2(x)

        if self.res:
            return x + res

        return x


@gin.configurable
class V2EncoderBlock1D(nn.Module):

    def __init__(self, in_c, out_c, kernel_size, ratio):
        super().__init__()
        self.conv = V2ConvBlock1D(in_c, in_c, kernel_size)
        if ratio == 1:
            pad = "same"
        else:
            pad = ((2 * ratio - 1) // 2)
        self.pool = nn.Conv1d(in_c,
                              out_c,
                              kernel_size=2 * ratio,
                              stride=ratio,
                              padding=pad)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


@gin.configurable
class Encoder1D(nn.Module):

    def __init__(self,
                 in_size=1,
                 channels=[64, 128, 128, 256],
                 ratios=[4, 4, 2, 2],
                 kernel_size=3,
                 use_tanh=True,
                 average_out=True,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.average_out = average_out
        self.use_tanh = use_tanh
        n = len(self.channels)

        self.down_layers = nn.ModuleList()

        self.down_layers.append(
            V2EncoderBlock1D(in_size,
                             channels[0],
                             kernel_size,
                             ratio=ratios[0]))

        for i in range(1, n):
            self.down_layers.append(
                V2EncoderBlock1D(channels[i - 1], channels[i], kernel_size,
                                 ratios[i]))

        self.middle_block = V2ConvBlock1D(channels[-1], channels[-1],
                                          kernel_size)

    @torch.jit.export
    def forward(self, x):

        for layer in self.down_layers:
            x = layer(x)

        x = self.middle_block(x)

        if self.average_out:
            x = torch.mean(x, axis=-1)

        if self.use_tanh:
            x = torch.tanh(x)

        return x


@gin.configurable
class LinearEncoder(nn.Module):

    def __init__(self,
                 in_size=512,
                 channels=[512, 1024, 1024, 256, 8],
                 drop_out=0.15):
        #out_fn=nn.Identity(),
        #**kwargs):
        super().__init__()
        module_list = []
        module_list.append(nn.Linear(in_size, channels[0]))

        for i in range(len(channels) - 1):
            module_list.append(nn.SiLU())
            module_list.append(nn.Dropout(p=drop_out))
            module_list.append(nn.Linear(channels[i], channels[i + 1]))

        self.net = nn.Sequential(*module_list)

        #self.out_fn = out_fn

    def forward(self, x):
        return torch.tanh(self.net(x))

class TrascriptionModel(nn.Module):

    def __init__(self):
        """TrascriptionModel class constructor. This class is intended to be used as a base model for all networks.
        """
        super().__init__()

    def init_weights(self):
        """Function to apply weight initialisation. Needs to be called once after the model is instantiated.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, gain=1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight, init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """Fuction to run a forward pass. Returns logits.
        Parameters
        ----------
        x (torch Tensor): input feature

        Returns
        -------
        (torch Tensor): model's output
        """
        return x

    # returns pseudo probabilities
    def predict(self, x):
        """Function to return model's prediction for a given input feature. This function will return pseudo-probabilities
        which will then need to be transformed into a multi-instrument roll using suitable thresholds.

        Parameters
        ----------
        x (torch Tensor): input feature

        Returns
        -------
        (torch Tensor): model's prediction in the form of pseudo-probabilities
        """
        return torch.sigmoid(self.forward(x))


class LinearModel(TrascriptionModel):
    def __init__(self, in_features=1024, out_features=128, hidden_units=512, dropout_prob=0.2):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_units = hidden_units
        self.dropout_prob = dropout_prob

        self.linear = nn.Sequential(
            nn.Linear(self.in_features, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob),
            nn.Linear(self.hidden_units, self.out_features)
        )


    def forward(self, x):
        """Function to run a forward pass.

        Parameters
        ----------
        x (torch Tensor): input feature [BATCH X TIME X FEATURES]

        Returns
        -------
        x (torch Tensor): predictions for this feature [BATCH x TIME X CLASSES]
        """
        output = self.linear(x)
        return output

@gin.configurable
class PitchEncoder1D(nn.Module):

    def __init__(self,
                 in_size=1,
                 channels=[64, 128, 128, 256],
                 ratios=[4, 4, 2, 2],
                 kernel_size=3,
                 use_tanh=True,
                 average_out=True,
                 **kwargs):
        super().__init__()
        self.channels = channels
        self.average_out = average_out
        self.use_tanh = use_tanh
        n = len(self.channels)

        self.down_layers = nn.ModuleList()

        self.down_layers.append(
            V2EncoderBlock1D(in_size,
                             channels[0],
                             kernel_size,
                             ratio=ratios[0]))

        for i in range(1, n):
            self.down_layers.append(
                V2EncoderBlock1D(channels[i - 1], channels[i], kernel_size,
                                 ratios[i]))

        self.middle_block = V2ConvBlock1D(channels[-1], channels[-1],
                                          kernel_size)

        self.pc = LinearModel(in_features=channels[-1])

    @torch.jit.export
    def forward(self, x):

        for layer in self.down_layers:
            x = layer(x)

        x = self.middle_block(x)

        if self.average_out:
            x = torch.mean(x, axis=-1)

        if self.use_tanh:
            x = torch.tanh(x)

        pitch_preds = self.pc(x.transpose(-1, -2))

        return x, pitch_preds
