import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    def __init__(self, num_classes, num_layers=4, in_channels=1,
                 init_num_fmaps=8, fmap_growth=2, pool='avg', fc_size=20):
        super().__init__()
        self.in_channels = in_channels
        self.channels = [in_channels, ] + [init_num_fmaps * fmap_growth**i
                                           for i in range(num_layers)]
        self.conv_layers = self.get_conv_layers(num_layers)
        assert pool in ['avg', 'max']
        self.pool = nn.AdaptiveAvgPool2d(1) if pool == 'avg' else nn.AdaptiveMaxPool3d(1)
        self.fc1 = nn.Linear(self.channels[-1], fc_size)
        self.fc2 = nn.Linear(fc_size, num_classes)

    def get_conv_layers(self, num):
        layers = []
        for i in range(num):
            in_ch, out_ch = self.channels[i], self.channels[i + 1]
            layer = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(layer)
        return nn.ModuleList(layers)

    def embed(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, inp):
        inp = self.embed(inp)
        inp = self.fc2(inp)
        inp = torch.squeeze(inp, dim=1)
        return inp
