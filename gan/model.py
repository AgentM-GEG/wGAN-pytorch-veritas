from torch import nn
from einops.layers.torch import Rearrange
from collections import OrderedDict

class UpConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, norm=True, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        layers = [nn.ConvTranspose2d(in_channels, out_channels, **kwargs)]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [activation]

        super().__init__(*layers)


class DownConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, activation, iteration, norm=False, **kwargs):
        if activation == 'sigmoid':
            activation = nn.Sigmoid()
        elif activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)

        layers = [nn.Conv2d(in_channels, out_channels, **kwargs)]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [activation]

        super().__init__(*layers)

# class DownConvLayer(nn.Sequential):
#     def __init__(self, in_channels, out_channels, activation, iteration, norm=False, **kwargs):
#         if activation == 'sigmoid':
#             activation = nn.Sigmoid()
#         elif activation == 'leakyrelu':
#             activation = nn.LeakyReLU(0.2)

#         layers = {f'conv2d_{iteration}': nn.Conv2d(in_channels, out_channels, **kwargs)}
#         if norm:
#             layers[f'norm_{iteration}'] = nn.InstanceNorm2d(out_channels)
        
#         layers[f'act_{iteration}'] = activation

#         super().__init__(OrderedDict(nn.ModuleDict(layers)))

class Generator(nn.Sequential):
    def __init__(self, n_z, input_filt=512, norm=False, n_layers=5, out_channels=3, final_size=256):
        self.n_z = n_z

        layers = []

        prev_filt = input_filt
        for _ in range(n_layers):
            layers.append(UpConvLayer(prev_filt, int(prev_filt / 2), activation='leakyrelu', norm=norm,
                                      kernel_size=(6, 6), stride=(2, 2), padding=2))
            prev_filt = int(prev_filt / 2)

        initial_size = final_size / 2 ** n_layers
        if initial_size % 1 != 0:
            raise ValueError(f"Cannot create a model to produce a {final_size} x {final_size} image with {n_layers} layers")

        initial_size = int(initial_size)

        super().__init__(
            nn.Linear(n_z, initial_size * initial_size * input_filt),
            nn.LeakyReLU(0.2, True),
            Rearrange('b (h w z) -> b z h w', h=initial_size, w=initial_size, z=input_filt),
            *layers,
            nn.Conv2d(prev_filt, out_channels, (5, 5), stride=(1, 1), padding=2),
            nn.Sigmoid()
        )


class Discriminator(nn.Sequential):
    def __init__(self, in_channels, n_layers=5, input_size=256):
        prev_filt = 8
        layers = []
        for i in range(n_layers):
            layers.append(DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=i))
            prev_filt = prev_filt * 2
            input_size = input_size / 2

        super().__init__(
            *layers,
            Rearrange('b z h w -> b (z h w)'),
            nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
        )

# class Discriminator(nn.Sequential):
#     def __init__(self, in_channels, n_layers=5, input_size=256):
#         prev_filt = 8
#         modules = {}
#         for i in range(n_layers):
#             modules[f'Down_{i}'] = DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
#                                         kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=i)
#             prev_filt = prev_filt * 2
#             input_size = input_size / 2
#         modules[f'rearr_final'] = Rearrange('b z h w -> b (z h w)')
#         modules[f'act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, 1)
#         module_dict = nn.ModuleDict(modules)

#         super().__init__( OrderedDict(module_dict)
            
#         )

class DiscriminatorFeatures(nn.Module):
    def __init__(self, in_channels, n_layers=5, input_size=256):
        super().__init__()
        
        self.Down_0 = DownConvLayer(1, 16, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=0)
        self.Down_1 = DownConvLayer(16, 32, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=1)
        
        self.Down_2 = DownConvLayer(32, 64, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=2)
        
        self.Down_3 = DownConvLayer(64, 128, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=3)
        
        self.rearr_final = Rearrange('b z h w -> b (z h w)')
        self.act_final = nn.Linear(6 * 6 * 128, 1)
        
        
    def forward(self, x):
        x = self.Down_0(x)
        x = self.Down_1(x)
        x = self.Down_2(x)
        x = self.Down_3(x)
        feats = self.rearr_final(x)
        output = self.act_final(feats)
        return output, feats
        
        
        

class Encoder(nn.Sequential):
    def __init__(self, in_channels , n_z = 128, n_layers = 5, input_size = 256):
        prev_filt = 8
        modules = {}
        for i in range(n_layers):
            modules[f'Down_{i}'] = DownConvLayer(prev_filt if i > 0 else in_channels, prev_filt * 2, activation='leakyrelu',
                                        kernel_size=(6, 6), stride=(2, 2), padding=2, norm=False, iteration=i)
            prev_filt = prev_filt * 2
            input_size = input_size / 2
        modules[f'rearr_final'] = Rearrange('b z h w -> b (z h w)')
        modules[f'act_final'] = nn.Linear(int(input_size) * int(input_size) * prev_filt, n_z)
        module_dict = nn.ModuleDict(modules)

        super().__init__( OrderedDict(module_dict)
            
        )