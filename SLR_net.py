import torch
import torch.nn as nn

from torchsummary import summary


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)

class SLRnet(nn.Module):
    def __init__(self, in_channel=1, out_channel=4, output_size=128, add_fc : bool=True, filters=[64, 128, 256, 512]):
        super(SLRnet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(in_channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.residual_conv_3= ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], out_channel, 1, 1), # The number of output channels is set to 2 for non-orthogonal multiplexing with phase modulation only or amplitude modulation only, 
                                                      # and the number of output channels is set to 4 for non-orthogonal multiplexing with complex amplitude modulation.
            nn.Sigmoid(),
        )

        self.add_fc = add_fc
        self.input_size = 200
        self.middle_size = 128
        if self.add_fc:
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.input_size * self.input_size, self.middle_size * self.middle_size),
                nn.Dropout(0.02),
                nn.Unflatten(dim=1, unflattened_size=(1, self.middle_size, self.middle_size)),

            )
        self.output_size = output_size
        if self.output_size !=  self.middle_size:
            self.AAP = nn.AdaptiveAvgPool2d(self.output_size)
            
    def forward(self, x):
        if self.add_fc:
            x = self.fc(x)
        if self.output_size !=  self.middle_size:
            x = self.AAP(x)
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        x4 = self.residual_conv_3(x3)
        
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        output = self.output_layer(x10)

        return output
    
    
    
if __name__=='__main__':



    model = SLRnet(in_channel=1, out_channel=4, output_size=256, add_fc=True, filters=[64, 128, 256, 512])

    x = torch.randn(6,1,200,200)
    b = model(x)
    print(b.shape)
