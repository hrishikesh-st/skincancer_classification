import torch
import torch.nn as nn
import torchvision.models as models

class UNetMobileNet(nn.Module):
    def __init__(self):
        super(UNetMobileNet, self).__init__()
        # load MobileNetV2 model
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.encoder = mobilenet.features

        # skip connections
        self.skip_layers = [1, 3, 6, 13]

        # decoder layers
        self.upconv1 = nn.ConvTranspose2d(1280, 512, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(512+96, 256, kernel_size=4, stride=2, padding=1)  # Skip connection
        self.upconv3 = nn.ConvTranspose2d(256+32, 128, kernel_size=4, stride=2, padding=1)  # Skip connection
        self.upconv4 = nn.ConvTranspose2d(128+24, 64, kernel_size=4, stride=2, padding=1)  # Skip connection
        self.upconv5 = nn.ConvTranspose2d(64+16, 3, kernel_size=4, stride=2, padding=1)   # Skip connection

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skip_connections = []
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            if i in self.skip_layers:
                skip_connections.append(x)

        # decoder forward pass
        x = self.relu(self.upconv1(x))
        x = torch.cat((x, skip_connections[-1]), dim=1)
        x = self.relu(self.upconv2(x))
        x = torch.cat((x, skip_connections[-2]), dim=1)
        x = self.relu(self.upconv3(x))
        x = torch.cat((x, skip_connections[-3]), dim=1)
        x = self.relu(self.upconv4(x))
        x = torch.cat((x, skip_connections[-4]), dim=1)
        x = self.sigmoid(self.upconv5(x))
        return x