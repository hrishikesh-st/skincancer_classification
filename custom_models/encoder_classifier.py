import torch
import torch.nn as nn


class EncoderClassifier(nn.Module):
    def __init__(self, encoder, dropout_rate=0.3):
        super(EncoderClassifier, self).__init__()
        self.encoder = encoder
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1280, 128)
        self.fc2 = nn.Linear(128, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)

        x = self.dropout(self.fc1(x))
        x = self.relu(x)

        x = self.dropout(self.fc2(x))
        x = self.relu(x)

        x = self.fc3(x)
        return x
