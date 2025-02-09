import torch.nn as nn

class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # flattened size: 64 * (224//8) * (224//8) = 64 * 28 * 28 = 50176
        self.classifier = nn.Sequential(
            nn.Linear(64 * (224 // 8) * (224 // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  
        x = self.classifier(x)  
        return x
