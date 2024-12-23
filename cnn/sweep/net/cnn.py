import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, conv_size, layer_size, dropout):
        super(Net, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, conv_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_size, conv_size*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_size*2, conv_size*3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_size*3, conv_size*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_size*4, conv_size*5, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(conv_size*5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(16 * 3 * 3, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size, layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(layer_size, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


