import string
import torch.nn as nn

LETTERS = string.ascii_uppercase          # EMNIST Letters merges upper/lower case -> 26 classes
MEAN, STD = 0.1736, 0.3317                # approx. EMNIST pixel stats


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Net(nn.Module):
    def __init__(self, n_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            block(1, 32), nn.MaxPool2d(2), nn.Dropout(0.1),     # 28 -> 14
            block(32, 64), nn.MaxPool2d(2), nn.Dropout(0.2),    # 14 -> 7
            block(64, 128), nn.MaxPool2d(2), nn.Dropout(0.3),   # 7 -> 3
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 3 * 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.4), nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))
