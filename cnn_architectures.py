from torch import nn

class VeryBasicNN(nn.Module):
    """Model to test the classifier."""
    def __init__(self, img_size):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(img_size[0]*img_size[1], 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return self.sigmoid(logits)

class BasicCNN(nn.Module):
    """Vanilla CNN."""
    def __init__(self, img_size):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32,
                               kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64,
                               kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        logits = self.fc2(x)
        return self.sigmoid(logits)
