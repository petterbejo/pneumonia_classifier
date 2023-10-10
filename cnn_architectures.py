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
