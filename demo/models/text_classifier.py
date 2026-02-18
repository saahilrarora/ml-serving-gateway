import torch
import torch.nn as nn


class TextClassifier(nn.Module):
    """A minimal 2-layer classifier for demo purposes.
    Input:  vector of size 10 (simulates a text embedding)
    Output: probabilities over 3 classes
    """

    def __init__(self, input_size=10, hidden_size=32, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # first layer: 10 → 32
        self.relu = nn.ReLU()                           # activation function
        self.fc2 = nn.Linear(hidden_size, num_classes)  # second layer: 32 → 3
        self.softmax = nn.Softmax(dim=-1)               # convert raw scores to probabilities

    def forward(self, x):
        x = self.fc1(x)       # linear transform
        x = self.relu(x)      # non-linearity
        x = self.fc2(x)       # linear transform to class scores
        x = self.softmax(x)   # normalize to probabilities that sum to 1.0
        return x

    @staticmethod
    def get_sample_input():
        """Returns a dummy input tensor for testing and ONNX export."""
        return torch.randn(1, 10)  # batch of 1, input size 10
