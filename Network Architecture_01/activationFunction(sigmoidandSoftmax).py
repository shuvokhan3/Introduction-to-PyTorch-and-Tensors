import torch
import torch.nn as nn

# Test with different raw values
raw_values = torch.tensor([[-5.0], [-2.0], [0.0], [2.0], [5.0]])

sigmoid = nn.Sigmoid()

for raw_value in raw_values:
    value = sigmoid(raw_value)
    print(raw_value, value)


