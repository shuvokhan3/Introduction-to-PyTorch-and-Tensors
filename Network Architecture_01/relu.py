import torch
import torch.nn as nn

torch.manual_seed(42)

# Model with Leaky ReLU instead of ReLU
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.LeakyReLU(negative_slope=0.01),  # Leaky ReLU activation
    nn.Linear(8, 3)
)

# Sample input
x = torch.randn(2, 4)

# Forward pass
output = model(x)

print(f"Input shape:  {x.shape}")
print(f"Output shape: {output.shape}")