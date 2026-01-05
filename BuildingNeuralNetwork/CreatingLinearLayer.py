import torch
import torch.nn as nn

# Create input tensor: 1 sample with 3 features
input_tensor = torch.tensor([[0.5, 0.8, 0.2]])
print("Input shape:", input_tensor.shape)

#Create a linear layer 3 -> input 2 -> output
linear_layer = nn.Linear(in_features=3, out_features=2)

#Pass input throught the layer
output = linear_layer(input_tensor)

print("Output shape:", output.shape)

print(output)


