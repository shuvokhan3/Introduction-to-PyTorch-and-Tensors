import torch
import torch.nn as nn


#Create a sample of data
data = torch.tensor([
    [1, 2, 3]
], dtype=torch.float32)

#Create a Linear Layer
layer = nn.Linear(in_features=3, out_features=2)

#pass data in the layer
output = layer(data)

print("Output value is : ",output)

#Check weight and bias
weight = layer.weight
print("Weight value is : ",weight)
print("Weight shape is : ", weight.shape)

bias = layer.bias
print("Bias value is : ",bias)
print("Bias shape is : ", bias.shape)







