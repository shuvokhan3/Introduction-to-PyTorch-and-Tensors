import torch
import torch.nn as nn
from numpy.ma.core import equal

# Create two identical layers - different random values!
layer_1 = nn.Linear( 10,3)
layer_2 = nn.Linear(10, 3)

print(layer_1.weight)
print(layer_2.weight)

print( torch.equal(layer_1.weight, layer_2.weight) )