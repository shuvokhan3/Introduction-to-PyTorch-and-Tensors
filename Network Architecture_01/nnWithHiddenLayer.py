import torch
import torch.nn as nn

Data = torch.tensor([
    [1,2,3,4,5,6,5,4,3,2],
    [1,2,3,4,5,6,5,4,3,2]
], dtype=torch.float32)


model = nn.Sequential(
    nn.Linear(in_features=10, out_features=20),#input layer
    nn.Linear(in_features=20, out_features=30),#hidden layer
    nn.Linear(in_features=30, out_features=10),#hidden layer
    nn.Linear(in_features=10, out_features=2) # Output layar
)


