import torch
import torch.nn as nn

data = torch.randn(5,6)
print("row data shape : ", data.shape)

layer1 = nn.Linear(in_features=6,out_features=4)
x = layer1(data)
print("layer 1 output shape : ", x.shape) #[5,4]


layer2 = nn.Linear(in_features=4,out_features=2)
output = layer2(x)
print("layer 2 output shape : ", output.shape)#[5,2]

#The batch size (5 samples) is preserved, while features transform: 6 → 4 → 1.






