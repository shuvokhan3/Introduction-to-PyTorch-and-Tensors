import torch
import torch.nn as nn

X = torch.tensor([
    [85, 60, 40],
    [90, 55, 30],
    [40, 85, 90],
    [45, 88, 85],
    [70, 65, 60],
    [75, 40, 90],
], dtype=torch.float32)

layer = nn.Linear(in_features=3, out_features=2)

output = layer(X)

final_output = torch.softmax(output, dim=0)
print(final_output)