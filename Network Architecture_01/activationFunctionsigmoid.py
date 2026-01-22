import torch
import torch.nn as nn

# Input: image features (e.g., from a feature extractor)
# Shape: [1 sample, 4 features]
image_features = torch.tensor([[0.8, 0.2, 0.6, 0.9]])

#network with sigmoid activeation

model = nn.Sequential(
    nn.Linear(4,8),
    nn.Linear(8, 4),
    nn.Linear(4,1),
    nn.Sigmoid()
)

output = model(image_features)


if(output.item() > 0.5):
    print("It is a dog")
else:
    print("It is not a dog")




