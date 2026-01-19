import torch
from pyexpat import features

#Mean squre error for calculate loss
mse_criterion = torch.nn.MSELoss()

#feature data
features_data = torch.tensor([2.5, 0.5, 2.0])
labels_data = torch.tensor([3.0, 0.5, 1.5])

#calculate loss
loss = mse_criterion(features_data,labels_data)
print(loss)

