import torch
import torch.nn.functional as F

# Tensor dataset
data = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

#work with first feature
first_data = data[0][0]
second_data = data[0][1]
third_data = data[0][2]

#convert those data into one hot
first_data_label = F.one_hot(first_data, num_classes=10)
second_data_label = F.one_hot(second_data, num_classes=10)
third_data_label = F.one_hot(third_data, num_classes=10)

print(first_data_label)
print(second_data_label)
print(third_data_label)




