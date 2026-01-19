import torch
import torch.nn.functional as F

#label data
labels = torch.tensor([1,2,3,4])

#convert one-hot this label data (binery vector)
one_hot_labels = F.one_hot(labels,num_classes=5).float()
print(one_hot_labels)


#back in label data (binary vector to label data)
back_in_label_data = torch.argmax(one_hot_labels,dim=1)
print(back_in_label_data)




