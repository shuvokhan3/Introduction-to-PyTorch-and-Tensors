import numpy as np
import torch

#zero tensor
#Use case: Initialize weight matrices, create placeholder tensors

zero_tensor = torch.zeros(3,4)
print(zero_tensor)
print("----")
#create tensor file with one
#Use case: Masking, normalization, bias initialization
one_tensor = torch.ones(3,4)
print(one_tensor)
print("----")

#Creates a tensor with random values from uniform distribution [0, 1).
# Set seed for reproducibility
#Use case: Weight initialization, data augmentation, sampling
torch.manual_seed(42)

random_tensor = torch.rand(3,4)
print(random_tensor)
print("-----")
#Creates a tensor with random values from standard normal distribution (mean=0, std=1).

randn_tensor = torch.randn(5,6)
print(randn_tensor)

print("-----")
#torch.arange()
#Use case: Creating indices, positional encodings, plotting
#create sequence from 0 to 9
seq1 = torch.arange(10)
print(seq1)

#create sequence from 2 to 10 with step 2
seq2 = torch.arange(2,11,2)
print(seq2)

#create float seq
seq3 = torch.arange(0.0,1.0,0.4)
print(seq3)



#torch.linspace()
#torch.linspace(start, end, how_many_numbers)

linespace_tensor = torch.linspace(1,100,5)
print("Linspace Value : ", linespace_tensor)

print("........")
# Create 7 points between -1 and 1
point = torch.linspace(-1,1, 7)
print("point value : ", point)


































