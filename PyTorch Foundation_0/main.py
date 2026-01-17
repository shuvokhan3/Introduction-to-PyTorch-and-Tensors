import torch
import numpy as np

print(torch.__version__)


#Creating tensor form python list 1D tensor
data = [1,2,3,4,5,6]
tensor_1D = torch.tensor(data)
print("1D tensor : ",tensor_1D)
print(tensor_1D.dtype)


# Create a 2D tensor (matrix)
data_matrix = [[1.0, 2.0, 3.0],
               [4.0, 5.0, 6.0],
               [7.0, 8.0, 9.0]]

tensor_2D = torch.tensor(data_matrix)
print(tensor_2D)
print(tensor_2D.dtype)

#NumPy array to tensor
numpy_array = np.array([1,2,3,4,5,6])
tensor_array = torch.from_numpy(numpy_array)
print("Tensor array : ",tensor_array)


# Torch to numpy
numpy_arr = tensor_array.numpy()
print("torch_to_numpy : ",numpy_arr)

# tensor_array = torch.tensor(numpy_array)
# print(tensor_array.dtype)
























