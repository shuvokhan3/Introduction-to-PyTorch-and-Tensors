import torch


#work with .shape
tensor_a = torch.tensor([
    [1,2,3,4],
    [3,4,5,6],
    [7,8,9,10]
])

tensor_b = torch.tensor([1,2,3,4,5])
print("tensor_a shape : ", tensor_a.shape)
print("tensor_b shape : ", tensor_b.shape)

print("Indivisual index shape : ", tensor_a[0].shape)

print("Indivisual index shape : ", tensor_b[2].shape)



#work with data type of tensor dtype
print("dtype of tensor_a : ", tensor_a.dtype)
print("dtype of tensor_b : ", tensor_b.dtype)

# Default float type
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print("Float tensor dtype:", float_tensor.dtype)

#specify the dtype
float_tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
print("Float tensor customize dtype:",  float_tensor2.dtype)

# Integer tensor
int_tensor = torch.tensor([1, 2, 3])
print("Int tensor dtype:", int_tensor.dtype)

# Boolean tensor
bool_tensor = torch.tensor([True, False, True])
print("Bool tensor dtype:", bool_tensor.dtype)


#Device
#The device attribute shows where the tensor is stored (CPU or GPU):

#I want to check tesor_a where this tensor are stored (CPU or GPU)
print("Tensor_a is stored in a : ",tensor_a.device)


print("----------------------")

#create tensor in CPU

cpu_tensor = torch.tensor([1, 2, 3])
print("Device:", cpu_tensor.device)

# Check if GPU is available
if torch.cuda.is_available():
    # Create tensor directly on GPU
    gpu_tensor = torch.tensor([1, 2, 3], device='cuda')
    print("GPU Device:", gpu_tensor.device)

    # Move CPU tensor to GPU
    cpu_to_gpu = cpu_tensor.to('cuda')
    print("Moved to GPU:", cpu_to_gpu.device)
else:
    print("CUDA not available - using CPU")























