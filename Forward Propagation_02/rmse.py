import torch
import torch.nn as nn

# PyTorch doesn't have built-in RMSE, but it's easy to compute
mse_criterion = nn.MSELoss()
predictions = torch.tensor([25.0, 30.0, 15.0])
targets = torch.tensor([24.0, 35.0, 16.0])

mse_loss = mse_criterion(predictions, targets)
rmse_loss = torch.sqrt(mse_loss)

print(f"MSE: {mse_loss.item():.4f}")   # 9.0000
print(f"RMSE: {rmse_loss.item():.4f}") # 3.0000