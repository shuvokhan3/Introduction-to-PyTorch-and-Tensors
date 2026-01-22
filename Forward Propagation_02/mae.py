import torch
import torch.nn as nn

mae_criterion = nn.L1Loss()  # L1 Loss = MAE

predictions = torch.tensor([2.5, 0.5, 2.0])
targets = torch.tensor([3.0, 0.5, 1.5])

loss = mae_criterion(predictions, targets)
print(f"MAE Loss: {loss.item():.4f}")  # 0.3333