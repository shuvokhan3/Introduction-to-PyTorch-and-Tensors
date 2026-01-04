import torch
import torch.nn as nn

# # Weather data: [temperature, humidity, wind_speed]
# # Values are normalized (scaled between 0 and 1)
# weather_data = torch.tensor([[0.5, 0.8, 0.2]])  # Hot, very humid, light wind
weather_data = torch.tensor([1,2,3], dtype=torch.float32)

#Create network
layer = nn.Linear(in_features=3, out_features=2)

output = layer(weather_data)
print("Output : ", output)

