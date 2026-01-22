import torch
import torch.nn as nn

# Input: image features
# Shape: [1 sample, 4 features]
image_features = torch.tensor([[0.8, 0.2, 0.6, 0.9]])

# Network with softmax activation
model = nn.Sequential(
    nn.Linear(4, 8),        # 4 features -> 8 hidden neurons
    nn.Linear(8, 4),        # 8 hidden -> 4 classes (Cat, Rabbit, Fish, Dog)
    nn.Softmax(dim=-1)      # Convert to probability distribution
)

# Get prediction
probabilities = model(image_features)
print(f"Probabilities: {probabilities}")
print(f"Sum: {probabilities.sum().item():.4f}")  # Should be 1.0

# Get predicted class
class_names = ['Cat', 'Rabbit', 'Fish', 'Dog']
predicted_class = torch.argmax(probabilities, dim=-1)
print(f"Prediction: {class_names[predicted_class.item()]}")