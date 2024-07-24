import torch
from torchvision import transforms
from PIL import Image
import io
import torch
from torch import nn, optim

# Load the model class definition
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 18 * 18, 128)  # Adjust input size according to your image size after pooling
        self.fc2 = nn.Linear(128, 1)  # Binary classification (1 output neuron)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 128 * 18 * 18)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize the model
model = SimpleCNN()
model.load_state_dict(torch.load('forest_model.pth'))
model.eval()

# Image preparation function
def prepare_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Prediction function
def predict(image_bytes):
    image_tensor = prepare_image(image_bytes)
    with torch.no_grad():
        output = model(image_tensor)
        predicted = (output > 0.5).float()
    return predicted.item()

# Example usage with an image file
with open(r'/home/leoinferos/Endurosat/forest1.jpg', 'rb') as f:
    image_bytes = f.read()

prediction = predict(image_bytes)
print(f"Prediction: {prediction}")  # 1.0 for forest, 0.0 for not forest
