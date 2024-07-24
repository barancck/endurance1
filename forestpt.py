import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn, optim
from PIL import Image
from huggingface_hub import HfFileSystem
import io

# Initialize HfFileSystem
fs = HfFileSystem()

# Define paths to your data
splits = {
    'train': 'data/train-00000-of-00001-8a3b67cb35e5c8a7.parquet',
    'test': 'data/test-00000-of-00001-ed08959539202db4.parquet'
}
train_path = "hf://datasets/himanshusrivastava/ucm_satellite_images/" + splits["train"]
test_path = "hf://datasets/himanshusrivastava/ucm_satellite_images/" + splits["test"]

# Read the parquet files
df_train = pd.read_parquet(fs.open(train_path))
df_test = pd.read_parquet(fs.open(test_path))

# Filter the DataFrame for forest images
forest_train_df = df_train[df_train['label'] == 7]
forest_test_df = df_test[df_test['label'] == 7]

# Check if filtered DataFrame is empty
print(f"Train dataset size: {len(forest_train_df)}")
print(f"Test dataset size: {len(forest_test_df)}")

# Convert DataFrame to a list of image-label tuples
def parse_image_and_label(row):
    if isinstance(row['image'], dict) and 'bytes' in row['image']:
        image_bytes = row['image']['bytes']
    else:
        raise ValueError("Image data is not in expected byte format.")
    label = 1  # Set label to 1 for forest class
    return image_bytes, label

forest_train_tuples = forest_train_df.apply(parse_image_and_label, axis=1).tolist()
forest_test_tuples = forest_test_df.apply(parse_image_and_label, axis=1).tolist()

# Custom Dataset class for PyTorch
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_bytes, label = self.data[idx]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CustomDataset(forest_train_tuples, transform=transform)
test_dataset = CustomDataset(forest_test_tuples, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the CNN model
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

# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.BCELoss()  # Binary cross entropy for binary classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            labels = labels.unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluation function
def evaluate_model():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.unsqueeze(1)  # Ensure labels are of shape (batch_size, 1)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"\nTest Accuracy: {100 * correct / total:.2f}%")

# Function to get model summary
def model_summary():
    print(model)

# Function to predict using the model
def predict(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted = (output > 0.5).float()
    return predicted.item()

# Function to save the model
def save_model(path='model.pth'):
    torch.save(model.state_dict(), path)

# Function to load the model
def load_model(path='model.pth'):
    model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Example usage
    train_model(num_epochs=10)
    evaluate_model()
    model_summary()
    save_model()
