import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Step 1: Define a Toy CNN Model
class ToyCNN(nn.Module):
    def __init__(self):
        super(ToyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(
            8 * 16 * 16, 10
        )  # Assuming input image size is (1, 32, 32)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 8 * 16 * 16)  # Flatten for the fully connected layer
        x = self.fc1(x)
        return x


# Step 2: Create Random Data for a Toy Dataset
def create_toy_dataset(batch_size=32):
    inputs = torch.randn(100, 1, 32, 32)  # 100 random images of size (1, 32, 32)
    labels = torch.randint(0, 10, (100,))  # Random labels (10 classes)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Step 3: Initialize Model, Loss, Optimizer
model = ToyCNN().to(device)  # Move model to GPU (if available)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Step 4: Train the Model for 1 Epoch
def train_one_epoch(model, dataloader):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU

        optimizer.zero_grad()  # Reset gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()

    print(f"Training loss: {running_loss / len(dataloader)}")


if __name__ == "__main__":
    dataloader = create_toy_dataset()
    train_one_epoch(model, dataloader)
    print("Training complete!")
