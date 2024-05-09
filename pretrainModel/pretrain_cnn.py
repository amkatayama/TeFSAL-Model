import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from SelectAdditiveLearning.pretrainModel.mosei_to_tensor import preprocess_mosi, multi_collate

class BaseCNN(nn.Module):
    def __init__(self, input_dim):
        super(BaseCNN, self).__init__()
        # Adjusting kernel sizes and removing problematic pooling
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=128, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1, padding=0)
        self.conv4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        self.conv5 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1, padding=0)
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1, padding=0)
        self.conv7 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1, padding=0)
        
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), 129, -1)  # Ensure [batch_size, channels, length]
        x = F.relu(self.conv1(x))
        # Removed problematic pooling layers
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))

        # Flatten the output for the fully connected layer
        x = torch.flatten(x, 1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

train, valid, test = preprocess_mosi()
batch_sz = 32

# Example usage
input_dim = 129  # This should match the output feature size of the Tensor Fusion Network
# num_classes = 2  # Adjust as per your classification problem (e.g., binary or multi-class)
model = BaseCNN(input_dim)

# Assume that 'train_dataset', 'val_dataset', and 'test_dataset' are PyTorch datasets
# containing your prepared data according to the structure you mentioned:
# Each dataset item is (text_features, visual_features, acoustic_features, labels, lengths)

train_loader = DataLoader(train, shuffle=True, batch_size=batch_sz, collate_fn=multi_collate, drop_last=True)
valid_loader = DataLoader(valid, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate, drop_last=True)
test_loader = DataLoader(test, shuffle=False, batch_size=batch_sz*3, collate_fn=multi_collate, drop_last=True)

# Initialize the model
# model = BaseCNN(input_dim=1, num_classes=2)  # Adjust 'input_dim' and 'num_classes' appropriately
if torch.cuda.is_available():
    model = model.cuda()

# Loss Function
# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for fused_input, labels, lengths in train_loader:
        # Assuming the output from the TFN is the 'text' tensor after processing
        labels = labels.float()
        fused_input, labels = fused_input.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(fused_input).squeeze() # Ensure outputs are properly squeezed
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

# Validation Function
def validate(model, device, valid_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for fused_input, labels, lengths in valid_loader:
            labels = labels.float()
            fused_input, labels = fused_input.to(device), labels.to(device)

            outputs = model(fused_input).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(valid_loader)

import math

# Function to calculate Root Mean Squared Error
def rmse(predictions, targets):
    return math.sqrt(((predictions - targets) ** 2).mean())

# Function to calculate Mean Absolute Error
def mae(predictions, targets):
    return (abs(predictions - targets)).mean()

# Testing Function
def test(model, device, test_loader):
    model.eval()
    total_mse = 0
    total_mae = 0
    count = 0
    with torch.no_grad():
        for fused_input, labels, lengths in test_loader:
            labels = labels.float()

            fused_input, labels = fused_input.to(device), labels.to(device)

            outputs = model(fused_input).squeeze()
            print(outputs)
            total_mse += ((outputs - labels) ** 2).sum().item()  # Sum up squared errors
            total_mae += abs(outputs - labels).sum().item()  # Sum up absolute errors
            count += labels.size(0)  # Total number of items

    # Calculate mean of the errors
    mean_mse = total_mse / count
    mean_rmse = math.sqrt(mean_mse)
    mean_mae = total_mae / count

    return mean_rmse, mean_mae

# Example usage of training and testing loops
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, device, train_loader, optimizer, criterion)
    val_loss = validate(model, device, valid_loader, criterion)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

# After training, evaluate on the test set using RMSE and MAE
test_rmse, test_mae = test(model, device, test_loader)
print(f'Test RMSE: {test_rmse:.4f}')
print(f'Test MAE: {test_mae:.4f}')