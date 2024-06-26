import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
# import to use batched up preprocessed tensors
from SelectAdditiveLearning.pretrainModel.mosei_to_tensor import preprocess_mosi, tensor_fusion 
from tensor_fusion import TFN

epsilon = 1e-7
seed = 0

# Define activation functions
def tanh(x):
    return torch.tanh(x)

def relu(x):
    return torch.maximum(epsilon + torch.zeros_like(x), x)

def linear(x):
    return x

def sigmoid(x):
    return torch.sigmoid(x)

# Define LeNetConvPoolLayer class
class LeNetConvPoolLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, pool_size, activation):
        super(LeNetConvPoolLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size)
        self.pool = nn.MaxPool2d(pool_size)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.activation(x)
        return x

class LeNetConvPoolLayer(nn.Module):
    def __init__(self, input_channels, activation):
        super(LeNetConvPoolLayer, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.activation = activation
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 64 * (x.size(2) // 2))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define DropoutHiddenLayer class
class DropoutHiddenLayer(nn.Module):
    def __init__(self, input_size, output_size, activation, dropout_rate):
        super(DropoutHiddenLayer, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x

def evaluate_lenet5(learning_rate=1e-7, n_epochs=200, batch_size=50, l1=0., l2=0.0, dropout=0.0):
    torch.manual_seed(seed)

    # Load data
    # last two is not used in this pretraining phase
    batch_sz = 32
    train_set, valid_set, test_set = preprocess_mosi()

    train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_sz, collate_fn=tensor_fusion)
    valid_loader = DataLoader(valid_set, shuffle=False, batch_size=batch_sz*3, collate_fn=tensor_fusion)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_sz*3, collate_fn=tensor_fusion)

    # Initialize model
    model = nn.Sequential(
        LeNetConvPoolLayer(1, 25, (1, 5), (1, 4), tanh),
        LeNetConvPoolLayer(25, 50, (25, 1), (1, 5), tanh),
        DropoutHiddenLayer(50 * 1 * 9, 80, tanh, dropout),
        LogisticRegression(80, 2)
    )

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(n_epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                valid_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_loss /= len(valid_loader)
        accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{n_epochs}], Valid Loss: {valid_loss:.4f}, Valid Acc: {100 * accuracy:.2f}%')

if __name__ == '__main__':
    evaluate_lenet5(learning_rate=5e-4, n_epochs=5000, batch_size=15, l1=0.0, l2=0., dropout=0)