import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ----- 1. Custom Dataset class -----
class FingerprintDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ----- 2. Simple CNN model -----
class FingerprintClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(FingerprintClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))      # Downsample
        x = self.pool(F.relu(self.conv2(x)))      # Downsample
        x = self.adaptive_pool(x)                 # Output shape: (B, 32, 4, 4)
        x = self.dropout(x)
        x = torch.flatten(x, 1)                   # Shape: (B, 512)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ----- 3. Training loop -----
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# ----- 4. Evaluation function -----
def evaluate_model(model, dataloader, device, get_embeddings=False):
    model.eval()
    correct, total = 0, 0

    embeddings = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if get_embeddings:
                embs = model.get_embedding(inputs)  # shape: (B, D)
                embeddings.extend(embs.detach().cpu())  # append each embedding tensor to the list

    if get_embeddings:
        return embeddings, labels  # both are lists of tensors
    else:
        return correct / total
    