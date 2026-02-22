"""
FedFortress Demo — Client Training Script
==========================================
This is a sample federated learning client training script.
It simulates a hospital client training on patient data.

Upload this file to the FedFortress Diagnostic Analyzer to see
how common (unintentional) mistakes cause a client to be flagged.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np

# ─── HYPERPARAMETERS ─────────────────────────────────────────────
# ⚠ Common mistake: learning rate is too high for federated settings
lr = 0.5
batch_size = 8        # ⚠ Too small — causes gradient variance

# ⚠ Too many local epochs — causes client drift from global model
local_epochs = 10

NUM_CLASSES = 2       # Binary: disease / no-disease
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── SIMPLE MODEL ────────────────────────────────────────────────
class DiagnosticNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ─── DATASET ─────────────────────────────────────────────────────
class PatientDataset(Dataset):
    def __init__(self, csv_path):
        import csv
        self.samples = []
        self.labels = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                features = [
                    float(row['age']),
                    float(row['bmi']),
                    float(row['glucose']),
                    float(row['blood_pressure']),
                    float(row['cholesterol']),
                ]
                # ⚠ No normalization! Raw values fed directly to model
                self.samples.append(torch.tensor(features, dtype=torch.float32))
                self.labels.append(int(row['diagnosis']))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


# ─── TRAINING FUNCTION ───────────────────────────────────────────
def train_client(model, dataloader, optimizer, criterion):
    model.train()
    for epoch in range(local_epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()

            # ⚠ No gradient clipping! Large updates go straight to optimizer
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{local_epochs} — Loss: {avg_loss:.4f}")

    return model.state_dict()


# ─── MAIN ────────────────────────────────────────────────────────
def main():
    # Load dataset
    dataset = PatientDataset('demo_dataset.csv')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model + optimizer
    model = DiagnosticNet(input_dim=5, num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Training on {len(dataset)} samples | LR={lr} | Epochs={local_epochs}")
    updated_weights = train_client(model, dataloader, optimizer, criterion)
    print("Training complete. Sending weights to federation server...")

    return updated_weights


if __name__ == '__main__':
    main()
