import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.simple_model import SimpleCNN

def train_baseline(epochs=5):
    """
    Centralized baseline training for CIFAR-10.
    Yields results after each epoch for live display.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                           (0.2470, 0.2435, 0.2616))
    ])

    # Use subset for faster training
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )
    
    # Use 5000 samples for baseline
    from torch.utils.data import Subset
    train_dataset = Subset(train_dataset, range(5000))

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )
    test_dataset = Subset(test_dataset, range(1000))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("Starting Centralized Training...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # Yield results for live display
        yield {
            "epoch": epoch + 1,
            "loss": avg_loss,
            "accuracy": train_accuracy
        }

    # Final Evaluation
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    yield {
        "epoch": epochs,
        "loss": avg_loss,
        "accuracy": test_accuracy,
        "test_accuracy": test_accuracy
    }


def run_centralized_training(epochs=3):
    """Run centralized training and return final accuracy."""
    results = list(train_baseline(epochs))
    return results[-1].get('accuracy', 0) if results else 0
