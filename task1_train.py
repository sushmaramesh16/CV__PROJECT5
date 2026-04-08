# Sushma Ramesh - CS 5330 Project 5
# Task 1: Build, train, and save a CNN for MNIST digit recognition

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Network class definition
class MyNetwork(nn.Module):
    """CNN for MNIST digit recognition.
    Architecture:
      Conv(10, 5x5) -> MaxPool(2x2) + ReLU
      Conv(20, 5x5) -> Dropout(0.5) -> MaxPool(2x2) + ReLU
      Flatten -> Linear(50) + ReLU
      Linear(10) -> log_softmax
    """
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass through the network
    def forward(self, x):
        # Conv1 -> MaxPool -> ReLU
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Conv2 -> Dropout -> MaxPool -> ReLU
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Flatten
        x = x.view(-1, 320)
        # FC1 -> ReLU
        x = F.relu(self.fc1(x))
        # FC2 -> log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x


def get_data_loaders(batch_size_train=64, batch_size_test=1000):
    """Load MNIST train and test sets; test set is NOT shuffled."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size_train, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size_test, shuffle=False
    )

    return train_loader, test_loader


def plot_first_six_test(test_loader):
    """Plot the first six examples from the test set."""
    examples = enumerate(test_loader)
    _, (example_data, example_targets) = next(examples)

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    fig.suptitle("First 6 Test Set Examples (MNIST)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(example_data[i][0], cmap='gray', interpolation='none')
        ax.set_title(f"Label: {example_targets[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('first_six_test.png', dpi=150)
    plt.show()
    print("Saved first_six_test.png")


def train_network(model, optimizer, train_loader, epoch, train_losses, train_counter):
    """Train the model for one epoch, recording loss."""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"  Train Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}"
                  f" ({100. * batch_idx / len(train_loader):.0f}%)]  Loss: {loss.item():.6f}")
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
            )


def test_network(model, test_loader, test_losses, test_accuracies):
    """Evaluate the model on the full test set."""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_losses.append(test_loss)
    test_accuracies.append(accuracy)
    print(f"  Test set: Avg loss: {test_loss:.4f}, "
          f"Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return accuracy


def plot_training_curves(train_counter, train_losses, test_accuracies, n_epochs):
    """Plot training loss and test accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_counter, train_losses, color='blue', alpha=0.7, label='Train Loss')
    ax1.set_xlabel('Number of training examples seen')
    ax1.set_ylabel('Negative log likelihood loss')
    ax1.set_title('Training Loss')
    ax1.legend()

    epochs = list(range(1, n_epochs + 1))
    ax2.plot(epochs, test_accuracies, color='red', marker='o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Test Accuracy per Epoch')
    ax2.set_ylim([95, 100])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("Saved training_curves.png")


def save_model(model, path='mnist_model.pth'):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


# main function: orchestrates data loading, training, evaluation, and saving
def main(argv):
    n_epochs = 5
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    print("=== CS 5330 Project 5 - Task 1: Train MNIST CNN ===\n")

    train_loader, test_loader = get_data_loaders(batch_size_train, batch_size_test)

    plot_first_six_test(test_loader)

    model = MyNetwork()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    print(f"\nModel architecture:\n{model}\n")
    print(f"Training for {n_epochs} epochs...\n")

    train_losses = []
    train_counter = []
    test_losses = []
    test_accuracies = []

    print("Before training:")
    test_network(model, test_loader, test_losses, test_accuracies)

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}:")
        train_network(model, optimizer, train_loader, epoch, train_losses, train_counter)
        test_network(model, test_loader, test_losses, test_accuracies)

    plot_training_curves(train_counter, train_losses, test_accuracies[1:], n_epochs)

    save_model(model, 'mnist_model.pth')

    print("\nDone! Model saved as mnist_model.pth")
    return


if __name__ == "__main__":
    main(sys.argv)