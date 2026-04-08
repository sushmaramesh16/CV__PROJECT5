# Sushma Ramesh - CS 5330 Project 5
# Task 3: Transfer Learning on Greek Letters

# import statements
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# import network from task 1
from task1_train import MyNetwork


# custom transform for greek letter images (133x133 color -> 28x28 grayscale inverted)
class GreekTransform:
    def __init__(self):
        pass

    # converts color greek letter image to match MNIST format
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def load_greek_data(training_set_path, batch_size=5):
    """Load greek letter dataset using ImageFolder and GreekTransform."""
    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(
            training_set_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
            ])
        ),
        batch_size=batch_size,
        shuffle=True
    )
    return greek_train


def build_greek_model(mnist_model_path='mnist_model.pth'):
    """Load pretrained MNIST model, freeze weights, replace last layer with 3-node output."""
    model = MyNetwork()
    model.load_state_dict(torch.load(mnist_model_path, weights_only=True))

    # freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # replace last layer: 50 -> 3 (alpha, beta, gamma)
    model.fc2 = nn.Linear(50, 3)

    print("Modified network structure:")
    print(model)
    return model


def train_greek(model, optimizer, greek_train, epoch, train_losses, train_counter):
    """Train the modified model for one epoch on greek data."""
    model.train()
    for batch_idx, (data, target) in enumerate(greek_train):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        train_counter.append(batch_idx + (epoch - 1) * len(greek_train))

    print(f"  Epoch {epoch}: Loss: {train_losses[-1]:.4f}")


def evaluate_greek(model, greek_train):
    """Evaluate accuracy on the greek training set."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in greek_train:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += len(target)
    accuracy = 100. * correct / total
    print(f"  Accuracy: {correct}/{total} ({accuracy:.1f}%)")
    return accuracy


def plot_greek_loss(train_losses, train_counter):
    """Plot training loss curve for greek letter training."""
    plt.figure(figsize=(8, 4))
    plt.plot(train_counter, train_losses, color='purple', alpha=0.8, label='Train Loss')
    plt.xlabel('Batch iterations')
    plt.ylabel('Negative log likelihood loss')
    plt.title('Greek Letter Transfer Learning - Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('greek_training_loss.png', dpi=150)
    plt.show()
    print("Saved greek_training_loss.png")


# main function: loads pretrained model, fine-tunes on greek letters, evaluates
def main(argv):
    training_set_path = 'greek_train'
    n_epochs = 30
    learning_rate = 0.01
    momentum = 0.5

    print("=== CS 5330 Project 5 - Task 3: Transfer Learning on Greek Letters ===\n")

    # load data
    greek_train = load_greek_data(training_set_path)
    print(f"Classes: {greek_train.dataset.classes}\n")

    # build modified model
    model = build_greek_model('mnist_model.pth')

    # only train the new last layer
    optimizer = optim.SGD(model.fc2.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []

    # train for n_epochs
    for epoch in range(1, n_epochs + 1):
        train_greek(model, optimizer, greek_train, epoch, train_losses, train_counter)

    # evaluate
    print("\nFinal evaluation on training set:")
    evaluate_greek(model, greek_train)

    # plot loss
    plot_greek_loss(train_losses, train_counter)

    # save greek model
    torch.save(model.state_dict(), 'greek_model.pth')
    print("Saved greek_model.pth")

    print("\nDone!")
    return


if __name__ == "__main__":
    main(sys.argv)