# Sushma Ramesh - CS 5330 Project 5
# Task 1E: Read network and run on first 10 test examples

# import statements
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# import network class from task 1
from task1_train import MyNetwork


def load_model(path='mnist_model.pth'):
    """Load trained MNIST model from file and set to eval mode."""
    model = MyNetwork()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def get_test_examples(n=10):
    """Load first n examples from the MNIST test set (no shuffle)."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_data = datasets.MNIST('./data', train=False, download=False, transform=transform)
    images = []
    labels = []
    for i in range(n):
        img, label = test_data[i]
        images.append(img)
        labels.append(label)
    return images, labels


def run_on_test_examples(model, images, labels):
    """Run model on first 10 test examples, print outputs and predictions."""
    predictions = []
    print(f"{'Idx':<5} {'Output Values (10)':<60} {'Pred':>5} {'Label':>6}")
    print("-" * 78)
    for i, (img, label) in enumerate(zip(images, labels)):
        with torch.no_grad():
            output = model(img.unsqueeze(0))  # add batch dim
        vals = output[0].numpy()
        pred = int(np.argmax(vals))
        predictions.append(pred)
        vals_str = "  ".join([f"{v:.2f}" for v in vals])
        print(f"{i:<5} {vals_str:<60} {pred:>5} {label:>6}")
    return predictions


def plot_3x3_grid(images, labels, predictions):
    """Plot first 9 test digits in a 3x3 grid with predictions as titles."""
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    fig.suptitle("First 9 Test Set Predictions", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i][0], cmap='gray', interpolation='none')
        color = 'green' if predictions[i] == labels[i] else 'red'
        ax.set_title(f"Pred: {predictions[i]}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('test_predictions.png', dpi=150)
    plt.show()
    print("Saved test_predictions.png")


# main function: loads model, runs on first 10 test examples, plots 3x3 grid
def main(argv):
    print("=== CS 5330 Project 5 - Task 1E: Run Model on Test Examples ===\n")

    model = load_model('mnist_model.pth')
    images, labels = get_test_examples(10)
    predictions = run_on_test_examples(model, images, labels)

    print(f"\nCorrect: {sum(p==l for p,l in zip(predictions,labels))}/10")
    plot_3x3_grid(images, labels, predictions)

    print("\nDone!")
    return


if __name__ == "__main__":
    main(sys.argv)