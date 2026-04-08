# Sushma Ramesh - CS 5330 Project 5
# Task 2: Examine the network - analyze filters and their effects

# import statements
import sys
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2

# import the network class from task 1
from task1_train import MyNetwork


def load_model(path='mnist_model.pth'):
    """Load the trained model from file."""
    model = MyNetwork()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    print("Model loaded successfully.\n")
    print("Model structure:")
    print(model)
    return model


def visualize_conv1_filters(model):
    """Get and visualize the 10 filters from the first conv layer."""
    weights = model.conv1.weight  # shape: [10, 1, 5, 5]
    print(f"\nconv1 weight shape: {weights.shape}")

    # print each filter's weights
    for i in range(10):
        filter_weights = weights[i, 0]
        print(f"\nFilter {i} shape: {filter_weights.shape}")
        print(np.round(filter_weights.detach().numpy(), 4))

    # plot 10 filters in a 3x4 grid (last 2 spots empty)
    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    fig.suptitle("Conv1 Filter Weights (10 filters, 5x5)", fontsize=14)
    for i in range(12):
        ax = axes[i // 4][i % 4]
        if i < 10:
            filt = weights[i, 0].detach().numpy()
            ax.imshow(filt, cmap='viridis')
            ax.set_title(f"Filter {i}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig('conv1_filters.png', dpi=150)
    plt.show()
    print("Saved conv1_filters.png")


def show_filter_effects(model):
    """Apply the 10 conv1 filters to the first training image using OpenCV filter2D."""
    # load first training example
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST('./data', train=True, download=False, transform=transform)
    first_image = train_data[0][0][0].numpy()  # shape: [28, 28]

    weights = model.conv1.weight  # [10, 1, 5, 5]

    # plot original + 10 filtered images in a 5x4 grid (filter, result pairs)
    fig, axes = plt.subplots(5, 4, figsize=(10, 12))
    fig.suptitle("Conv1 Filter Effects on First Training Image", fontsize=13)

    with torch.no_grad():
        for i in range(10):
            filt = weights[i, 0].detach().numpy()  # 5x5 filter
            filtered = cv2.filter2D(first_image, -1, filt)

            row = i // 2
            col = (i % 2) * 2  # columns 0,2 = filter; 1,3 = result

            # show filter
            axes[row][col].imshow(filt, cmap='gray')
            axes[row][col].set_title(f"Filter {i}")
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

            # show filtered result
            axes[row][col + 1].imshow(filtered, cmap='gray')
            axes[row][col + 1].set_title(f"Result {i}")
            axes[row][col + 1].set_xticks([])
            axes[row][col + 1].set_yticks([])

    plt.tight_layout()
    plt.savefig('conv1_filterResults.png', dpi=150)
    plt.show()
    print("Saved conv1_filterResults.png")


# main function: loads model, prints structure, visualizes filters and effects
def main(argv):
    print("=== CS 5330 Project 5 - Task 2: Examine Network ===\n")

    model = load_model('mnist_model.pth')
    visualize_conv1_filters(model)
    show_filter_effects(model)

    print("\nDone!")
    return


if __name__ == "__main__":
    main(sys.argv)