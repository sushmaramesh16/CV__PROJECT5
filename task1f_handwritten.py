# Sushma Ramesh - CS 5330 Project 5
# Task 1F: Test network on handwritten digit images

import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms

from task1_train import MyNetwork


def load_model(path='mnist_model.pth'):
    """Load trained MNIST model and set to eval mode."""
    model = MyNetwork()
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def load_and_preprocess(image_path):
    """Load image, convert to grayscale, resize to 28x28, match MNIST format."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    if np.mean(resized) > 127:
        resized = cv2.bitwise_not(resized)
    return resized


def run_handwritten(model, my_digits_path='my_digits'):
    """Run model on all handwritten digit images and plot results."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    images = []
    predictions = []
    true_labels = []

    for digit in range(10):
        path = os.path.join(my_digits_path, f'digit_{digit}.png')
        img = load_and_preprocess(path)
        images.append(img)
        img_tensor = transform(img).unsqueeze(0).float()
        with torch.no_grad():
            output = model(img_tensor)
        vals = output[0].numpy()
        pred = int(np.argmax(vals))
        predictions.append(pred)
        true_labels.append(digit)
        print(f"Digit {digit}: Predicted {pred} {'✓' if pred == digit else '✗'}")

    correct = sum(p == l for p, l in zip(predictions, true_labels))
    print(f"\nAccuracy: {correct}/10")

    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle("Handwritten Digit Predictions", fontsize=14)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap='gray')
        color = 'green' if predictions[i] == true_labels[i] else 'red'
        ax.set_title(f"True: {true_labels[i]}\nPred: {predictions[i]}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('handwritten_predictions.png', dpi=150)
    plt.show()
    print("Saved handwritten_predictions.png")


def main(argv):
    print("=== CS 5330 Project 5 - Task 1F: Handwritten Digit Test ===\n")
    model = load_model('mnist_model.pth')
    run_handwritten(model, 'my_digits')
    print("\nDone!")
    return

if __name__ == "__main__":
    main(sys.argv)
