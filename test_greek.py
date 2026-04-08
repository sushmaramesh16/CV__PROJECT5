# Sushma Ramesh & Dina Barua
# CS 5330 - Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
# Task 3: Test greek model on custom alpha, beta, gamma images
# Task 3: Test greek model on custom alpha, beta, gamma images
# Generates multiple examples per letter with varied sizes and tests them

import sys
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from PIL import Image as PILImage

from task1_train import MyNetwork


class GreekTransform:
    """Transform custom greek images to match MNIST format."""
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def generate_greek_images(output_dir='my_greek'):
    """Generate 3 examples per greek letter at different font sizes."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    symbols = {
        'alpha': r'$\alpha$',
        'beta':  r'$\beta$',
        'gamma': r'$\gamma$'
    }
    sizes = [60, 80, 100]

    for name, symbol in symbols.items():
        for j, size in enumerate(sizes):
            fig, ax = plt.subplots(figsize=(1.5, 1.5))
            ax.text(0.5, 0.5, symbol, fontsize=size,
                    ha='center', va='center',
                    transform=ax.transAxes, color='black')
            ax.set_facecolor('white')
            ax.axis('off')
            plt.tight_layout(pad=0)
            path = f'{output_dir}/{name}_{j+1}.png'
            plt.savefig(path, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
    print("Generated 9 greek letter images (3 per letter)\n")


def tight_crop(img_gray):
    """Crop tightly around the symbol to remove whitespace."""
    _, thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img_gray
    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(img_gray.shape[1], x + w + pad)
    y2 = min(img_gray.shape[0], y + h + pad)
    # make it square
    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(img_gray.shape[1], x1 + side)
    y2 = min(img_gray.shape[0], y1 + side)
    return img_gray[y1:y2, x1:x2]


def load_greek_model(path='greek_model.pth'):
    """Load trained greek letter model."""
    model = MyNetwork()
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, 3)
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    return model


def test_custom_greek(model, input_dir='my_greek'):
    """Run model on all custom greek images and plot results."""
    class_names = ['alpha', 'beta', 'gamma']
    symbols = ['alpha', 'beta', 'gamma']
    true_label_map = {'alpha': 0, 'beta': 1, 'gamma': 2}

    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    images, predictions, true_labels, titles = [], [], [], []
    correct = 0
    total = 0

    for name in symbols:
        for j in range(1, 4):
            path = f'{input_dir}/{name}_{j}.png'
            img_bgr = cv2.imread(path)
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # tight crop before passing to transform
            cropped = tight_crop(img_gray)
            resized = cv2.resize(cropped, (133, 133))
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            pil_img = PILImage.fromarray(img_rgb)

            tensor = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                output = model(tensor)
            pred = int(output[0].numpy().argmax())

            true = true_label_map[name]
            predictions.append(pred)
            true_labels.append(true)
            images.append(cropped)
            titles.append((name, class_names[pred]))

            match = pred == true
            correct += int(match)
            total += 1
            print(f'{name}_{j}: Predicted {class_names[pred]} '
                  f'{"✓" if match else "✗"}')

    print(f'\nAccuracy: {correct}/{total}')

    # plot 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    fig.suptitle('Custom Greek Letter Predictions', fontsize=14)
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx], cmap='gray')
        true_name, pred_name = titles[idx]
        color = 'green' if true_name == pred_name else 'red'
        ax.set_title(f'True: {true_name}\nPred: {pred_name}', color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('greek_custom_results.png', dpi=150)
    plt.show()
    print('Saved greek_custom_results.png')


def main(argv):
    print('=== Task 3: Custom Greek Letter Test ===\n')
    generate_greek_images('my_greek')
    model = load_greek_model('greek_model.pth')
    test_custom_greek(model, 'my_greek')
    return


if __name__ == "__main__":
    main(sys.argv)
