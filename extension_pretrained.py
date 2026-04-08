# Sushma Ramesh & Dina Barua
# CS 5330 - Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
# Extension: Analyzing a Pre-trained Network (ResNet18)

# The assignment says to load a pre-trained network from torchvision and
# examine its first couple of convolutional layers, similar to what Task 2
# did for our MNIST network.

# We use ResNet18, which was trained on ImageNet (1.2 million images, 1000 classes).
# Its filters have learned to detect real-world features like edges, colors,
# and textures - very different from our MNIST network which only ever saw
# handwritten digits.

# What this file does:
#   1. Load ResNet18 with pretrained ImageNet weights and print the model
#   2. Visualize the 64 filters from the first conv layer as color (RGB) images
#   3. Apply those filters to a test image and show the 64 resulting feature maps
#   4. Look at the second conv layer and compare filter shapes


import sys
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# load_pretrained_model loads ResNet18 with weights trained on ImageNet.
# ResNet18 is a classic 18-layer residual network - big enough to learn
# rich features but small enough to download and run quickly.
#
# returns the model set to eval mode (no dropout, no training behavior)
def load_pretrained_model():
    print("Loading pretrained ResNet18...")

    # weights='DEFAULT' loads the best available pretrained weights.
    # this will download ~45 MB on the first run, then cache them locally.
    # for older versions of torchvision you can use pretrained=True instead.
    model = models.resnet18(weights='DEFAULT')

    # set to eval mode since we're only inspecting the model, not training it
    model.eval()

    print("Model loaded.\n")
    print("Full model structure:")
    print(model)
    return model


# get_test_image loads one image from the MNIST test set and converts it
# to a 3-channel RGB image that ResNet18 can accept.
#
# ResNet18 was trained on color (RGB) images, not grayscale, so we need 3 channels.
# We just copy the grayscale digit across all three channels - it looks gray
# but the tensor shape is correct: [1, 3, 224, 224].
# We also resize to 224x224 since that's the standard ImageNet input size.
#
# returns a tensor of shape [1, 3, 224, 224]
def get_test_image():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # ResNet expects 224x224 inputs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean per channel
                             std=[0.229, 0.224, 0.225])   # ImageNet std per channel
    ])

    # load one MNIST test image (grayscale, 28x28)
    mnist = torchvision.datasets.MNIST(
        root='./data', train=False, download=True,
        transform=transforms.ToTensor()
    )
    img_gray, label = mnist[0]  # shape [1, 28, 28]

    # resize to 224x224 using interpolation, then repeat across 3 channels
    img_resized = transforms.Resize((224, 224))(img_gray)  # [1, 224, 224]
    img_rgb = img_resized.repeat(3, 1, 1)                   # [3, 224, 224]

    # normalize with ImageNet stats
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_rgb = normalize(img_rgb)

    print(f"Test image: MNIST digit '{label}', resized to 224x224, converted to 3-channel")
    return img_rgb.unsqueeze(0), label  # add batch dim -> [1, 3, 224, 224]


# visualize_first_layer_filters pulls out the 64 filters from ResNet18's
# first conv layer and displays them in an 8x8 grid.
#
# Each filter has shape [3, 7, 7] - three 7x7 grids, one per RGB channel.
# To display them as color images we stack the channels and normalize
# each filter's values to the 0-1 range so they look like proper images.
#
# This is different from Task 2 where our MNIST filters were grayscale [1, 5, 5].
# Real-world networks learn color-sensitive filters because they see RGB images.
def visualize_first_layer_filters(model, save_path="extension_filters_layer1.png"):
    # model.conv1.weight has shape [64, 3, 7, 7]
    # 64 filters, 3 input channels (RGB), 7x7 pixels each
    weights = model.conv1.weight.data
    print(f"\nconv1 filter shape: {weights.shape}")
    print(f"  {weights.shape[0]} filters, {weights.shape[1]} input channels, "
          f"{weights.shape[2]}x{weights.shape[3]} pixels each")

    # we need to detach from the computation graph to convert to numpy
    filters = weights.detach().cpu().numpy()  # [64, 3, 7, 7]

    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.suptitle("ResNet18 - Conv1 Filters (64 total, shown as RGB)", fontsize=12)

    for i in range(64):
        ax = axes[i // 8][i % 8]

        # grab this filter: shape [3, 7, 7]
        f = filters[i]

        # move channel dim to the back so it's [7, 7, 3] - what imshow expects
        f = np.transpose(f, (1, 2, 0))

        # normalize to [0, 1] so imshow displays it properly
        # each filter has different value ranges so we normalize individually
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)

        ax.imshow(f)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Layer 1 filters saved to: {save_path}")
    plt.close()


# apply_first_layer and visualize the feature maps.
#
# we pass the test image through just the first conv layer and show
# what each of the 64 filters responds to. brighter areas = stronger response.
#
# this is the same idea as Task 2's filter2D visualization, but we use
# PyTorch directly instead of OpenCV because we have a 3-channel input.
def visualize_first_layer_outputs(model, img_tensor, save_path="extension_outputs_layer1.png"):
    print("\nApplying conv1 to the test image...")

    # torch.no_grad() means we don't build a gradient graph - we're just
    # doing a forward pass for visualization, not training
    with torch.no_grad():
        # run the image through only the first conv layer
        # nn.functional.conv2d lets us apply the weights directly
        output = nn.functional.conv2d(
            img_tensor,
            model.conv1.weight,
            bias=model.conv1.bias,
            stride=model.conv1.stride,
            padding=model.conv1.padding
        )
        # output shape: [1, 64, 112, 112] - 64 feature maps, each 112x112

    print(f"Feature map shape after conv1: {output.shape}")
    # [batch, channels, height, width] = [1, 64, 112, 112]

    feature_maps = output.squeeze(0).detach().cpu().numpy()  # [64, 112, 112]

    fig, axes = plt.subplots(8, 8, figsize=(14, 14))
    fig.suptitle("ResNet18 - Conv1 Feature Maps on MNIST digit (64 total)", fontsize=12)

    for i in range(64):
        ax = axes[i // 8][i % 8]
        fmap = feature_maps[i]

        # normalize each feature map to [0, 1] for display
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

        ax.imshow(fmap, cmap='viridis')  # viridis colormap shows intensity clearly
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Layer 1 feature maps saved to: {save_path}")
    plt.close()


# examine_second_layer looks at the shape and basic stats of the second
# conv layer's filters. ResNet18's second conv is inside the first residual
# block: layer1[0].conv1, shape [64, 64, 3, 3].
#
# 64 filters with 64 input channels is too many to visualize directly as
# color images, so we just print the shape and show a few selected filters
# as a grid of their individual channel slices.
def examine_second_layer(model, save_path="extension_filters_layer2.png"):
    # layer1[0].conv1 is the first conv inside the first residual block
    weights = model.layer1[0].conv1.weight.data
    print(f"\nlayer1[0].conv1 filter shape: {weights.shape}")
    print(f"  {weights.shape[0]} filters, {weights.shape[1]} input channels, "
          f"{weights.shape[2]}x{weights.shape[3]} pixels each")
    print(f"  Total parameters in this layer: {weights.numel():,}")

    # visualize a 6x8 grid of 48 individual filter slices.
    # each cell shows one [3, 3] slice from one filter's first input channel.
    # this gives a rough sense of what patterns the second layer looks for.
    filters = weights.detach().cpu().numpy()  # [64, 64, 3, 3]

    num_show = 48  # just show the first 48 to keep the grid manageable
    fig, axes = plt.subplots(6, 8, figsize=(10, 8))
    fig.suptitle("ResNet18 - Layer 2 Filters (first channel slice, 48 of 64)", fontsize=11)

    for i in range(num_show):
        ax = axes[i // 8][i % 8]
        f = filters[i, 0]  # take the first input channel slice: shape [3, 3]

        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Layer 2 filter slices saved to: {save_path}")
    plt.close()


# print_layer_summary gives a quick comparison of the first two conv layers
# so it's easy to see how the network grows in depth and complexity.
def print_layer_summary(model):
    print("\n" + "=" * 55)
    print("  Layer comparison: conv1 vs layer1[0].conv1")
    print("=" * 55)

    w1 = model.conv1.weight.data
    w2 = model.layer1[0].conv1.weight.data

    print(f"  conv1            : {list(w1.shape)}  "
          f"-> {w1.shape[0]} filters of size {w1.shape[2]}x{w1.shape[3]}, "
          f"input channels: {w1.shape[1]}")
    print(f"  layer1[0].conv1  : {list(w2.shape)}  "
          f"-> {w2.shape[0]} filters of size {w2.shape[2]}x{w2.shape[3]}, "
          f"input channels: {w2.shape[1]}")
    print()
    print("  Notice:")
    print("  - conv1 uses 7x7 filters (large, to capture broad structure)")
    print("  - layer1[0].conv1 uses 3x3 filters (small, to capture fine detail)")
    print("  - layer1[0].conv1 has 64 input channels instead of 3")
    print("    because it receives the 64 feature maps from conv1, not raw pixels")
    print("=" * 55)


# main loads the model, grabs a test image, and runs all the visualizations.
def main(argv):
    print("=" * 55)
    print("  Extension - Pre-trained ResNet18 Filter Analysis")
    print("=" * 55 + "\n")

    # load pretrained resnet18 and print the full structure
    model = load_pretrained_model()

    # get a test image to apply the filters to
    img_tensor, label = get_test_image()

    # print a side-by-side summary of the first two conv layers
    print_layer_summary(model)

    # visualize the 64 first-layer filters as RGB color patches
    visualize_first_layer_filters(model)

    # apply conv1 to the test image and show the 64 feature maps
    visualize_first_layer_outputs(model, img_tensor)

    # look at the second conv layer filter shapes and visualize slices
    examine_second_layer(model)

    print("\nDone! Check the saved PNG files for the visualizations.")
    print("Things to look for in your report:")
    print("  - conv1 filters often look like colored edge/orientation detectors")
    print("  - compare them to the grayscale MNIST filters from Task 2")
    print("  - feature maps show which parts of the digit each filter responds to")
    return


if __name__ == "__main__":
    main(sys.argv)