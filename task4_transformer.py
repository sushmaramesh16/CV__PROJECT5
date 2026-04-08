# Sushma Ramesh & Dina Barua
# CS 5330 - Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
# Task 4: Re-implement the Network using Transformer Layers

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')  # needed so plots save to a file instead of popping up a window
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------
# NetTransformer
# -----------------------------------------------------------------------
# This is the Vision Transformer model. It takes a 28x28 grayscale image
# and outputs 10 log-probabilities, one for each digit class (0-9).
#
# It inherits from nn.Module, which is the base class for all PyTorch
# models. This gives us gradient tracking and lets us call model(x) easily.
# -----------------------------------------------------------------------
class NetTransformer(nn.Module):

    # __init__ is where we define all the layers.
    # Nothing actually runs here - we are just setting up the pieces.
    #
    # img_size   - height/width of the image in pixels (28 for MNIST)
    # patch_size - size of each square patch we cut the image into
    #              e.g. patch_size=7 gives a 4x4 grid = 16 patches
    #              must divide img_size evenly (4, 7, and 14 all work for 28)
    # in_channels - 1 for grayscale, 3 for RGB
    # embed_dim  - how many numbers we use to represent each patch token
    #              bigger = richer representation, but slower to train
    # num_heads  - number of attention heads in the transformer
    #              must divide embed_dim evenly (e.g. embed_dim=64, num_heads=4)
    # num_layers - how many transformer blocks to stack on top of each other
    # mlp_dim    - hidden size used inside the transformer and the classifier
    # num_classes - number of output classes (10 for MNIST digits)
    # dropout    - randomly zeros out this fraction of values during training
    #              helps prevent overfitting (0.0 = off, typical = 0.1)
    def __init__(
        self,
        img_size=28,
        patch_size=7,
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        num_classes=10,
        dropout=0.1
    ):
        super(NetTransformer, self).__init__()
        # super().__init__() sets up the internal PyTorch bookkeeping
        # we need before we can start adding layers

        # sanity check - patch_size must divide the image evenly
        assert img_size % patch_size == 0, (
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        )

        self.patch_size = patch_size  # save this so forward() can use it

        # figure out how many patches we get and how big each one is
        # e.g. 28 // 7 = 4, so we get a 4x4 = 16 patch grid
        patches_per_side = img_size // patch_size
        num_patches = patches_per_side * patches_per_side
        patch_dim = in_channels * patch_size * patch_size  # pixels per patch (7*7*1 = 49)

        # --- Step 1: Patch Embedding ---
        # Each patch is just a flat list of pixel values right now.
        # This linear layer learns to turn those raw pixels into a more
        # useful embed_dim-sized vector (called a "token").
        # Think of it like: pixels are just numbers, but tokens carry meaning.
        self.patch_embedding = nn.Linear(patch_dim, embed_dim)

        # --- CLS Token ---
        # This is a special extra token we stick at the front of the sequence.
        # It doesn't come from any specific patch - it starts as random noise
        # and the transformer learns to pour a summary of the whole image into it.
        # After the transformer runs, we only look at this token to make our prediction.
        # nn.Parameter means it gets updated by the optimizer just like any weight.
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # --- Positional Embedding ---
        # The transformer has no idea where in the image each patch came from.
        # Without this, the top-left patch and bottom-right patch look identical to it.
        # We fix that by adding a learnable position vector to each token.
        # One position per patch, plus one for the CLS token = num_patches + 1 total.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        # dropout right after embedding, just for extra regularization
        self.embedding_dropout = nn.Dropout(dropout)

        # --- Step 2: Transformer Encoder ---
        # One encoder layer does two things:
        #   1. Self-attention: every token looks at every other token and decides
        #      what information to pull in (this is the key transformer operation)
        #   2. A small feedforward network with two linear layers and ReLU
        # Both sub-layers have residual connections and layer normalization built in.
        #
        # batch_first=True just means our tensors are shaped [batch, seq, features]
        # which is the more intuitive ordering
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            batch_first=True
        )
        # stack num_layers of these blocks on top of each other
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Steps 3 & 4: Classifier ---
        # After the transformer, we take the CLS token and pass it through
        # two linear layers to get our final 10 class scores.
        # We don't apply softmax here because we use log_softmax at the end of
        # forward(), which pairs with nll_loss to give us cross-entropy loss.
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )


    # forward() describes how data flows through the network.
    # PyTorch calls this automatically when you do output = model(x).
    #
    # x is a batch of images: shape [batch_size, 1, 28, 28]
    # returns log-probabilities: shape [batch_size, 10]
    def forward(self, x):
        B = x.shape[0]  # batch size (how many images we're processing at once)
        p = self.patch_size

        # --- Step 1a: Cut the image into patches ---
        # unfold slides a window of size p along the height (dim=2),
        # then along the width (dim=3), with stride=p so there's no overlap.
        # After both unfolds: [B, 1, n_h, n_w, p, p]
        # where n_h and n_w are how many patches fit along each side
        x = x.unfold(2, p, p).unfold(3, p, p)
        n_h, n_w = x.shape[2], x.shape[3]

        # flatten the patch grid into a list and flatten each patch's pixels
        # .contiguous() makes sure memory is laid out correctly before .view()
        # result: [B, num_patches, patch_dim]  e.g. [64, 16, 49]
        x = x.contiguous().view(B, 1, n_h * n_w, p * p)
        x = x.squeeze(1)  # drop the channel dimension

        # --- Step 1b: Embed patches into tokens ---
        # linear layer turns raw pixels into meaningful embed_dim-sized vectors
        # [B, num_patches, patch_dim] -> [B, num_patches, embed_dim]
        x = self.patch_embedding(x)

        # --- Step 1c: Prepend the CLS token ---
        # expand copies the CLS token once per image in the batch
        # (-1 means keep that dimension the same size)
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat([cls, x], dim=1)           # [B, num_patches+1, embed_dim]

        # --- Step 1d: Add positional info and apply dropout ---
        x = x + self.pos_embedding
        x = self.embedding_dropout(x)

        # --- Step 2: Run through the transformer encoder ---
        # each layer lets every token attend to every other token
        # shape stays [B, num_patches+1, embed_dim] throughout
        x = self.transformer_encoder(x)

        # --- Step 3: Pull out the CLS token ---
        # index 0 in the sequence is our CLS token.
        # by now it has gathered context from every patch via attention.
        cls_out = x[:, 0]  # [B, embed_dim]

        # --- Step 4: Classify ---
        # pass through the two-layer classifier to get 10 scores
        logits = self.classifier(cls_out)  # [B, 10]

        # log_softmax turns raw scores into log-probabilities across the 10 classes.
        # we use log_softmax (not regular softmax) because nll_loss expects log-probs.
        return nn.functional.log_softmax(logits, dim=1)


# train_epoch runs the model through all 60k training images once (one epoch).
# It updates the weights after every batch using backpropagation.
#
# model        - the network we're training
# device       - cpu or cuda
# train_loader - gives us batches of (images, labels)
# optimizer    - the algorithm that updates weights (we use Adam)
# epoch        - current epoch number, only used for the print statement
#
# returns the average training loss for this epoch
def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    # model.train() turns on dropout. PyTorch needs to know we're in training
    # mode because some layers behave differently during training vs. evaluation.

    total_loss = 0.0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # always clear gradients before a new batch - PyTorch adds them up by
        # default, so if we skip this we'd be mixing gradients from different batches
        optimizer.zero_grad()

        output = model(data)  # forward pass: get predictions

        # nll_loss (negative log-likelihood) measures how wrong our predictions are
        # lower = the model is more confident and correct
        loss = nn.functional.nll_loss(output, target)

        loss.backward()    # backward pass: figure out how each weight affected the loss
        optimizer.step()   # update weights: nudge them in the direction that lowers loss

        total_loss += loss.item()  # .item() pulls the number out of the tensor

    avg_loss = total_loss / len(train_loader)
    print(f"  Epoch {epoch} - Train Loss: {avg_loss:.4f}")
    return avg_loss


# evaluate runs the model over a full dataset (train or test) without updating weights.
# returns the average loss and accuracy so we can track progress after each epoch.
#
# we call this on BOTH the training set and the test set each epoch so we can
# tell if the model is actually learning or just memorizing the training data.
# (if train accuracy is high but test accuracy is low = overfitting)
#
# model      - the network
# device     - cpu or cuda
# loader     - DataLoader for whatever split we want to check
# split_name - label for the print statement, e.g. "Train" or "Test"
#
# returns avg_loss and accuracy (0 to 100)
def evaluate(model, device, loader, split_name="Test"):
    model.eval()
    # model.eval() turns off dropout so the same input always gives the same output.
    # this is important for getting consistent accuracy numbers.

    total_loss = 0.0
    correct = 0

    # torch.no_grad() tells PyTorch not to track gradients here.
    # we aren't calling .backward(), so there's no point - it just wastes memory.
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # reduction='sum' adds up loss across the whole batch instead of averaging.
            # we'll divide by the total count at the end to get the real per-image average.
            total_loss += nn.functional.nll_loss(output, target, reduction='sum').item()

            pred = output.argmax(dim=1)  # pick the class with the highest score
            correct += pred.eq(target).sum().item()

    n = len(loader.dataset)
    avg_loss = total_loss / n
    accuracy = 100.0 * correct / n
    print(f"  {split_name:5s} - Loss: {avg_loss:.4f}  |  Accuracy: {correct}/{n} ({accuracy:.1f}%)")
    return avg_loss, accuracy


# get_mnist_loaders downloads the MNIST dataset (if needed) and returns
# two DataLoaders: one for training (60k images) and one for testing (10k images).
#
# We shuffle the training set each epoch so the model sees data in a different
# order every time - this helps it generalize better.
# We don't shuffle the test set so results are always the same.
#
# The Normalize transform uses MNIST's known mean (0.1307) and std (0.3081)
# to rescale pixel values - this makes training faster and more stable.
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                         # PIL image [0,255] -> float tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))    # center and scale pixel values
    ])

    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1000, shuffle=False
    )
    return train_loader, test_loader


# plot_training_curves saves a two-panel figure showing loss and accuracy over epochs.
# left panel:  train loss and test loss   (lower = better)
# right panel: train accuracy and test accuracy  (higher = better)
# having both train and test on the same plot makes overfitting easy to spot visually.
def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies,
                         save_path="transformer_training_curves.png"):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses,  'b-o', label='Train Loss')
    ax1.plot(epochs, test_losses,   'r-o', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Test Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, train_accuracies, 'b-o', label='Train Accuracy')
    ax2.plot(epochs, test_accuracies,  'r-o', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training curves saved to: {save_path}")
    plt.close()


# main is the entry point. it reads command-line args, builds the model,
# trains it for the given number of epochs, evaluates on train and test
# after each epoch, saves the weights, and plots the learning curves.
#
# usage: python task4_transformer.py [epochs] [batch_size] [patch_size] [save_path]
# all arguments are optional - defaults are used if not provided
# example: python task4_transformer.py 10 64 7 transformer_mnist.pth
def main(argv):
    # argv[0] is always the script name itself, so actual arguments start at index 1
    num_epochs = int(argv[1]) if len(argv) > 1 else 10
    batch_size = int(argv[2]) if len(argv) > 2 else 64
    patch_size = int(argv[3]) if len(argv) > 3 else 7
    save_path  = argv[4]      if len(argv) > 4 else "transformer_mnist.pth"

    print("=" * 55)
    print("  Task 4 - Vision Transformer for MNIST")
    print("=" * 55)
    print(f"  Epochs:     {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size: {patch_size}x{patch_size}")
    print(f"  Save path:  {save_path}")

    # use GPU if available, otherwise CPU
    # torch.cuda.is_available() checks if a compatible GPU is present
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:     {device}")
    print("=" * 55 + "\n")

    train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

    # build the model and move everything to the right device
    # .to(device) moves all the weight tensors to CPU or GPU memory
    model = NetTransformer(
        img_size=28,
        patch_size=patch_size,
        in_channels=1,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        num_classes=10,
        dropout=0.1
    ).to(device)

    # print the model so we can see every layer - useful for the report
    print("Model architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}\n")

    # Adam adjusts the learning rate per-parameter automatically.
    # lr=0.001 is a solid default starting point for this kind of model.
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # collect metrics each epoch so we can plot them later
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")

        # train for one full pass over the 60k training images
        train_epoch(model, device, train_loader, optimizer, epoch)

        # evaluate on training set - tells us how well we fit the training data
        # (note: model.eval() inside evaluate() disables dropout for a clean read)
        train_loss, train_acc = evaluate(model, device, train_loader, split_name="Train")

        # evaluate on test set - tells us how well we generalize to new images
        test_loss, test_acc = evaluate(model, device, test_loader, split_name="Test")

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print()

    # save the model weights to disk
    # state_dict() is a dictionary of all the learned parameter tensors
    # to load later: model.load_state_dict(torch.load(save_path))
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies)

    return


# this block makes sure main() only runs when we execute this file directly.
# if another file imports this one (like task5 does), main() won't run automatically.
if __name__ == "__main__":
    main(sys.argv)