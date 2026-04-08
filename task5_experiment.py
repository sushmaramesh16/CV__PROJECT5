# Sushma Ramesh & Dina Barua
# CS 5330 - Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
# Task 5: Design Your Own Experiment
#
# The goal is to see how changing different parts of the NetTransformer
# affects accuracy on MNIST. We test four things:
#   1. patch_size  - how big each image patch is
#   2. embed_dim   - how wide each token vector is
#   3. num_layers  - how many transformer blocks we stack
#   4. dropout     - how much regularization we apply
#
# Search strategy: round-robin (one dimension at a time)
#   Instead of testing every possible combination (which would be hundreds of runs),
#   we hold three dimensions fixed and sweep one at a time. After each sweep we
#   keep the best value we found and use it as the new starting point for the next
#   sweep. We repeat this for 3 rounds, giving us ~51 total runs.
#
# Hypotheses (written before running - required by the assignment):
#   H1 patch_size : smaller patches = more tokens = more detail for the transformer.
#                   I expect patch_size=4 to beat patch_size=14.
#   H2 embed_dim  : bigger embed_dim = richer token representations.
#                   I expect accuracy to go up but eventually plateau.
#   H3 num_layers : more layers = deeper processing.
#                   I expect 3-4 to beat 1, but too many might hurt.
#   H4 dropout    : some dropout should prevent overfitting.
#                   I expect 0.1-0.2 to work best; 0.0 may overfit, 0.4 may underfit.


import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# pull in the model we built in Task 4 - no need to redefine it here.
# both files need to be in the same folder for this import to work.
from task4_transformer import NetTransformer


# --- what values do we want to test for each dimension? ---
# patch_size values must divide 28 evenly (4, 7, 14 all work)
# embed_dim values must be divisible by num_heads (4), so 32, 64, 128 all work
SEARCH_SPACE = {
    "patch_size" : [4, 7, 14],
    "embed_dim"  : [32, 64, 128],
    "num_layers" : [1, 2, 3, 4],
    "dropout"    : [0.0, 0.1, 0.2, 0.4],
}

# starting configuration - all sweeps begin from here.
# after each dimension sweep, the best value found replaces the one here.
BASELINE = {
    "patch_size" : 7,
    "embed_dim"  : 64,
    "num_layers" : 2,
    "num_heads"  : 4,      # not being swept - must always divide embed_dim
    "mlp_dim"    : 128,    # not being swept
    "dropout"    : 0.1,
    "num_classes": 10,
}

# 3 epochs per run keeps the experiment fast enough to finish on a CPU.
# it's enough to see which settings work better relative to each other.
EPOCHS_PER_RUN = 3

# 3 rounds means we sweep all 4 dimensions 3 times = ~51 total runs
NUM_ROUNDS = 3


# get_mnist_loaders sets up the MNIST data.
# we load it ONCE and share it across all runs so every configuration
# sees the exact same data - that way any accuracy difference is because
# of the architecture, not the random data order.
def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),                        # PIL image -> float tensor [0,1]
        transforms.Normalize((0.1307,), (0.3081,))   # normalize using MNIST mean and std
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


# train_and_evaluate builds a fresh model using the given config, trains it,
# and returns the final test accuracy.
#
# we build a brand new model for every single run so that weights from a
# previous run can't influence the results of the next one.
#
# config       - dict of hyperparameters for this run
# train_loader - 60k training images
# test_loader  - 10k test images
# device       - cpu or cuda
# epochs       - how many epochs to train
# run_label    - short description string used in the print output
#
# returns test accuracy as a percentage (e.g. 97.5)
def train_and_evaluate(config, train_loader, test_loader, device, epochs, run_label):
    # build a fresh model using whatever hyperparameters this run is testing
    model = NetTransformer(
        img_size=28,
        patch_size=config["patch_size"],
        in_channels=1,
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        mlp_dim=config["mlp_dim"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # training loop - we skip mid-epoch evaluation to keep things fast.
    # we only care about the final test accuracy for comparing configurations.
    for epoch in range(1, epochs + 1):
        model.train()  # turn dropout on
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # always zero gradients before a new batch - PyTorch accumulates them
            # by default, so if we skip this we'd be mixing gradient info between batches
            optimizer.zero_grad()

            output = model(data)                             # forward pass
            loss = nn.functional.nll_loss(output, target)   # compute loss
            loss.backward()                                  # compute gradients
            optimizer.step()                                 # update weights

    # evaluate on the test set after all epochs are done
    model.eval()  # turn dropout off so results are deterministic
    correct = 0

    # no_grad() skips gradient tracking since we're not doing backprop here
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)  # predicted class = highest score
            correct += pred.eq(target).sum().item()

    test_acc = 100.0 * correct / len(test_loader.dataset)
    print(f"  {run_label:45s}  ->  Test Acc: {test_acc:.2f}%")
    return test_acc


# run_sweep tests all the values in SEARCH_SPACE for one dimension,
# keeping every other hyperparameter at its current baseline value.
# returns the value that gave the best accuracy.
#
# dim_name    - which hyperparameter to sweep, e.g. "patch_size"
# values      - the list of values to try, e.g. [4, 7, 14]
# baseline    - current baseline config (everything except dim_name stays the same)
# all_results - running list where we collect every result for later plotting
def run_sweep(dim_name, values, baseline, train_loader, test_loader, device, all_results):
    print(f"\n  -- Sweeping '{dim_name}' over {values} --")

    sweep_accs = []

    for val in values:
        # dict(baseline) makes a copy of the baseline dictionary.
        # this is important because if we just did config = baseline,
        # both names would point to the SAME dictionary object, and
        # changing config[dim_name] would also change the baseline we
        # want to keep fixed for the rest of the sweep.
        config = dict(baseline)
        config[dim_name] = val

        run_label = f"{dim_name}={val}"
        acc = train_and_evaluate(
            config, train_loader, test_loader, device, EPOCHS_PER_RUN, run_label
        )
        sweep_accs.append(acc)

        # save every result so we can plot them all at the end
        all_results.append({
            "dim_name" : dim_name,
            "value"    : val,
            "config"   : dict(config),
            "test_acc" : acc,
        })

    # pick whichever value had the highest accuracy
    best_idx   = sweep_accs.index(max(sweep_accs))
    best_value = values[best_idx]
    print(f"  Best value for '{dim_name}': {best_value}  ({sweep_accs[best_idx]:.2f}%)")
    return best_value, sweep_accs


# plot_sweep_results makes a grid of plots, one per dimension.
# each plot shows accuracy vs. the values we tested for that dimension.
# all three rounds are drawn on the same axes so we can see if results
# change between rounds (they should converge as the baseline improves).
def plot_sweep_results(round_sweep_data, save_path="task5_sweep_results.png"):
    dims = list(SEARCH_SPACE.keys())
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))

    colors = ['b', 'g', 'r']  # one color per round

    for col, dim_name in enumerate(dims):
        ax = axes[col]
        for round_idx, sweep_data in enumerate(round_sweep_data):
            if dim_name not in sweep_data:
                continue
            values, accs = sweep_data[dim_name]
            ax.plot(
                [str(v) for v in values],
                accs,
                f'{colors[round_idx]}-o',
                label=f'Round {round_idx + 1}'
            )

        ax.set_title(f"Effect of '{dim_name}'")
        ax.set_xlabel(dim_name)
        ax.set_ylabel("Test Accuracy (%)")
        ax.legend()
        ax.grid(True)

    plt.suptitle("Round-Robin Hyperparameter Search - NetTransformer on MNIST", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"\nSweep plot saved to: {save_path}")
    plt.close()


# plot_accuracy_over_runs makes a bar chart of test accuracy for every run
# in the order they were run. this gives a good overall picture of the experiment.
def plot_accuracy_over_runs(all_results, save_path="task5_all_runs.png"):
    accs   = [r["test_acc"] for r in all_results]
    labels = [f"{r['dim_name']}={r['value']}" for r in all_results]

    fig, ax = plt.subplots(figsize=(max(12, len(accs) * 0.35), 5))
    ax.bar(range(len(accs)), accs, color='steelblue', edgecolor='white')
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_xlabel("Run (dimension=value tested)")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy Across All Runs")
    ax.axhline(y=max(accs), color='r', linestyle='--', label=f"Best: {max(accs):.2f}%")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    print(f"All-runs plot saved to: {save_path}")
    plt.close()


# print_summary_table prints all runs sorted by accuracy so we can see
# which settings worked best and worst at a glance.
def print_summary_table(all_results):
    print("\n" + "=" * 70)
    print(f"  SUMMARY - all {len(all_results)} runs sorted by test accuracy")
    print("=" * 70)
    print(f"  {'Rank':<5} {'Dimension':<14} {'Value':<10} {'Test Acc':>10}")
    print("  " + "-" * 44)

    # sorted() with reverse=True puts the best result first
    sorted_results = sorted(all_results, key=lambda r: r["test_acc"], reverse=True)

    for rank, result in enumerate(sorted_results, start=1):
        print(f"  {rank:<5} {result['dim_name']:<14} {str(result['value']):<10} {result['test_acc']:>9.2f}%")

    print("=" * 70)
    best  = sorted_results[0]
    worst = sorted_results[-1]
    print(f"  Best run:  {best['dim_name']}={best['value']}  ->  {best['test_acc']:.2f}%")
    print(f"  Worst run: {worst['dim_name']}={worst['value']}  ->  {worst['test_acc']:.2f}%")
    print("=" * 70)


# main runs the full experiment.
# it loads the data, then does NUM_ROUNDS rounds of sweeping all 4 dimensions.
# after each sweep, the baseline is updated to the best value found.
# at the end it prints a summary table and saves the plots.
#
# usage: python task5_experiment.py [epochs_per_run] [num_rounds]
# example: python task5_experiment.py 3 3
def main(argv):
    epochs_per_run = int(argv[1]) if len(argv) > 1 else EPOCHS_PER_RUN
    num_rounds     = int(argv[2]) if len(argv) > 2 else NUM_ROUNDS

    # figure out how many total runs we're doing
    runs_per_round = sum(len(v) for v in SEARCH_SPACE.values())
    total_runs     = runs_per_round * num_rounds

    print("=" * 60)
    print("  Task 5 - Round-Robin Hyperparameter Search")
    print("  Network  : NetTransformer (Task 4)")
    print("  Dataset  : MNIST digits")
    print("=" * 60)
    print(f"  Dimensions to sweep : {list(SEARCH_SPACE.keys())}")
    print(f"  Epochs per run      : {epochs_per_run}")
    print(f"  Rounds              : {num_rounds}")
    print(f"  Runs per round      : {runs_per_round}")
    print(f"  Total runs          : {total_runs}")
    print("=" * 60)

    # print hypotheses before running so they show up in the terminal output
    print("\nHYPOTHESES (written before running):")
    print("  H1 patch_size : expect patch_size=4 to beat patch_size=14 (more detail)")
    print("  H2 embed_dim  : expect accuracy to rise then level off around 128")
    print("  H3 num_layers : expect 3-4 to beat 1, might hurt with too many")
    print("  H4 dropout    : expect 0.1-0.2 to be best; 0.0 overfits, 0.4 underfits\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # load MNIST once and reuse the same loaders for all runs
    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    # dict(BASELINE) makes a copy of BASELINE so we don't modify the original
    baseline = dict(BASELINE)

    all_results    = []   # every run's result goes here
    round_sweep_data = [] # stores (values, accs) per dimension per round for plotting

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {round_num} / {num_rounds}")
        print(f"  Starting baseline: {baseline}")
        print(f"{'='*60}")

        this_round_sweeps = {}

        for dim_name, values in SEARCH_SPACE.items():
            best_value, sweep_accs = run_sweep(
                dim_name, values, baseline,
                train_loader, test_loader, device,
                all_results
            )

            # update baseline so the next dimension sweep in this round
            # benefits from the improvement we just found
            baseline[dim_name] = best_value
            this_round_sweeps[dim_name] = (values, sweep_accs)

        round_sweep_data.append(this_round_sweeps)
        print(f"\n  End of round {round_num}. Updated baseline: {baseline}")

    print_summary_table(all_results)

    print(f"\nFinal best configuration:")
    for k, v in baseline.items():
        print(f"  {k}: {v}")

    plot_sweep_results(round_sweep_data)
    plot_accuracy_over_runs(all_results)

    return


# only run main() when this file is executed directly, not when it's imported
if __name__ == "__main__":
    main(sys.argv)