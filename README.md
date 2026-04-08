# CS 5330 Project 5: Recognition using Deep Networks

**Authors:** Sushma Ramesh & Dina Barua  
**Course:** CS 5330 - Pattern Recognition and Computer Vision  
**Northeastern University | April 2026**  
**Time Travel Days Used:** 2

---

## Project Description

This project explores deep learning for visual recognition tasks using PyTorch.
We build and train a CNN on the MNIST handwritten digit dataset, examine the
learned filters, test on custom digit images, and apply transfer learning to
classify Greek letters (alpha, beta, gamma) using only 27 training examples.

---

## File Structure

| File | Description |
|------|-------------|
| `task1_train.py` | Build, train and save MNIST CNN (Task 1A-D) |
| `task2_evaluate.py` | Load model, run on first 10 test examples (Task 1E) |
| `task1f_handwritten.py` | Test network on custom digit images (Task 1F) |
| `task2_examine.py` | Examine conv1 filters and their effects (Task 2) |
| `task3_greek.py` | Transfer learning on Greek letters (Task 3) |
| `test_greek.py` | Test Greek model on custom generated symbols |
| `generate_digits.py` | Generate digit test images using Arial font |
| `generate_greek.py` | Generate Greek letter test images using matplotlib |

---

## Setup & Usage

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision matplotlib opencv-python pillow

# Task 1: Train MNIST CNN (generates mnist_model.pth)
python3 task1_train.py

# Task 1E: Run on first 10 test examples
python3 task2_evaluate.py

# Task 1F: Test on custom digits
python3 generate_digits.py
python3 task1f_handwritten.py

# Task 2: Examine network filters
python3 task2_examine.py

# Task 3: Transfer learning on Greek letters
python3 task3_greek.py
python3 test_greek.py

# Task 4: Train Transformer network on MNIST
python3 task4_transformer.py

# Task 5: Run hyperparameter experiment
python3 task5_experiment.py

# Extension: Analyze pretrained ResNet18
python3 extension_pretrained.py
```

---

## Results Summary

| Task | Result |
|------|--------|
| MNIST Test Accuracy (5 epochs) | 98.8% |
| First 10 test examples | 10/10 correct |
| Custom digit images (Arial font) | 4/10 correct |
| Greek letter transfer learning | 27/27 (100%) |
| Custom Greek letter test | 8/9 correct |

---

## Output Files Generated

- `first_six_test.png` - First 6 MNIST test examples
- `training_curves.png` - Training loss and accuracy plots
- `test_predictions.png` - First 9 test predictions grid
- `handwritten_predictions.png` - Custom digit test results
- `conv1_filters.png` - Conv1 filter weight visualization
- `conv1_filterResults.png` - Filter effects on training image
- `greek_training_loss.png` - Greek letter training loss curve
- `greek_custom_results.png` - Custom Greek letter predictions

---

## Dependencies

- Python 3.13+
- PyTorch & torchvision
- OpenCV (opencv-python)
- matplotlib
- Pillow (PIL)
- numpy
