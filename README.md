# CS 5330 Project 5: Recognition using Deep Networks

**Authors:** Sushma Ramesh & Dina Barua

## Project Description
Built and trained a CNN on MNIST digits using PyTorch, examined network filters, and applied transfer learning to classify Greek letters (alpha, beta, gamma).

## Files
- `task1_train.py` - Build, train and save MNIST CNN
- `task2_evaluate.py` - Run model on first 10 test examples
- `task2_examine.py` - Examine conv1 filters and effects
- `task3_greek.py` - Transfer learning on Greek letters
- `task1f_handwritten.py` - Test on custom digit images
- `test_greek.py` - Test on custom Greek letter images
- `generate_digits.py` - Generate digit test images
- `generate_greek.py` - Generate Greek letter test images

## Usage
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision matplotlib opencv-python pillow
python3 task1_train.py
```
