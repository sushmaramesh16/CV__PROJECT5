# Sushma Ramesh & Dina Barua
# CS 5330 - Pattern Recognition and Computer Vision
# Project 5: Recognition using Deep Networks
# Helper: Generate Greek letter test images using matplotlib
import matplotlib.pyplot as plt
import matplotlib
import os
import numpy as np
from PIL import Image

os.makedirs('my_greek', exist_ok=True)

symbols = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'gamma': r'$\gamma$'}

for name, symbol in symbols.items():
    fig, ax = plt.subplots(1, 1, figsize=(1.5, 1.5))
    ax.text(0.5, 0.5, symbol, fontsize=80, ha='center', va='center',
            transform=ax.transAxes, color='black')
    ax.set_facecolor('white')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'my_greek/{name}_test.png', dpi=100,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Generated {name}_test.png')

print('Done!')
