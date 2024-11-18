# GraphZero: Rethinking Game State Representations in AlphaZero

## Overview
This undergraduate research project explores the use of Graph Attention Networks (GATs) as an alternative to Convolutional Neural Networks in AlphaZero's architecture. The implementation demonstrates the potential of attention-based architectures for game state representation, with experimental results in Chess and Connect4.

## Key Results
- Chess: 31-27 wins (42 draws) in favor of GraphZero vs CNN baseline
- Connect4: 47-46 wins (7 draws) in favor of GraphZero vs CNN baseline
- Comparable computational efficiency to traditional CNN approach
- Training conducted on 8xH100 GPUs over 2-hour periods

## Implementation
The project provides two implementations:
- Python version: https://github.com/vpareek2/pyGraphZero
- CUDA optimized version: https://github.com/vpareek2/GraphZero

## Model Architecture
- GraphZero GAT: 4 layers, 8 attention heads, 256-dimensional representations
- Baseline AlphaZero CNN: 19 residual blocks, 256 channels
More information in Paper.

[Read the full paper here](graphzero.pdf)


