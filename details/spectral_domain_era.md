# The Spectral Domain Era

## Overview
Early GNNs defined graph convolutions by migrating to the spectral domain using the Graph Laplacian and Fourier transforms. Kipf and Welling modernized this with the Graph Convolutional Network (GCN, 2016), using a localized first-order Chebyshev polynomial approximation.

## Architecture Diagram
```mermaid
graph TD
    A[Graph Representation] --> B[Fourier Transform via Graph Laplacian]
    B --> C[Spectral Filter Activation]
    C --> D[Inverse Graph Fourier Transform]
    D --> E[Output Signal]
```

## Further Reading
- [Return to Main Index](../README.md)
