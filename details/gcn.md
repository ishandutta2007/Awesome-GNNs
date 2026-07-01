# Graph Convolutional Networks (GCN)

## Overview
Anisotropic isotropic spatial operator. It computes a localized forward-pass update by taking a normalized average of a target node's immediate neighborhood features, scaling them via a static degree matrix.

## Architecture Diagram
```mermaid
graph TD
    A[Adjacency Matrix A] --> B[Normalized Laplacian D^-1/2 A D^-1/2]
    C[Node Features X] --> D[Feature Transformation XW]
    B --> E[Matrix Multiplication]
    D --> E
    E --> F[Activation Function ReLU]
    F --> G[Updated Node Embeddings]
```

## Further Reading
- [Return to Main Index](../README.md)
