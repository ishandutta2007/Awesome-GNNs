# Graph Attention Networks (GAT)

## Overview
Anisotropic attention-driven operator. It projects query vectors from a target node against key vectors from adjacent neighbors, applying a Softmax normalization to calculate dynamic weighting scalars.

## Architecture Diagram
```mermaid
graph LR
    A[Target Node i] -->|Query| C(Attention Coefficients alpha_ij)
    B[Neighbor Node j] -->|Key| C
    C -->|Softmax Normalized| D[Weighted Aggregation]
    D --> E[Updated Feature h_i']
```

## Further Reading
- [Return to Main Index](../README.md)
