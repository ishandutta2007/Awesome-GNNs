# Node-Level Classification

## Overview
Categorizing individual entities. The model evaluates a node's combined spatial features to predict its target class. A prime example is detecting fraudulent accounts or bot profiles within a social media connection web.

## Architecture Diagram
```mermaid
graph LR
    A[Graph Representation] --> B[GNN Propagation]
    B --> C[Node Embedding h_v]
    C --> D[Classifier / MLP]
    D --> E[Node Label Prediction]
```

## Further Reading
- [Return to Main Index](../README.md)
