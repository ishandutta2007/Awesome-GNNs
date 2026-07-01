# Edge-Level Link Prediction

## Overview
Relationship forecasting. The network calculates a distance or similarity score between the hidden states of two separate nodes, predicting the likelihood that a connection exists or will form between them.

## Architecture Diagram
```mermaid
graph TD
    A[Node u Embedding] --> C[Link Prediction Layer]
    B[Node v Embedding] --> C
    C -->|Similarity / Inner Product| D[Score / Probability of Edge]
```

## Further Reading
- [Return to Main Index](../README.md)
