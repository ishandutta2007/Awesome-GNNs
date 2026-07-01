# The Neighborhood Explosion Memory Wall

## Overview
Because spatial message-passing cascades across successive layers, calculating gradients over a deep GNN forces the model to read an exponentially expanding tree of historical nodes (the neighborhood explosion). This quickly saturates VRAM during training.

## Architecture Diagram
```mermaid
graph TD
    A[Layer L Target Node] --> B[Layer L-1 Neighbors]
    B --> C[Layer L-2 Sub-neighbors]
    C -->|Exponential Growth| D[VRAM Saturation]
```

## Further Reading
- [Return to Main Index](../README.md)
