# The Oversmoothing and Oversquashing Stagnation

## Overview
When stacking standard spatial message-passing blocks, the constant averaging of neighbor parameters causes all node hidden vectors to converge on identical numerical states (Oversmoothing), or compresses global topological data into bottlenecked edges (Oversquashing).

## Architecture Diagram
```mermaid
graph TD
    A[Deep Layer Stacking] --> B[Repeated Node Averaging]
    B --> C[Oversmoothing: Embeddings Converge]
    B --> D[Oversquashing: Structural Loss]
```

## Further Reading
- [Return to Main Index](../README.md)
