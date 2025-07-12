# GCN-with-Mutag
## Project Overview

This project uses the MUTAG dataset to train a GCN with the task of graph classification. Given a graph representing a molecule, the goal is to lab el the molecule as mutagenic or non-mutagenic.

## Results

The highest test accuracy obtaining throughout testing and hyper parameter tuning was 86% which is an encouraging result. However, implenting a GIN instead of a GCN might be more suitable to avoid oversmoothing on small graphs such as the ones in the MUTAG dataset.
