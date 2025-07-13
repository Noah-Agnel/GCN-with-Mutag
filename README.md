# GCN-with-Mutag
## Project Overview

This goal project of this project is to classify the molecules in the Mutag dataset with a GCN and a GIN and comapre the results. Given a graph representing a molecule, the goal is to label the molecule as mutagenic or non-mutagenic.

## Results

Afters training each model during 200 epochs, these are the results.

### GCN
The test accuracy at the highest validation epoch is 77.78% with a validation accuracy of 80.00%.

### GIN
The test accuracy at the highest validation epoch is 89.47% with a validation accuracy of 86.67%.

## Conclusion

The GIN architecture works better on the Mutag dataset. This is probably due to the fact the the graphs in the dataset are relatively small (the average number of nodes per graph is 17.9). Therefore the GIN architecture which is much more sensitive to small structural variations will work much better on classifying the molecules than the GCN architecture which will tend to oversmooth and blur the architectural differences.
