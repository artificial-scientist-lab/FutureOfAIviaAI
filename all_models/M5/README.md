# Hand-crafted features + PCA + random forest prediction

Author: Francisco Valente

This is a modified version of the method described [in this report](https://arxiv.org/pdf/2202.03393.pdf) and of its [repository](https://github.com/PFranciscoValente/science4cast_topological) .

Requirements: numpy, scipy, networkx, sklearn

## Files

- **evaluate_model.py**: main script used to run the developed model on the datasets.
- **features_functions.py**: functions used to compute the network features. 
- **model_functions.py**: functions used to compute PCA variables and perform link predicions.
- **utils.py**: additional useful functions, based on the utils.py described in the root of the repository.

## Setup

### Data preparation

Download and unzip the [data files](https://zenodo.org/record/7882892#.ZE-Egx9BwuU) to the `semantic_graphs` folder

### Run model

Run `evaluate_model.py` to evaluate the developed model on the datasets

The AUC results for all models will be stored in models_performance.txt
