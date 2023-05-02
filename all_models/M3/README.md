# Model M3: LSTM + 5 graph features in 3 years

The code calculates graph features and feed them to an LSTM. Then uses the trained model on an evaluation set.
Computing the features requires calculation of the square of the adjacency matrix. The code breaks the calculations in 100 steps, each time for a section of all of the nodes, which makes the computation feasible with limited hardware resources.
There are 18 datasets, to test the model on all the possible combinations of 3 values for 𝛿, 3 values for cutoff, and 2 values for edge strengh.

## Running the code
The datasets must be downloaded from [https://zenodo.org/record/7882892#.ZFBSZHZBxPa] to the root directory of the notebook. Then the notebook trains and evaluates the model on each of the 18 datasets and stores the scores in a log file.
