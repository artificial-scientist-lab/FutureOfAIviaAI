# Predicting the Future of AI with AI

## Files

- **create_data.py**: A simple python file for creating the datasets SemanticGraph_delta_N_cutoff_M_minedge_P.pkl from the full semantic network all_edges.pkl.
- **evaluate_model.py**: Runs my simple baseline model on all datasets
- **simple_model.py**: My baseline, containing 15 predefined properties that are computed for each unconnected pair. Same model as in the competition.
- **utils.py**: Contains useful functions, such as the creation of datasets from the full semantic network (unbiased for test-set, and biased [i.e. same number of positive and negative solutions] for training if desired), and AUC computation.


Datasets can be downloaded via [dropbox](https://www.dropbox.com/sh/3wlm7njhkxiyw08/AADvetI5VeLfa26c7Wci8kMka?dl=0) (file names: SemanticGraph_delta_N_cutoff_M_minedge_P.pkl).


## Result of MK's Baseline



- Prediction from Year (2021-delta,2021), with delta=[1,3,5]
- Minimal Vertex Degree: cutoff=[0,5,25]
- Prediction from unconnceted to edge_weight=[1,3] edges

*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8538      | 0.8551     | 0.8531      |
| *delta=3* | 0.8439      | 0.8389     | 0.8330      |
| *delta=5* | 0.8081      | 0.8022     | 0.7908      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9399      | 0.9331     | 0.9529      |
| *delta=3* | 0.9340      | 0.9455     | 0.9359      |
| *delta=5* | 0.8978      | 0.9002     | 0.8855      |
