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
| *delta=1* | 0.8520      | 0.8526     | 0.8512      |
| *delta=3* | 0.8411      | 0.8379     | 0.8317      |
| *delta=5* | 0.8201      | 0.8093     | 0.8045      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9473      | 0.9317     | 0.9490      |
| *delta=3* | 0.9408      | 0.9465     | 0.9296      |
| *delta=5* | 0.9055      | 0.9160     | 0.9030      |



## Result of Yichao's Model


*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9330      | 0.9252     | 0.9248      |
| *delta=3* | 0.9172      | 0.9191     | 0.9096      |
| *delta=5* | 0.8960      | 0.8987     | 0.8935      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9926      | 0.9945     | 0.9982      |
| *delta=3* | 0.9853      | 0.9965     | 0.9949      |
| *delta=5* | 0.9793      | 0.9893     | 0.9990      |

## Transformer Method


*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8232      | 0.8253     | 0.8321      |
| *delta=3* | 0.7418      | 0.7659     | 0.7435      |
| *delta=5* | 0.6980      | 0.7023     | 0.6743      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9407      | 0.9373     | 0.9636      |
| *delta=3* | 0.8518      | 0.8804     | 0.8754      |
| *delta=5* | 0.7365      | 0.7977     | 0.7467      |
