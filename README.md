# Predicting the Future of AI with AI


**[Predicting the Future of AI with AI: High-quality link prediction in an exponentially growing knowledge network](https://arxiv.org/abs/2210.00881)**\
_Mario Krenn, Lorenzo Buffoni, Bruno Coutinho, Sagi Eppel, Jacob Gates Foster, Andrew Gritsevskiy, Harlin Lee, Yichao Lu, Joao P. Moutinho, Nima Sanjabi, Rishi Sonthalia, Ngoc Mai Tran, Francisco Valente, Yangxinyu Xie, Rose Yu, Michael Kopp_

This page collects results for an AI benchmark for link prediction in exponentially growing knowledge networks. It is a follow-up of the [IEEE BigData Science4Cast competition 2021](https://github.com/iarai/science4cast), described in our [paper](https://arxiv.org/abs/2210.00881).

## Files

- **create_data.py**: A simple python file for creating the datasets SemanticGraph_delta_N_cutoff_M_minedge_P.pkl from the full semantic network all_edges.pkl.
- **evaluate_model.py**: Runs my simple baseline model on all datasets
- **simple_model.py**: My baseline, containing 15 predefined properties that are computed for each unconnected pair. Same model as in the competition.
- **utils.py**: Contains useful functions, such as the creation of datasets from the full semantic network (unbiased for test-set, and biased [i.e. same number of positive and negative solutions] for training if desired), and AUC computation.


Datasets can be downloaded via [zenodo.org](https://zenodo.org/record/7882892#.ZE-Egx9BwuU) (file names: SemanticGraph_delta_N_cutoff_M_minedge_P.pkl).

##
![alt text](
miscellaneous/node_degree_loglog.gif)


## Result of [Yichao's Model (M1)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M1)


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


## Result of [Team HashBrown's Model (M2)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M2)


*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9147      | 0.9175     | 0.9156      |
| *delta=3* | 0.8953      | 0.8977     | 0.8949      |
| *delta=5* | 0.8610      | 0.8645     | 0.8630      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9900      | 0.9876     | 0.9944      |
| *delta=3* | 0.9786      | 0.9861     | 0.9867      |
| *delta=5* | 0.9595      | 0.9689     | 0.9692      |


## Result of [Nima Sanjabi's Model (M3)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M3)


*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8979      | 0.8980     | 0.9010      |
| *delta=3* | 0.8830      | 0.8823     | 0.8823      |
| *delta=5* | 0.8489      | 0.8433     | 0.8409      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9496      | 0.9687     | 0.9481      |
| *delta=3* | 0.9652      | 0.9765     | 0.9788      |
| *delta=5* | 0.9480      | 0.9538     | 0.9488      |



## Results of [Bacalhink Team (M4)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M4)

### Preferential Attachment (M4A)
*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8838      | 0.8862     | 0.8836      |
| *delta=3* | 0.8695      | 0.8673     | 0.8628      |
| *delta=5* | 0.8422      | 0.8359     | 0.8300      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9754      | 0.9649     | 0.9789      |
| *delta=3* | 0.9590      | 0.9620     | 0.9646      |
| *delta=5* | 0.9380      | 0.9442     | 0.9386      |

### Common Neighbours (M4B)
*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8942      | 0.9016     | 0.9009      |
| *delta=3* | 0.8476      | 0.8761     | 0.8783      |
| *delta=5* | 0.7677      | 0.8266     | 0.8345      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9369      | 0.9771     | 0.9889      |
| *delta=3* | 0.9247      | 0.9760     | 0.9786      |
| *delta=5* | 0.8658      | 0.9520     | 0.9526      |



## Result of [Francisco Valente's Model (M5)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M5)

*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* |             |            |             |
| *delta=3* | 0.8467      | 0.8490     | 0.8335      |
| *delta=5* | 0.7897      | 0.8023     | 0.8004      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* |             |            | 0.9819      |
| *delta=3* | 0.9420      | 0.9562     | 0.9461      |
| *delta=5* | 0.8914      | 0.9262     | 0.9150      |



## Result of MK's Baseline (M6)

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



## Resuts of [Andrew Gritsevskiy -- Feature embedding method (M7)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M7)

### Node2Vec (M7A)

*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8768      | 0.8558     | 0.8467      |
| *delta=3* | 0.8361      | 0.5039     | 0.5127      |
| *delta=5* | 0.8755      | 0.6106     | 0.6026      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9258      |  0.9624    | 0.9891      |
| *delta=3* | 0.8648      |  0.5025    | 0.5402      |
| *delta=5* | 0.8573      |  0.6133    | 0.6423      |

### ProNE (M7B)

*Area under the Curve (AUC) for prediction of new edge_weights of 1*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.8354      | 0.8538     | 0.7375      |
| *delta=3* | 0.8210      | 0.7043     | 0.7763      |
| *delta=5* | 0.7383      | 0.7063     | 0.6872      |


*Area under the Curve (AUC) for prediction of new edge_weights of 3*

|           | *cutoff=0*  | *cutoff=5* | *cutoff=25* |
|-----------|-------------|------------|-------------|
| *delta=1* | 0.9952      |  0.9898    | 0.9989      |
| *delta=3* | 0.8844      |  0.9817    | 0.9862      |
| *delta=5* | 0.8586      |  0.8609    | 0.8251      |

## Result of [Team Harlin -- Transformer Method (M8)](https://github.com/artificial-scientist-lab/FutureOfAIviaAI/tree/main/all_models/M8)


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

