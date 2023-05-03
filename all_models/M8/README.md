# M8

H Lee, R Sonthalia, and JG Foster

Modified version of the transformation method described in [Dynamic Embedding-based Methods for Link Prediction in Machine Learning Semantic Network](https://math.ucla.edu/~harlin/papers/science4cast.pdf).

### Node embedding

We used PecanPy ([Github](https://github.com/krishnanlab/PecanPy), [Citation](https://doi.org/10.1093/bioinformatics/btab202)), a fast implementation of node2vec. 

1. Convert graph information in the `pkl` files to `edg` format with `convert_to_edg.py`.
2. `convert_to_emb.py` writes a bash script that will run multiple PecanPy commands in one go, e.g. `run_pecanpy_all.sh`.

Edit file paths as you need.

### Transformer pre-training and classifier

All transformer-related code is in `TransformerClassifier.ipynb`, which was written and run in Google Colab.

In order to use this code, you will need to edit file paths and/or move `emb` folder from the earlier step.

This script does the following for every combination of delta, cutoff, and min_edges:

1. Aligns node2vec embeddings using [procrustes method as implemented in SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.orthogonal_procrustes.html), and saves it to `Transformer-xxx/embeddings` folder.

2. Pre-trains a transformer model to perform "fill in the blank" tasks with sequences of embedding vectors over the years. The Dataset object randomly masks 30% of the embedding vectors for this step. The model is then saved in `Transformer-xxx/model`.

3. Samples edges in the graph to create training data for the classifier with `create_training_data_biased`. 

4. Trains a small classifier on top of the pre-trained transformer model. Final prediction results on the provided test data is saved in `TransformerClassifier-xxx/`, which is used to calculate AUC.



