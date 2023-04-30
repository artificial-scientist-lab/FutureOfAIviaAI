# science4cast_topological

Code used for a proposed solution to the Science4Cast competition (https://www.iarai.ac.at/science4cast/), organized by the Institute of Advanced Research in Artificial Intelligence, and an official competition within the 2021 IEEE BigData Cup Challenges. 
The solution was awarded a Special Prize by the organizers.

Paper: https://arxiv.org/abs/2202.03393

In order to run the proposed approach, the following steps must be followed:
1. run the script "extract_features", to extract the topological features and generate the initial training and evaluation datasets.
2. run the script "pca_transformation", to transform the correlated features  and generate the final training and evaluation datasets.
3. run the script "parameters_optimization", to optimize the prediction models and save their best parameters.
4. run the script "evaluation_prediction", to apply the final model to the evaluation dataset.

The source data can be found here: https://cloud.iarai.ac.at/index.php/s/iTx3bXgMdwsngPn


