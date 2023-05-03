# Link prediction using simple network metrics

This code stems from the original model developed by the Bacalhink team for the Science4Cast competition (https://www.iarai.ac.at/science4cast/), as described in the `Bacalhink Scientific Report.pdf` file, also available here https://arxiv.org/abs/2201.07978.

Authors: João P. Moutinho, Bruno Coutinho, Lorenzo Buffoni

In the original competition the full method included a component based on a Preferential Attachment score (PA) and one based on a Common Neighbours score (CN), with a balance free parameter as well as a link-weighting function. 

For this paper, given that we were working with a much larger dataset, we evaluated only the PA and CN scores separately, corresponding to the models M4A and M4B as described in the Predicting the Future of AI paper.

## Running the models

Besides the main dependencies, our `evaluate_model` script uses also the `multiprocessing` package for parallelization.

The models are defined in `preferential_attachment.py` and `common_neighbours.py`. To run them:
1. Download the datasets following [FutureOfAIviaAI](https://github.com/MarioKrenn6240/FutureOfAIviaAI)
2. Add the `SemanticGraph_delta_{N}_cutoff_{M}_minedge_{P}.pkl` files to the same folder
2. Select one model or the other in `evaluate_model.py`
3. Run `evaluate_model.py`