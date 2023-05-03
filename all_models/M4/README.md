# Link prediction using simple network metrics

This code stems from the original model developed by the Bacalhink team for the Science4Cast competition (https://www.iarai.ac.at/science4cast/), as described in the `Bacalhink Scientific Report.pdf` file, also available here https://arxiv.org/abs/2201.07978.

In the original competition the full method included a component based on a Preferential Attachment score (PA) and one based on a Common Neighbours score (CN), with a balance free parameter as well as a link-weighting function. 

For this paper, given that we were working with a much larger dataset, we evaluated only the PA and CN scores separately, corresponding to the models M4A and M4B as described in the Predicting the Future of AI paper.

## Running the models

The models are defined in `preferential_attachment.py` and `common_neighbours.py`. To run them:
- Download the datasets
- Select on or the other in `evaluate_model.py`
- Run `evaluate_model.py`