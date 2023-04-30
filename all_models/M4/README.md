# Link prediction using simple network metrics
Working repository for the science4cast competition.

In the notebook full_code.ipynb there is all the code to replicate the results of the submissions of team Bacalhink. The data to run the algorithm can be found in the science4cast website https://www.iarai.ac.at/science4cast/.

These methods have been initially inspired by a relaxation of a quantum walker L3 and L2 methods but then departed in favour of other metrics like PA. More details of the approach used can be found in the scientific report.

# Competition submissions

Here is what we have submitted to the leaderboard:

- Bacalhau à Brás: the PA method - performance: 0.89715
- Bacalhau à Gomes de Sá: the AA method - performance: 0.87091
- Bacalhau com Todos: combines AA and PA - performance: 0.91385
- Bacalhau com Natas: PA with time weights - performacne: 0.90364
- Bacalhau à Lagareiro: combines AA and PA with time weights - performance: 0.91853

## Free parameters:

- a: defines the balance in the linear combination of AA with PA
- (θ0, θ1, θ2, θ3): defines the time-weighting function
