# Science4cast2021

## Data Preparation

After cloning this [repo](git@github.com:YichaoLu/Science4cast2021.git), download and extract [data](https://cloud.iarai.ac.at/index.php/s/iTx3bXgMdwsngPn) to the `data` folder:

```
|-- data
|   |-- CompetitionSet2017_3.pkl
|   |-- TrainSet2014_3.pkl
|   |-- TrainSet2014_3_solution.pkl
```

## Usage

To create a submission, run

```
python preprocess_gbm.py
python preprocess_gnn.py
python inference_gbm.py
python inference_gnn.py
python blend.py
```

This creates a submission file named `submission.json` under the `submission` folder.

## Acknowledgements
This repository is based on [science4cast](https://github.com/iarai/science4cast) from [the Institute of Advanced Research in Artificial Intelligence (IARAI)](http://www.iarai.ac.at).
