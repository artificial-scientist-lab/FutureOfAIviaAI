This code reproduces Team Hash Brown's (@princengoc, @Xieyangxinyu) model evaluation against multiple benchmarks presented in [FutureOfAIviaAI](https://github.com/MarioKrenn6240/FutureOfAIviaAI).

Our team came *second* in all benchmarks. 

Authors: Ngoc Tran and Yangxinyu Xie

## Setup


Getting the data. 

1. Download data files following instructions [FutureOfAIviaAI](https://github.com/MarioKrenn6240/FutureOfAIviaAI).
2. Unzip and put the data files of the form `SemanticGraph_delta_{N}_cutoff_{M}_minedge_{P}.pkl` in the subfolder `data/`

Run the following to install all the required packages.

```buildoutcfg
mkdir data
cd data
mkdir HOPREC
cd HOPREC 
mkdir 2017_raw_count
cd ..
cd ..
pip install -r requirement.txt
cd MLP\ code/
git submodule add https://github.com/cnclabs/smore
cd smore
make
cd ..
```

### Reproduce the results

To reproduce the results, do

```
cd MLP\ code
python evaluate_model.py
```

This automatically adds the results of our model on each benchmark to `logs.txt`. 

Since HOPREC is random, this may produce different embeddings, and therefore possibly slightly model performance. But we expect the difference to be small. 