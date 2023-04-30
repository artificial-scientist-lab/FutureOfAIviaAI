# Team Hash Brown Science4Cast Submission

This code reproduces Team Hash Brown's (@princengoc, @Xieyangxinyu) best submission (ee5a) for the competition https://www.iarai.ac.at/science4cast

Our team came *second* with a score of 0.92738 (0.01 below the winner). 

Authors: Ngoc Tran and Yangxinyu Xie

## Setup


Easiest way is to clone the directory

```
git clone https://github.com/princengoc/s4s-final
cd s4s-final
```

Getting the data. 

1. Download data files from the competition's organizers: https://www.iarai.ac.at/science4cast/ 
2. Unzip and put the data files in the subfolder `data/raw/`

Run the following to install all the required packages.

```buildoutcfg
pip install -r requirement.txt
```

##### Optional: rerun HOPREC Embedding

You can get *new* HOPREC embedding by running the following shell codes.

```buildoutcfg
cd HOPREC
git submodule add https://github.com/cnclabs/smore
cd smore
make
cd ..
python get_HOPREC_embedding.py --year 2017 --t_min 0.5 --t_max 0.9
python get_HOPREC_embedding.py --year 2017 --t_min 0.5 --t_max 1
```

Since HOPREC is random, this may produce different embeddings, and therefore possibly different cosine scores than what we had. 

Our particular HOPREC embedding is already included under `data/HOPREC/2017_raw_count/`

Scripts to check the differences in cosine similarities between two different HOPREC embeddings are in `HOPREC/check_two_HOPREC_embeddings.py`


### Reproduce the submission

To reproduce the submission file, do

```
cd MLP\ code
python main.py
```

This automatically creates a json file for submission (named after the current git commit hash). The json file and the MLP model parameters are saved under *model_outputs*. 

*utils_common.py* has the function `reproducibility_check`, which eats two json submissions and print out statistics on their agreements and differences. We used this to verify that what we submitted and what is produced by the code above agree in 99.999% out of 1 million entries of the test set. 
