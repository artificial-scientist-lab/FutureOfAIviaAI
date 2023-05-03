for delta in 1 3 5
do
  for cutoff in 0 5 25
  do
    for minedge in 1 3
    do
      python preprocess_gbm.py --delta $delta --cutoff $cutoff --minedge $minedge
      python preprocess_gnn.py --delta $delta --cutoff $cutoff --minedge $minedge
      python train_xgboost.py
      python train_lightgbm.py
      python train_gnn.py --delta $delta --cutoff $cutoff --minedge $minedge
      echo "delta ${delta} cutoff ${cutoff} minedge ${minedge}"
      python blend.py
    done
  done
done
