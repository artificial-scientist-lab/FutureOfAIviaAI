import numpy as np
import xgboost as xgb


def main():
    train_features = np.load('cache/train_features.npy')
    train_targets = np.load('cache/train_targets.npy')
    valid_features = np.load('cache/valid_features.npy')

    dtrain = xgb.DMatrix(data=train_features, label=train_targets)
    dvalid = xgb.DMatrix(data=valid_features)

    params = {
        'booster': 'gbtree',
        'eta': 0.01,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 0.9,
        'tree_method': 'gpu_hist',
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'seed': 0
    }
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=10_000,
        evals=[(dtrain, 'train')]
    )

    valid_predictions = bst.predict(dvalid)

    np.save('cache/xgb_valid_predictions.npy', valid_predictions)


if __name__ == '__main__':
    main()
