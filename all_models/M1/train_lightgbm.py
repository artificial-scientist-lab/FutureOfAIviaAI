from multiprocessing import cpu_count

import lightgbm as lgb
import numpy as np

from time_utils import Timer


def main():
    train_features = np.load('cache/train_features.npy')
    train_targets = np.load('cache/train_targets.npy')
    valid_features = np.load('cache/valid_features.npy')

    lgb_train = lgb.Dataset(data=train_features, label=train_targets)

    params = {
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,
        'num_leaves': 2 ** 4,
        'max_depth': 4,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'feature_fraction': 0.9,
        'force_row_wise': 'true',
        'device_type': 'cpu',
        'num_threads': cpu_count() // 2,
        'seed': 0,
        'verbosity': 0
    }

    with Timer('lgb_train'):
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_train]
        )

    valid_predictions = gbm.predict(data=valid_features)

    np.save('cache/lgb_valid_predictions.npy', valid_predictions)


if __name__ == '__main__':
    main()
