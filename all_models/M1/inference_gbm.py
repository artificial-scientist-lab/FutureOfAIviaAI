import lightgbm as lgb
import numpy as np


def main():
    valid_features = np.load('cache/valid_gbm_features.npy')

    bst = lgb.Booster(model_file=f'model/model.txt')

    valid_predictions = bst.predict(valid_features, num_iteration=4_900)

    np.save('submission/gbm_predictions.npy', valid_predictions)


if __name__ == '__main__':
    main()
