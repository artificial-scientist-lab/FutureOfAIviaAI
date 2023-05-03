import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    valid_targets = np.load('cache/valid_targets.npy')

    valid_predictions = np.load('cache/xgb_valid_predictions.npy') ** 3.0
    valid_predictions += np.load('cache/lgb_valid_predictions.npy') ** 3.0
    valid_predictions += (np.load('cache/gnn_valid_predictions_200000.npy') ** 3.0) * 0.3

    print(f'AUC {roc_auc_score(y_true=valid_targets, y_score=valid_predictions)}')


if __name__ == '__main__':
    main()
