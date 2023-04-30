import json

import numpy as np


def main():
    gbm_predictions = np.load('submission/gbm_predictions.npy')
    gnn_predictions = np.load('submission/gnn_predictions.npy')
    predictions = (gbm_predictions ** 3.0) * 1.0 + (gnn_predictions ** 3.0) * 0.3
    sorted_predictions = np.flip(np.argsort(predictions, axis=0))

    sorted_predictions = list(map(float, sorted_predictions))

    with open(f'submission/submission.json', mode='w', encoding='utf8') as file_out:
        json.dump(sorted_predictions, file_out)


if __name__ == '__main__':
    main()
