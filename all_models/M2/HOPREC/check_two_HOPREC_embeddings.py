import numpy as np
import random
import argparse
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Get 2 embedding')
parser.add_argument('--first', help="folder containing the 1st HOPREC embedding")
parser.add_argument('--second', help="folder containing the 2nd HOPREC embedding")
args = parser.parse_args()


def get_submission(file_name):
    return pickle.load(open(file_name, 'rb'))


def get_diff(first, second):
    pairs = []
    size = 10000
    for i in range(size):
        pair = [random.randrange(0, len(first)), random.randrange(0, len(first))]
        pairs.append(pair)

    first_dist = np.array([np.dot(first[i],first[j]) for i,j in pairs])
    print(f"first: mean: {np.mean(first_dist)} std: {np.std(first_dist)}")
    second_dist = np.array([np.dot(second[i],second[j]) for i,j in pairs])
    print(f"second: mean: {np.mean(second_dist)} std: {np.std(second_dist)}")
    std = np.mean([np.std(first_dist), np.std(second_dist)])

    # Sum of absolute difference between pairs, then divide by N * average std of the two distributions
    return sum(abs(first_dist - second_dist))/(size*std)


if __name__ == "__main__":
    first = get_submission(args.first + '/embedding_2017_0.5_1_raw_count_dim_128.pkl')
    second = get_submission(args.second + '/embedding_2017_0.5_1_raw_count_dim_128.pkl')

    get_diff(first, second)
    diff_list = [get_diff(first, second) for _ in tqdm(range(1000))]

    diff_list = [get_diff(first, second) for _ in tqdm(range(1000))]
    #pickle.dump(diff_list,open('diff_list.pkl', "wb"))
    fig = plt.hist(diff_list, bins='auto')
    plt.savefig("distribution of the average differences.png")
