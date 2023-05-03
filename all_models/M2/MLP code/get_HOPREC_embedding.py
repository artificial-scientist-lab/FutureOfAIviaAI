import pickle
from collections import Counter
import numpy as np
import argparse
import os
from utils_HOPREC import get_field_file, get_embed_dict

parser = argparse.ArgumentParser(description='Get HOPREC Embedding')
parser.add_argument('--year', help='year of data')
parser.add_argument('--t_min',type=float, help='Using all edges after t_min')
parser.add_argument('--t_max',type=float, help='Using all edges before t_max')

args = parser.parse_args()

def generate_hoprec_embedding(yr, t_min, t_max):
    if t_max == 1:
        t_max = int(t_max)

    data_source = '../data/raw/CompetitionSet' + yr + '_3_fractional_time.pkl'
    full_graph,unseen_pairs = pickle.load(open(data_source,'rb'))

    full_graph = np.array(full_graph)
    full_graph = full_graph[full_graph[:,2] <= t_max]
    full_graph = full_graph[full_graph[:,2] >= t_min]

    ls = list(zip(full_graph[:,0],full_graph[:,1]))
    ctr = Counter(ls)
    train_file = f'train_full_{yr}_{t_min}_{t_max}.txt'
    try:
        f = open(train_file, "w")
    except:
        open(train_file, "x")
        f = open(train_file, "w")
    finally:
        for item in ctr.items():
            f.write(f'{int(item[0][0])} {int(item[0][1])} {item[1]}\n')
        f.close()

    field_file = 'field.txt'
    get_field_file(train_file, field_file)

    embed_file = 'embed'
    dim = 128
    embed_out_file = f'../data/HOPREC/2017_raw_count/embedding_{yr}_{t_min}_{t_max}_raw_count_dim_{dim}'

    print('Running HOPREC...')
    os.system(f'./smore/cli/hoprec -train {train_file} -save {embed_file} -field {field_file} -dimensions {dim} -sample_times 500 -threads 8')

    print('Getting embed_dict...')
    get_embed_dict(embed_file, embed_out_file, dim)

if __name__ == '__main__':
    yr = args.year
    t_min = args.t_min
    t_max = args.t_max
    generate_hoprec_embedding(yr, t_min, t_max)