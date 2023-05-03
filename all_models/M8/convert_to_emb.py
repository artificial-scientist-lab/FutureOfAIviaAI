import os

edg_fns = os.listdir('edg')

with open('run_pecanpy_all.sh', 'w') as f:
    f.write('#!/bin/sh')
    f.write('\n')
    for edg in edg_fns:
        emb = os.path.join('emb', edg[:-4] + '.emb')
        if os.path.isfile(emb):
            continue
        s = ('pecanpy --input ' + 'edg/' + edg + ' --output ' + emb + ' --verbose --mode SparseOTF --p 2 --epochs 2')
        f.write(s)
        f.write('\n')

#https://github.com/krishnanlab/PecanPy
