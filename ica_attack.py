import os
import argparse
import pdb
import numpy as np

from utils import generate_orthogonal_matrix
from ica import FastICA
from scipy.stats import pearsonr
from multiprocessing import Pool, cpu_count
from functools import partial


def one_side_attack(px):
    fast_ica = FastICA(whiten=False, algorithm='deflation', n_components=px.shape[0])
    return fast_ica.fit_transform(px.T).T


def two_size_attack(pxq):
    xq = one_side_attack(pxq)
    return one_side_attack(xq.T).T


def correlation(real_x, rec_x):
    m, n = real_x.shape
    results = []
    if m < n:
        for i in range(real_x.shape[0]):
            with Pool(cpu_count()) as p:
                tmp_result = p.map(partial(pearsonr, real_x[i]), rec_x)
            results.append(max([e[0] for e in tmp_result]))
        print(f'Dim 0 Mean {np.mean(results)} Max {np.max(results)}')
    else:
        for i in range(real_x.shape[1]):
            with Pool(cpu_count()) as p:
                tmp_result = p.map(partial(pearsonr, real_x[:, i]), rec_x.T)
            results.append(max([e[0] for e in tmp_result]))
        print(f'Dim 1 Mean {np.mean(results)} Max {np.max(results)}')
    return results


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--data', '-d', help='dataset', choices=('wine', 'mnist', 'ml100k'), default='wine')
    args_parser.add_argument('--seed', '-s', help='random seed', type=int)
    args = args_parser.parse_args()

    x = np.load(os.path.join('data', args.data + '.npy'))

    print('Data shape', x.shape)

    m, n = x.shape

    np.random.seed(args.seed)

    p = generate_orthogonal_matrix(m, block_reduce=1000)
    q = generate_orthogonal_matrix(n, block_reduce=1000)

    pxq = np.zeros(x.shape)
    index = 0
    for i in range(len(p)):
        pxq[index:index+len(p[i])] = p[i] @ x[index:index+len(p[i])]
        index += len(p[i])

    print('#' * 30)
    print('Attack PX')

    dec_x = one_side_attack(pxq)
    correlation(real_x=x, rec_x=dec_x)

    xq = np.zeros(x.shape)
    index = 0
    for i in range(len(q)):
        xq[:, index:index + len(q[i])] = x[:, index:index + len(q[i])] @ q[i]
        index += len(q[i])

    print('#' * 30)
    print('Attack XQ')

    dec_x = one_side_attack(xq.T).T
    correlation(real_x=x, rec_x=dec_x)

    index = 0
    for i in range(len(q)):
        pxq[:, index:index + len(q[i])] = pxq[:, index:index + len(q[i])] @ q[i]
        index += len(q[i])

    print('#' * 30)
    print('Attack PXQ')

    dec_x = two_size_attack(pxq)
    correlation(real_x=x, rec_x=dec_x)

    print('#' * 30)
    print('Correlation to random data')

    correlation(real_x=x, rec_x=np.random.random(x.shape))
